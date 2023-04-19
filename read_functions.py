import tarfile
import zipfile
import os

import numpy as np
import nibabel as nib
import tensorflow as tf

from IPython.display import clear_output


def read_nifti_file(filepath: str):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def load_scans_raw(scans_paths: list[str], limit=None):
    """It loads raw scans into memory"""

    scans = []
    n = len(scans_paths) if limit is None or limit > len(scans_paths) else limit
    i = 0
    while i in range(n):
        clear_output(wait=True)

        scans.append(read_nifti_file(scans_paths[i]))
        print(f'scans: {i+1} / {n}')
        i += 1

    return scans


def normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize the volume using min max normalization"""
    min_val = -1000
    max_val = 400

    volume[volume < min_val] = min_val
    volume[volume > max_val] = max_val
    volume = (volume - min_val) / (max_val - min_val)
    volume = volume.astype("float32")
    return volume


def prepare_mask(mask: np.ndarray) -> np.ndarray:
    """It prepares the mask by casting values to 0 or 1"""
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask.astype('int32')


def resize_volume(img: np.ndarray, depth: int) -> np.ndarray:
    """Resize across z-axis"""

    # Depth should not exceed the current depth
    desired_depth = depth  # Depth is also number of slices in this case
    desired_width = 256
    desired_height = 256
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]

    if desired_depth > current_depth:
        raise ValueError()

    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def preprocess_raw_scans(scan_paths: list[str], mask: bool, depth: int, limit: int = None) -> np.ndarray:
    """ It takes the list of raw volumes (scans), and preprocess them"""

    scans_raw = load_scans_raw(scan_paths, limit)
    scans_preprocessed = []

    for scan_raw in scans_raw:
        if mask:
            scan_preprocessed = prepare_mask(scan_raw)
        else:
            scan_preprocessed = normalize(scan_raw)

        scan_preprocessed = resize_volume(scan_preprocessed, depth)
        scans_preprocessed.append(scan_preprocessed)

    scans_preprocessed = np.array(scans_preprocessed)

    # Combine slices into one array. There should be limit x 64 slices in total.
    total_slices = np.concatenate((scans_preprocessed), axis=-1)

    # (number of slices) x 256 x 256 instead of 256 x 256 x (number of slices)
    total_slices = total_slices.T
    return total_slices


def split_train_val(slices: np.ndarray, slices_with_masks: np.ndarray):
    train_test_ratio = 0.8
    train_size = int(train_test_ratio * slices.shape[0])

    x_train = slices[:train_size, :, :]
    x_val = slices[train_size:, :, :]

    y_train = slices_with_masks[:train_size, :, :]
    y_val = slices_with_masks[train_size:, :, :]

    # Add one more dimension at the end, such that the shape looks like this:
    # (num_of_images, width, height, 1)
    x_train = x_train[..., tf.newaxis]
    y_train = y_train[..., tf.newaxis]
    x_val = x_val[..., tf.newaxis]
    y_val = y_val[..., tf.newaxis]

    return (x_train, y_train), (x_val, y_val)


def prepare_data_batches(x: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    """It takes X and y and creates batches"""
    # Create dataset object
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    # Cache dataset
    ds = ds.cache()
    # Shuffle the data to prevent over-fitting caused by sequences
    ds = ds.shuffle(buffer_size=3600, seed=222)
    # Create a batch of specified size
    ds = ds.batch(batch_size)
    # Prefetch
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def add_sample_weights(image: tf.Tensor, label: tf.Tensor):
    """It adds weights to the classes to reduce class inbalance"""

    # The weights for each class, with the constraint that:
    # sum(class_weights) == 1.0
    class_weights = tf.constant([1.0, 10.0])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the class_weights
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


def get_data_with_non_zero_masks(slices: np.ndarray, masks: np.ndarray):
    """It returns slices and masks pairs where masks are non-zero"""

    mask_sums = np.sum(masks, axis=(1, 2))
    nonzero_masks_indices = np.where(mask_sums > 0)

    according_slices = slices[nonzero_masks_indices]
    nonzero_masks = masks[nonzero_masks_indices]

    return according_slices, nonzero_masks


def load_data(dataset_path: str, extension: str, depth: int, batch_size: int, reduced_batch_size: int):
    target_folder_path = f'./{dataset_path.split("/")[-1].split(".")[0]}'

    volumes_folder_path = os.path.join(target_folder_path, 'imagesTr')
    masks_folder_path = os.path.join(target_folder_path, 'labelsTr')

    # Extract data from compressed file
    print('Extracting files from compressed file')
    if extension == 'zip':
        with zipfile.ZipFile(dataset_path, 'r') as z_fp:
            z_fp.extractall('./')
    else:
        with tarfile.TarFile(dataset_path, 'r') as t_fp:
            t_fp.extractall('./')

    print('Extraction complete')

    volume_paths = sorted([os.path.join(volumes_folder_path, x) for x in os.listdir(volumes_folder_path) if not x.startswith('.')])
    masks_paths = sorted([os.path.join(masks_folder_path, x) for x in os.listdir(masks_folder_path) if not x.startswith('.')])

    # Now load raw slices (volumes and segmentations)
    slices = preprocess_raw_scans(volume_paths, mask=False, depth=depth)
    masks = preprocess_raw_scans(masks_paths, mask=True, depth=depth)

    reduced_slices, reduced_masks = get_data_with_non_zero_masks(slices, masks)

    # Split slices and its masks to train and validation datasets
    (x_train, y_train), (x_val, y_val) = split_train_val(slices, masks)
    (x_train_reduced, y_train_reduced), (x_val_reduced, y_val_reduced) = split_train_val(reduced_slices, reduced_masks)

    # Create batches given batch sizes for full and reduced ds
    train_ds = prepare_data_batches(x_train, y_train, batch_size)
    validation_ds = prepare_data_batches(x_val, y_val, batch_size)

    train_ds_reduced = prepare_data_batches(x_train_reduced, y_train_reduced, reduced_batch_size)
    validation_ds_reduced = prepare_data_batches(x_val_reduced, y_val_reduced, reduced_batch_size)

    return (train_ds, validation_ds), (train_ds_reduced, validation_ds_reduced)
