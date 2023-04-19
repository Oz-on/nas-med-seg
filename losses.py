"""
Author: Oskar Domingos
File containing loss functions
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def jaccard_coef(y_true, y_pred):
    epsilon = K.epsilon()

    y_true_f = K.flatten(tf.cast(y_true, dtype=tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, dtype=tf.float32))

    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + epsilon) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + epsilon)


def jaccard_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)


def dice_coef(y_true, y_pred):
    epsilon = K.epsilon()

    y_true_f = K.flatten(tf.cast(y_true, dtype=tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, dtype=tf.float32))

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    dice = (2 * intersection + epsilon) / (union + epsilon)

    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def binary_dual_focal_loss(y_true, y_pred, alpha=0.55, beta=3, gamma=2, phi=1):
    """
    This is an implementation of Dual Focal Loss function for binary classes. It was proposed by Hossain et al. 2021
    link to the paper: https://www.sciencedirect.com/science/article/abs/pii/S0925231221011310

    :param y_true: tensor of true labels for the image
    :param y_pred: tensor of the predictions for the labels
    :param alpha: adjustable parameter
    :param beta: adjustable parameter
    :param gamma: adjustable parameter
    :param phi: adjustable parameter
    :return: tensor of dual focal loss for every input pixel which will be reduced to singular value by Loss class
    """
    epsilon = K.epsilon()  # Value which is very small used to avoid division by 0 or log(0).

    # Convert true labels and predicted labels to float32
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    focal_factor_modified = alpha * (
                tf.pow(tf.math.abs(y_true - y_pred), gamma) + tf.pow(tf.math.abs((1 - y_true) - (1 - y_pred)), gamma))
    regularization_term_modified = beta * (
                (1 - y_true) * tf.math.log(phi - y_pred + epsilon) + y_true * tf.math.log(phi - (1 - y_pred) + epsilon))

    # Reduce the dimensionality by using reduce sum function
    # such that the shape of these tensors are (num_of_slices x w x h x 1)
    focal_factor_modified = tf.reduce_sum(focal_factor_modified, axis=-1)
    regularization_term_modified = tf.reduce_sum(regularization_term_modified, axis=-1)

    return bce - focal_factor_modified - regularization_term_modified


@tf.keras.utils.register_keras_serializable()
class BinaryDualFocalLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        :param y_true: tensor-like Binary (0 or 1) class labels
        :param y_pred: tensor-like probabilities for the positive class.
            The shapes of 'y_ture' and 'y_pred' should be broadcastable.
        :return: Tensor - per-example focal loss. Reduction to a scalar is handled by __call__ method
        """

        return binary_dual_focal_loss(y_true=y_true, y_pred=y_pred)
