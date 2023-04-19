"""
Author: Oskar Domingos
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random

from tensorflow import keras
from tensorflow.keras import layers
from typing import Union

from read_functions import load_data
from losses import BinaryDualFocalLoss, dice_coef, jaccard_coef
from utils import constrain


class NasMedSeg:
    _BATCH_SIZE: int
    _REDUCED_BATCH_SIZE: int
    _EPOCHS: int
    _OPTIMIZER: str

    _inputs: tf.keras.Input

    _train_ds: Union[None, tf.data.Dataset]
    _validation_ds: Union[None, tf.data.Dataset]
    _train_ds_reduced: Union[None, tf.data.Dataset]
    _validation_ds_reduced: Union[None, tf.data.Dataset]

    _operations = operations = [
        [1, 3, "relu"],
        [1, 3, "mish"],
        [1, 3, "IN", "relu"],
        [1, 3, "IN", "mish"],
        [1, 5, "relu"],
        [1, 5, "mish"],
        [1, 5, "IN", "relu"],
        [1, 5, "IN", "mish"],
        [0, 3, "relu"],
        [0, 3, "mish"],
        [0, 3, "IN", "relu"],
        [0, 3, "IN", "mish"],
        [0, 5, "relu"],
        [0, 5, "mish"],
        [0, 5, "IN", "relu"],
        [0, 5, "IN", "mish"],
    ]

    _connections = {
        1: [[0]],
        2: [[0], [1]],
        3: [[0], [1], [2], [1, 2]],
        4: [[0], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]],
        5: [
            [0], [1], [2], [3], [4], [1, 2], [1, 3], [1, 4],
            [2, 3], [2, 4], [3, 4], [1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2], [1, 2, 3, 4]
        ]
    }

    _augmentation_techniques = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomFlip("vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ]

    def __init__(self,
                 input_shape: tuple,
                 batch_size: int = 20,
                 reduced_batch_size: int = 16,
                 epochs: int = 25,
                 optimizer: str = 'adam'):

        self._inputs = tf.keras.Input(input_shape)

        self._BATCH_SIZE = batch_size
        self._REDUCED_BATCH_SIZE = reduced_batch_size
        self._EPOCHS = epochs
        self._OPTIMIZER = optimizer

        self._train_ds = None
        self._validation_ds = None
        self._train_ds_reduced = None
        self._validation_ds_reduced = None

    def __initialise_fly(self, population: np.ndarray, fly_idx: int) -> None:
        for i in range(7):
            population[fly_idx, i, 0] = random.randint(0, len(self._operations) - 1)
            population[fly_idx, i, 1] = random.randint(0, len(self._connections[1]) - 1)
            population[fly_idx, i, 2] = random.randint(0, len(self._connections[2]) - 1)
            population[fly_idx, i, 3] = random.randint(0, len(self._connections[3]) - 1)
            population[fly_idx, i, 4] = random.randint(0, len(self._connections[4]) - 1)
            population[fly_idx, i, 5] = random.randint(0, len(self._connections[5]) - 1)

    def __initialise_population(self, pop_size: int) -> np.ndarray:
        """It generates population of specified size"""

        population = np.empty((pop_size, 7, 6))
        for i in range(pop_size):
            self.__initialise_fly(population, i)

        return population.astype(int)

    def __update_fly(self,
                     new_population: np.ndarray,
                     population: np.ndarray,
                     fly_idx: int,
                     best_neighbour_i: int,
                     best_fly_i: int):
        """It updates fly"""

        for i in range(7):
            # Firstly update first position from the column - it indicates the operation used in every node
            new_population[fly_idx, i, 0] = constrain(
                min_v=0,
                max_v=len(self._operations) - 1,
                value=np.round(population[best_neighbour_i, i, 0] + np.random.uniform() * (
                            population[best_fly_i, i, 0] - population[fly_idx, i, 0])).astype(int)
            )

            # Then update 5 positions representing blocks of the architecture
            for j in range(1, 6):
                new_population[fly_idx, i, j] = constrain(
                    min_v=0,
                    max_v=len(self._connections[j]) - 1,
                    value=np.round(population[best_neighbour_i, i, j] + np.random.uniform() * (
                                population[best_fly_i, i, j] - population[fly_idx, i, j])).astype(int)
                )

    def __apply_operations(self, node: tf.keras.layers.Layer, operation: list, node_name: str, filters: tuple):
        """It applies operations specified by the first number of the block vector"""
        conv = layers.Conv2D(
            filters,
            operation[1],
            padding="same",
            name=node_name,
        )

        # Choose activation function
        if operation[2] == 'relu' or (len(operation) == 4 and operation[3] == 'relu'):
            activation = layers.ReLU()
        elif operation[2] == 'mish' or (len(operation) == 4 and operation[3] == 'mish'):
            activation = tfa.activations.mish
        else:
            raise ValueError()

        # Choose normalization_function
        normalization = None
        if operation[2] == 'IN':
            normalization = tfa.layers.InstanceNormalization(
                axis=3,
                center=True,
                scale=True,
                beta_initializer="random_uniform",
                gamma_initializer="random_uniform")

        if operation[0]:
            # If convolution is firstly applied
            node = conv(node)
            if normalization:
                node = normalization(node)
            node = activation(node)
        else:
            # If activation function is firstly applied
            if normalization:
                node = normalization(node)
            node = activation(node)
            node = conv(node)

        return node

    def __generate_block(self,
                         block_num: int,
                         block_params: list[int],
                         inputs: tf.keras.Input,
                         initial_filters: int,
                         skip_connection: tf.keras.layers.Layer | None):
        operation_type, *connections_combinations = block_params
        # Firstly extract operation type
        operation = self._operations[operation_type]

        # First four blocks form an encoder part
        if block_num <= 3:
            filters = initial_filters * 2 ** block_num
            initial_node = layers.Conv2D(
                filters,
                operation[1],
                name=f'block_{block_num}_initial_node',
                padding="same"
            )(inputs)
        # The next three blocks form a decoder part
        else:
            filters = initial_filters * 2 ** (6 - block_num)
            initial_node = layers.Conv2DTranspose(
                filters,
                operation[1],
                name=f'block_{block_num}_initial_node',
                padding="same",
                strides=(2, 2)
            )(inputs)
            initial_node = layers.add([initial_node, skip_connection])

        # Nodes within the block
        nodes = [initial_node]

        # Iterate over fly from position 1 to n to extract connections between nodes
        for curr_node_i, v_i in enumerate(connections_combinations):
            # Firstly get list of possible combinations of connections for specific node
            possible_connections = self._connections[curr_node_i + 1]

            # Then get connection combination
            connection_combination = possible_connections[v_i]

            # Generate new node
            if len(connection_combination) == 1:
                new_node = nodes[connection_combination[0]]
            else:
                new_node = layers.Add()([nodes[node_num] for node_num in connection_combination])

            new_node = self.__apply_operations(
                new_node,
                operation,
                f'block_{block_num}_node_{curr_node_i + 1}',
                filters
            )
            nodes.append(new_node)

        # Final block results from the last element in the list of nodes
        block = nodes[-1]

        # Each block is always followed by max pooling or up sampling depending
        # On block_type
        if block_num <= 3:
            # before performing pooling operations in terms of decoder block, store block
            # to use it in decoder block as a skip connection
            skip_connection = block
            if block_num < 3:
                block = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block)
        else:
            skip_connection = None
            # block = layers.UpSampling2D(size=(2,2), strides=(2,2))(block)

        return block, skip_connection

    def __map_fly_to_architecture(self, fly: np.ndarray, inputs: tf.keras.Input):
        """
        It decodes particular agent (fly) neural architecture represented by that fly
        """

        # Initial number of filters is set to 16
        # Each with every stage, deeper number of filters doubles
        initial_filters = 16

        blocks = [
            self.__generate_block(
                block_num=0,
                block_params=fly[0],
                inputs=inputs,
                initial_filters=initial_filters,
                skip_connection=None
            )
        ]

        for k in range(1, fly.shape[0]):
            new_block, skip_connection = self.__generate_block(
                block_num=k,
                block_params=fly[k],
                inputs=blocks[-1][0],
                initial_filters=initial_filters,
                skip_connection=blocks[fly.shape[0] - (k + 1)][1] if k > 3 else None
            )

            blocks.append([new_block, skip_connection])

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(blocks[-1][0])
        return outputs

    def __cost_function(self, fly: np.ndarray) -> float:
        """It maps the fly to the architecture and the creates and trains model based on that architecture"""

        # Create model and compile it using adam optimiser and F1 loss function
        architecture = self.__map_fly_to_architecture(fly, self._inputs)

        model = keras.Model(inputs=self._inputs, outputs=architecture)

        model.compile(
            optimizer=self._OPTIMIZER,
            loss=BinaryDualFocalLoss(),
            metrics=[dice_coef, jaccard_coef, 'accuracy']
        )

        # Train the model
        model.fit(
            self._train_ds_reduced,
            validation_data=self._validation_ds_reduced,
            epochs=self._EPOCHS,
            batch_size=self._BATCH_SIZE
        )

        # Evaluate the model using Dice coefficient. Use Dice Coefficient as a cost function
        return model.get_metrics_result()['dice_coef'].numpy()

    def __dfo(self, pop_size: int = 10, iter_num: int = 20, delta: float = 0.001) -> np.ndarray:
        """It uses the Dispersive Flies Optimization algorithm to find the best architecture"""
        # Initialise population
        population = self.__initialise_population(pop_size)

        # Initially set best fly to -1
        best_fly_i = -1

        # Main loop
        for i in range(iter_num):
            print(f'iteration: {i}')

            # List of fitness values
            f_list = list(map(lambda fly: self.__cost_function(fly), population))

            # Index of best fly
            best_fly_i = np.argmax(f_list)
            print(f'best fly: {best_fly_i}')

            # Create list of next interation population
            next_iter_pop = np.empty((pop_size, 7, 6), dtype=np.int64)

            # Update every fly accordingly
            for curr_fly_i in range(pop_size):

                # if best fly is the current fly, copy that fly
                # and leave it for the next iteration
                if curr_fly_i == best_fly_i:
                    next_iter_pop[curr_fly_i] = population[best_fly_i]

                else:
                    # Find best neighbour according to the fitness values
                    best_neighbour_i = np.argmax([
                        f_list[(curr_fly_i - 1) % pop_size],
                        f_list[(curr_fly_i + 1) % pop_size]
                    ])

                    if np.random.uniform() < delta:
                        # In case of disturbance assign random fly
                        self.__initialise_fly(next_iter_pop, curr_fly_i)
                    else:
                        # Update fly
                        self.__update_fly(next_iter_pop, population, curr_fly_i, best_neighbour_i, best_fly_i)

            # set next iteration population as global population
            population = next_iter_pop

        return population[best_fly_i]

    def __adjust_augmentation(self, best_fly):
        """It iteratively finds the best augmentation technique out of the existing list"""
        best_score = 0
        best_technique = None

        for augmentation_technique in self._augmentation_techniques:
            augmented_inputs = augmentation_technique(self._inputs)

            architecture = self.__map_fly_to_architecture(best_fly, augmented_inputs)

            model = tf.keras.Model(inputs=self._inputs, outputs=architecture)

            model.compile(
                optimizer=self._OPTIMIZER,
                loss=BinaryDualFocalLoss(),
                metrics=[dice_coef, jaccard_coef, 'accuracy']
            )

            model.fit(
                self._train_ds,
                validation_data=self._validation_ds,
                epochs=self._EPOCHS,
                batch_size=self._BATCH_SIZE
            )

            dice_coef_val = model.get_metrics_result()['dice_coef'].numpy()
            if dice_coef_val > best_score:
                best_technique = augmentation_technique

        return best_technique

    def load(self, dataset_path: str, extension: str, depth: int) -> None:
        """It loads the data to the NAS object. Note that is usually take some time"""
        (train_ds, validation_ds), (train_ds_reduced, validation_ds_reduced) = load_data(
            dataset_path,
            extension,
            depth,
            self._BATCH_SIZE,
            self._REDUCED_BATCH_SIZE,
        )

        self._train_ds = train_ds
        self._validation_ds = validation_ds
        self._train_ds_reduced = train_ds_reduced
        self._validation_ds_reduced = validation_ds_reduced

    def search(self, pop_size: int, iter_num: int, delta: float = 0.001) -> tf.keras.Model:
        """It searches for the best architecture and best augmentation method.
         Then, function returns the compiled model based on that architecture"""

        best_fly = self.__dfo(pop_size, iter_num, delta)

        best_augmentation = self.__adjust_augmentation(best_fly)

        augmented_inputs = best_augmentation(self._inputs)

        best_architecture = self.__map_fly_to_architecture(best_fly, augmented_inputs)

        model = tf.keras.Model(inputs=self._inputs, outputs=best_architecture)
        model.compile(
            optimizer=self._OPTIMIZER,
            loss=BinaryDualFocalLoss(),
            metrics=[dice_coef, jaccard_coef, 'accuracy']
        )

        return model
