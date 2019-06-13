import random
import numpy as np


class RandomSearch:

    def sample_cnn(self, T):
        spaces = [self._sample_single_cnn_space() for _ in range(T)]
        return spaces

    @staticmethod
    def _sample_single_cnn_space():
        num_layers = random.choice(range(3, 5))
        convolutional_layers = []
        filters = random.choice([8, 16, 32])
        kernel_size = random.choice([3, 5, 7, 9])
        pool_size = kernel_size-1
        dropout = random.choice(np.linspace(0.1, 0.8, 10))
        for _ in range(num_layers):
            filters = filters*2
            layer = {
                'kernel_size': [kernel_size, kernel_size],
                'pool_size': [pool_size, pool_size],
                'filters': filters,
                'dropout': dropout,
            }
            convolutional_layers.append(layer)

        space = {
            'learning_rate': random.choice(np.geomspace(1e-3, 1)),
            'batch_size': random.choice(range(100, 150)),
            'convolutional_layers': convolutional_layers,
            'epochs': random.choice(range(5, 25)),
        }
        return space
