import random
import numpy as np


class RandomSearch:

    def sample_cnn(self, T):
        spaces = [self._sample_single_cnn_space() for _ in range(T)]
        return spaces

    @staticmethod
    def _sample_single_cnn_space():
        num_layers = random.choice(range(1, 5))
        convolutional_layers = []
        for _ in range(num_layers):
            kernel_size = random.choice(range(1, 15))
            pool_size = min(kernel_size-1, random.choice(range(1, 10)))
            layer = {
                'kernel_size': [kernel_size, kernel_size],
                'pool_size': [pool_size, pool_size],
                'filters': random.choice(range(10, 300)),
            }
            convolutional_layers.append(layer)

        space = {
            'learning_rate': random.choice(np.geomspace(1e-2, 1)),
            'batch_size': random.choice(range(100, 300)),
            'convolutional_layers': convolutional_layers,
            'epochs': random.choice(range(5, 15)),
        }
        return space
