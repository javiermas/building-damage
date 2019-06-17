import random
import numpy as np


class RandomSearch:

    def sample_cnn(self, T):
        spaces = [self._sample_single_cnn_space() for _ in range(T)]
        return spaces

    def sample_cnn_pretrained(self, T):
        spaces = [self._sample_single_cnn_pretrained_space() for _ in range(T)]
        return spaces

    @staticmethod
    def _sample_single_cnn_space():
        num_layers = random.choice(range(2, 7))
        convolutional_layers = []
        filters = random.choice([8, 16, 32, 64])
        kernel_size = random.choice([3, 5, 7, 9])
        pool_size = kernel_size-1
        dropout = random.choice(np.linspace(0.1, 0.8, 10))
        total_filters = filters
        for _ in range(num_layers):
            filters = filters*2
            layer = {
                'kernel_size': [kernel_size, kernel_size],
                'pool_size': [pool_size, pool_size],
                'filters': filters,
                'dropout': dropout,
            }
            convolutional_layers.append(layer)
            total_filters += filters

        space = {
            'dense_units': random.choice([16, 32, 64, 128, 256]),
            'learning_rate': random.choice(np.geomspace(1e-3, 1)),
            'batch_size': random.choice(range(50, 100)),
            'convolutional_layers': convolutional_layers,
            'epochs': random.choice(range(5, 15)),
            'layer_type': random.choice(['cnn', 'vgg']),
        }
        if total_filters > 1800:
            space = RandomSearch._sample_single_cnn_space()

        return space

    @staticmethod
    def _sample_single_cnn_pretrained_space():
        space = {
            'dense_units': random.choice([16, 32, 64, 128, 256]),
            'learning_rate': random.choice(np.geomspace(1e-3, 1)),
            'batch_size': random.choice(range(50, 100)),
            'epochs': random.choice(range(5, 15)),
        }
        return space
