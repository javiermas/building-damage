import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from damage.models.network_graph import ImageClassificationGraph


class CNNGraph(ImageClassificationGraph):

    def __init__(self, convolutional_layers, image_shape, learning_rate=0.1, num_classes=2, weight_positives=1):
        self.num_classes = num_classes
        self.convolutional_layers = convolutional_layers
        super().__init__(image_shape=image_shape, learning_rate=learning_rate, weight_positives=weight_positives)

    def _create_model(self):
        layers = []
        for config in self.convolutional_layers:
            if not layers:
                layers.append(self._create_convolutional_and_pooling_layer(inputs=self.x, **config))
            else:
                layers.append(self._create_convolutional_and_pooling_layer(inputs=layers[-1], **config))

        layer_flat = self._flatten_layer(layers[-1])
        logits = Dense(inputs=layer_flat, units=self.num_classes, activation=tf.nn.relu)
        self.logits = logits

    @staticmethod
    def _create_convolutional_and_pooling_layer(inputs, filters, kernel_size, pool_size):
        conv = Conv2D(inputs=inputs, filters=filters, kernel_size=kernel_size,
                      padding="same", activation=tf.nn.relu)
        pool = tf.layers.MaxPooling2D(inputs=conv, pool_size=pool_size, strides=pool_size[0])
        return pool

    @staticmethod
    def _flatten_layer(layer):
        return Flatten()(layer)
