from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential


class CNN:
    def __init__(self, convolutional_layers, learning_rate=0.1, num_classes=2, weight_positives=1):
        self.num_classes = num_classes
        self.convolutional_layers = convolutional_layers
        self.model = self._create_model()

    def _create_model(self):
        layers = []
        for config in self.convolutional_layers:
            layers.append(self._create_convolutional_and_pooling_layer(**config))

        layers.extend([
            Flatten(),
            Dense(units=64),
            BatchNormalization(),
            Dense(units=self.num_classes, activation='softmax'),
        ])
        model = Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def fit(self, x, y, epochs):
        self.model.fit_generator(x, y, epochs=epochs)

    def fit_generator(self, generator, epochs, steps_per_epoch):
        self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

    @staticmethod
    def _create_convolutional_and_pooling_layer(filters, kernel_size, pool_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu')
        pool = MaxPooling2D(pool_size=pool_size, strides=pool_size[0])
        return pool
