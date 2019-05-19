from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential

from damage.models.losses import (precision, recall, true_positives, true_negatives,
                                  false_positives, false_negatives, positives, negatives)
from damage.models.base import Model


class CNN(Model):

    def __init__(self, convolutional_layers, dense_units=64, learning_rate=0.1,
                 num_classes=2, loss_weights=None, **kwargs):
        self.convolutional_layers = convolutional_layers
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.model = self._create_model()

    def fit_generator(self, generator, epochs, steps_per_epoch, **kwargs):
        self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def validate_generator(self, train_generator, test_generator, validation_steps,
                           epochs, steps_per_epoch, class_weight=None, **kwargs):
        model_fit = self.model.fit_generator(train_generator, validation_data=test_generator,
                                             epochs=epochs, validation_steps=validation_steps,
                                             steps_per_epoch=steps_per_epoch, class_weight=class_weight)
        return model_fit.history

    def predict_generator(self, generator, **kwargs):
        return self.model.predict_generator(generator)

    def _create_model(self):
        layers = []
        for config in self.convolutional_layers:
            layers.append(self._create_convolutional_and_pooling_layer(**config))

        layers.extend([
            Flatten(),
            Dense(units=self.dense_units),
            BatchNormalization(),
            Dense(units=self.num_classes, activation='softmax'),
        ])
        model = Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy', learning_rate=self.learning_rate,
                      metrics=['accuracy', precision, recall, true_positives, true_negatives,
                               false_negatives, false_positives, positives, negatives],
                      loss_weights=self.loss_weights)
        return model

    @staticmethod
    def _create_convolutional_and_pooling_layer(filters, kernel_size, pool_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu')
        pool = MaxPooling2D(pool_size=pool_size, strides=1)
        return pool
