from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from damage.models.losses import *
from damage.models.base import Model


class CNN(Model):

    metrics = ['accuracy', recall, specificity, precision, negatives,
               positives, sensitivity_specificity_average]

    def __init__(self, convolutional_layers, dense_units=64, learning_rate=0.1, layer_type='cnn', **kwargs):
        self.convolutional_layers = convolutional_layers
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.layer_type = layer_type
        self.layer_funcs = {
            'cnn': self._create_convolutional_and_pooling_layer_cnn,
            'vgg': self._create_convolutional_and_pooling_layer_vgg,
        }
        self._create_convolutional_and_pooling_layer = self.layer_funcs[self.layer_type]
        self.model = self._create_model()

    def fit_generator(self, generator, epochs, steps_per_epoch, class_weight=None, **kwargs):
        self.model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                 class_weight=class_weight)

    def validate_generator(self, train_generator, test_generator, validation_steps,
                           epochs, steps_per_epoch, class_weight=None, **kwargs):
        model_fit = self.model.fit_generator(train_generator, validation_data=test_generator,
                                             epochs=epochs, validation_steps=validation_steps,
                                             steps_per_epoch=steps_per_epoch, class_weight=class_weight)
        return model_fit.history

    def predict_generator(self, generator, **kwargs):
        return self.model.predict_generator(generator, **kwargs)

    def _create_model(self):
        layers = []
        for config in self.convolutional_layers:
            layers.extend(self._create_convolutional_and_pooling_layer(**config))

        layers.extend([
            Flatten(),
            Dense(units=self.dense_units),
            BatchNormalization(),
            Dense(units=1, activation='relu'),
        ])
        model = Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy', learning_rate=self.learning_rate,
                      metrics=self.metrics)
        return model

    @staticmethod
    def _create_convolutional_and_pooling_layer_vgg(filters, kernel_size, pool_size, dropout=0):
        conv_0 = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu')
        conv_1 = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu')
        pool = MaxPooling2D(pool_size=pool_size, strides=1)
        dropout = Dropout(dropout)
        return [conv_0, conv_1, pool, dropout]

    @staticmethod
    def _create_convolutional_and_pooling_layer_cnn(filters, kernel_size, pool_size, dropout=0):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu')
        pool = MaxPooling2D(pool_size=pool_size, strides=1)
        dropout = Dropout(dropout)
        return [conv, pool, dropout]
