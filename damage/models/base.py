from abc import ABC, abstractmethod

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout


class CNNModel(ABC):

    def __init__(self):
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

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
