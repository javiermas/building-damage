from abc import abstractmethod, ABC
import tensorflow as tf


class ImageClassificationGraph(ABC):

    def __init__(self, learning_rate, image_shape, weight_positives):
        self.learning_rate = learning_rate
        self.image_shape = image_shape
        self.weight_positives = weight_positives
        self._create_graph()

    def _create_graph(self):
        tf.reset_default_graph()
        self._create_placeholders()
        self._create_model()
        self._create_predictions()
        self._create_loss()
        self._create_train_operation()

    def _create_placeholders(self):
        shape_x = [None, self.image_shape[0], self.image_shape[1], self.image_shape[2]]
        self.x = tf.placeholder(tf.float32, shape_x)
        self.y = tf.placeholder(tf.int64, [None])

    @abstractmethod
    def _create_model(self):
        pass

    def _create_predictions(self):
        self.probabilities = tf.nn.softmax(self.logits, name='probabilities')
        self.predictions = tf.to_int64(tf.argmax(self.probabilities, 1, name='predictions'))

    def _create_loss(self):
        weights = (self.weight_positives*tf.to_float(self.y > 0)) + \
            ((1 - self.weight_positives)*tf.to_float(self.y < 1))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.probabilities, weights=weights)
        self.loss = tf.reduce_mean(loss)

    def _create_train_operation(self):
        self.train_operation = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
