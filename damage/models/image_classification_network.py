import numpy as np
import tensorflow as tf
from damage.data import DataStream


class ImageClassificationNetwork:

    def __init__(self, graph):
        self._check_graph_validity(graph)
        self.graph = graph

    def fit(self, x, y, epochs, batch_size):
        num_batches = round(len(y)/batch_size)
        self.session = tf.Session()
        tf.global_variables_initializer().run(session=self.session)
        data_stream = DataStream(x, y, batch_size)
        global_loss, global_probabilities = [], []
        print('===Training===')
        for epoch in range(epochs):
            epoch_loss, epoch_probabilities = [], []
            data_stream.restart()
            for batch in range(num_batches):
                x_batch, y_batch = data_stream.get_batch()
                feed_dict = {self.graph.x: x_batch, self.graph.y: y_batch}
                operations = [self.graph.probabilities, self.graph.loss, self.graph.train_operation]
                probabilities, loss, _ = self.session.run(operations, feed_dict=feed_dict)
                epoch_loss.append(round(loss, 2))
                epoch_probabilities.extend(probabilities)
                data_stream.update()
                print(f'batch {batch} loss {epoch_loss[-1]}')

            global_loss.extend(epoch_loss)
            global_probabilities.append(epoch_probabilities)
            print(f'--- epoch {epoch} loss {np.mean(epoch_loss):.2f} --- ')

        self.global_loss = global_loss
        self.global_probabilities = global_probabilities

    def predict(self, x):
        probabilities = self.session.run([self.graph.probabilities], feed_dict={self.graph.x: x})
        return probabilities

    @staticmethod
    def _check_graph_validity(graph):
        for attribute in ['x', 'y', 'train_operation', 'probabilities', 'loss']:
            if not (hasattr(graph, attribute)):
                raise AttributeError(f'Graph missing attribute {attribute}')
