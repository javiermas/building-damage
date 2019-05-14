from math import ceil
import random
import numpy as np


class DataStream:
    def __init__(self, batch_size, test_proportion):
        self.batch_size = batch_size
        self.test_proportion = test_proportion
        self.current_batch_num = 0

    def split(self, x, y):
        train_index = random.sample(range(len(y)), round(len(y)*0.8))
        train_generator = self.get_single_generator(x[train_index], y[train_index], self.batch_size)
        test_generator = self.get_single_generator(np.delete(x, train_index, axis=0),
                                                   np.delete(y, train_index, axis=0),
                                                   len(y) - len(train_index))
        return train_generator, test_generator

    def get_single_generator(self, x, y, batch_size):
        num_batches = ceil(len(y) / batch_size)
        while True:
            for i in range(num_batches):
                batch = (
                    x[(i*batch_size): ((i+1)*self.batch_size)],
                    y[(i*batch_size): ((i+1)*self.batch_size)]
                )
                yield batch

    def get_batch(self, x, y):
        x_batch = x[(self.current_batch_num*self.batch_size):
                    ((self.current_batch_num+1)*self.batch_size)]
        y_batch = y[(self.current_batch_num*self.batch_size):
                    ((self.current_batch_num+1)*self.batch_size)]
        return x_batch, y_batch

    def update(self):
        self.current_batch_num += 1

    def restart(self):
        self.current_batch_num = 0
