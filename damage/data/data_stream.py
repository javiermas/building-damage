from math import ceil
import random
import numpy as np


class DataStream:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.current_batch_num = 0
        self.num_batches = ceil(len(self.y) / batch_size)

    def get_generators(self, test_proportion=.8):
        train_index = random.sample(range(len(self.y)), round(len(self.y)*0.8))
        train_generator = self.get_single_generator(self.x[train_index], self.y[train_index], self.batch_size)
        test_generator = self.get_single_generator(np.delete(self.x, train_index, axis=0),
                                                   np.delete(self.y, train_index, axis=0),
                                                   len(self.y) - len(train_index))
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
    
    def get_batch(self):
        x_batch = self.x[(self.current_batch_num*self.batch_size):
                         ((self.current_batch_num+1)*self.batch_size)]
        y_batch = self.y[(self.current_batch_num*self.batch_size):
                         ((self.current_batch_num+1)*self.batch_size)]
        return x_batch, y_batch

    def update(self):
        self.current_batch_num += 1

    def restart(self):
        self.current_batch_num = 0
