from math import ceil
import random
import numpy as np


class DataStream:
    def __init__(self, batch_size, train_proportion):
        self.batch_size = batch_size
        self.train_proportion = train_proportion
        self.current_batch_num = 0

    def split_by_patch_id(self, x, y):
        unique_patches = x.index.get_level_values('patch_id').unique().tolist()
        train_patches = random.sample(unique_patches, round(len(unique_patches)*self.train_proportion))
        x_train = np.stack(x.loc[x.index.get_level_values('patch_id').isin(train_patches)].values)
        y_train = y.loc[y.index.get_level_values('patch_id').isin(train_patches)].values
        train_generator = self.get_single_generator(x_train, y_train, self.batch_size)

        test_patches = [patch for patch in unique_patches if patch not in train_patches]
        x_test = np.stack(x.loc[x.index.get_level_values('patch_id').isin(test_patches)].values)
        y_test = y.loc[y.index.get_level_values('patch_id').isin(test_patches)].values
        test_generator = self.get_single_generator(x_test, y_test, self.batch_size)
        return train_generator, test_generator, test_patches

    def get_single_generator(self, x, y, batch_size):
        num_batches = ceil(len(y) / batch_size)
        while True:
            for i in range(num_batches):
                batch = (
                    x[(i*batch_size): ((i+1)*self.batch_size)].astype(float),
                    y[(i*batch_size): ((i+1)*self.batch_size)].astype(float)
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
