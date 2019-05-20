from math import ceil
import random
import numpy as np


class DataStream:
    def __init__(self, batch_size, train_proportion):
        self.batch_size = batch_size
        self.train_proportion = train_proportion

    def split_by_patch_id(self, x):
        unique_patches = x.index.get_level_values('patch_id').unique().tolist()
        train_patches = random.sample(unique_patches, round(len(unique_patches)*self.train_proportion))
        test_patches = [patch for patch in unique_patches if patch not in train_patches]
        train_index = x.loc[x.index.get_level_values('patch_id').isin(train_patches)].index
        test_index = x.loc[x.index.get_level_values('patch_id').isin(test_patches)].index
        train_index_generator = self._get_index_generator(train_index)
        test_index_generator = self._get_index_generator(test_index)
        return train_index_generator, test_index_generator

    def _get_index_generator(self, index):
        num_batches = ceil(len(index) / self.batch_size)
        for i in range(num_batches):
            index_batch = index[(i*self.batch_size): ((i+1)*self.batch_size)]
            yield index_batch

    @staticmethod
    def get_train_data_generator_from_index(features, target, index):
        while True:
            for _index in index:
                batch = (
                    np.stack(features.loc[_index]).astype(float),
                    target.loc[_index].values
                )
                yield batch
    
    @staticmethod
    def get_test_data_generator_from_index(features, index):
        for _index in index:
            batch = np.stack(features.loc[_index]).astype(float)
            yield batch
