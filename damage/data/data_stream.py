from math import ceil
import random
import numpy as np
import pandas as pd


class DataStream:
    def __init__(self, batch_size, train_proportion, class_proportion=None):
        self.batch_size = batch_size
        self.train_proportion = train_proportion
        self.class_proportion = class_proportion

    def split_by_patch_id(self, x, y):
        data = x.join(y)
        unique_patches = data.index.get_level_values('patch_id').unique().tolist()
        train_patches = random.sample(unique_patches, round(len(unique_patches)*self.train_proportion))
        train_data = data.loc[data.index.get_level_values('patch_id').isin(train_patches)]
        if self.class_proportion is not None:
            train_data = self._upsample_class_proportion(train_data).sample(frac=1)
            train_index = train_data.index

        test_patches = list(set(unique_patches) - set(train_patches))
        test_index = data.loc[data.index.get_level_values('patch_id').isin(test_patches)].index
        train_index_generator = self._get_train_index_generator(train_index)
        test_index_generator = self._get_test_index_generator(test_index)
        return train_index_generator, test_index_generator

    def _get_train_index_generator(self, index):
        num_batches = ceil(len(index) / self.batch_size)
        while True:
            for i in range(num_batches):
                index_batch = index[(i*self.batch_size): ((i+1)*self.batch_size)]
                yield index_batch

    def _get_test_index_generator(self, index):
        num_batches = ceil(len(index) / self.batch_size)
        for i in range(num_batches):
            index_batch = index[(i*self.batch_size): ((i+1)*self.batch_size)]
            yield index_batch

    @staticmethod
    def get_train_data_generator_from_index(data, index):
        for _index in index:
            batch = tuple([np.stack(dataframe.loc[_index]) for dataframe in data])
            yield batch

    @staticmethod
    def get_test_data_generator_from_index(data, index):
        for _index in index:
            batch = np.stack(data.loc[_index]) 
            yield batch

    def _upsample_class_proportion(self, data):
        current_proportion = data['destroyed'].mean()
        assert self.class_proportion[1] > current_proportion
        sampling_proportion = {1: self.class_proportion[1]/current_proportion, 0: 1}
        data_positives = data.loc[data['destroyed'] == 1].sample(frac=sampling_proportion[1]-1, replace=True)
        data_resampled = pd.concat([data_positives, data]) 
        return data_resampled
