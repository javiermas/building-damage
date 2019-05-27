from math import ceil
import random
import numpy as np


class DataStream:
    def __init__(self, batch_size, train_proportion, class_proportion=None):
        self.batch_size = batch_size
        self.train_proportion = train_proportion
        self.class_proportion = class_proportion

    def split_by_patch_id(self, x, y):
        unique_patches = x.index.get_level_values('patch_id').unique().tolist()
        train_patches = random.sample(unique_patches, round(len(unique_patches)*self.train_proportion))
        test_patches = list(set(unique_patches) - set(train_patches))
        if self.class_proportion is not None:
            train_patches = self._upsample_class_proportion_in_patches(y, train_patches)

        train_index = x.loc[x.index.get_level_values('patch_id').isin(train_patches)].index
        test_index = x.loc[x.index.get_level_values('patch_id').isin(test_patches)].index
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
    def get_data_generator_from_index(data, index):
        for _index in index:
            batch = tuple([np.stack(dataframe.loc[_index]) * 1.0 for dataframe in data])
            yield batch

    def _upsample_class_proportion_in_patches(self, y, patches):
        for level, proportion in self.class_proportion.items():
            level_patches = y.loc[y == level].index.get_level_values('patch_id').unique().tolist()
            if not level_patches:
                raise ValueError('No patches found with class {}'.format(level))
            level_patches_current = [patch for patch in level_patches if patch in patches]
            proportion_current = len(level_patches_current)/len(patches)
            if proportion_current > proportion:
                raise NotImplementedError('Downsampling not implemented')

            # This doesn't take into consideration that the total amount of patches
            # will change, so we will not get the actual proportion
            n_patches_to_add = round((proportion - proportion_current)*len(patches))
            patches_to_add = [random.choice(level_patches_current) for _ in range(n_patches_to_add)]
            patches += patches_to_add

        return patches
