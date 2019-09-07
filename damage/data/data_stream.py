from math import ceil
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


class DataStream:
    def __init__(self, batch_size, test_batch_size, train_proportion, class_proportion=None):
        self.batch_size = batch_size
        self.test_batch_size= test_batch_size
        self.train_proportion = train_proportion
        self.class_proportion = class_proportion

    def split_by_patch_id(self, x, y):
        data = x.join(y)
        unique_patches = data.index.get_level_values('patch_id').unique().tolist()
        train_patches = random.sample(unique_patches, round(len(unique_patches)*self.train_proportion))
        train_data = data.loc[data.index.get_level_values('patch_id').isin(train_patches)]
        if self.class_proportion is not None:
            train_data = self._upsample_class_proportion(train_data, self.class_proportion).sample(frac=1)

        test_patches = list(set(unique_patches) - set(train_patches))
        test_data = data.loc[data.index.get_level_values('patch_id').isin(test_patches)]
        train_index_generator = self._get_index_generator(train_data, self.batch_size)
        test_index_generator = self._get_index_generator(test_data, self.test_batch_size)
        return train_index_generator, test_index_generator
    
    @staticmethod
    def _get_index_generator(features, batch_size, Splitter=StratifiedKFold):
        num_batches = ceil(len(features) / batch_size)
        splitter = Splitter(n_splits=num_batches)
        batches = splitter.split(features['destroyed'], features['destroyed'])
        batches = list(map(lambda x: x[1], batches))
        return batches

    def get_train_data_generator_from_index(
        self,
        data,
        index,
        augment_flip,
        augment_brightness):
        while True:
            for _index in index:
                features = data[0].iloc[_index].values
                target = np.stack(data[1].iloc[_index])
                if augment_flip:
                    features = [self._flip(image) for image in features]
                elif augment_brightness:
                    features = [self._augment_brightness(image) for image in features]

                features = np.stack(features)
                yield features, target
    
    @staticmethod
    def _flip(image):
        flipping_funcs = [
            lambda image: image,
            lambda image: np.fliplr(image),
            lambda image: np.flipud(image),
            lambda image: np.fliplr(np.flipud(image)),
        ]
        func = random.choice(flipping_funcs)
        return func(image)
    
    @staticmethod
    def _augment_brightness(image):
        alpha = random.choice(np.linspace(0.85, 1.4))
        image_augmented = image * alpha
        return image_augmented
    
    @staticmethod
    def get_test_data_generator_from_index(data, index):
        while True:
            for _index in index:
                batch = np.stack(data.iloc[_index]) 
                yield batch
    
    @staticmethod
    def _upsample_class_proportion(data, proportion):
        current_proportion = data['destroyed'].mean()
        assert proportion[1] > current_proportion
        data_non_destroyed = data.loc[data['destroyed'] == 0]
        data_destroyed = data.loc[data['destroyed'] == 1]
        data_destroyed_upsampled = data_destroyed.sample(
            int(len(data_non_destroyed)*proportion[1]),
            replace=True
        )
        data_resampled = pd.concat([data_non_destroyed, data_destroyed_upsampled])
        return data_resampled
