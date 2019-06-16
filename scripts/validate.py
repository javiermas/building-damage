import os
import argparse
import json
from math import ceil
from time import time
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset

from damage.data import DataStream
from damage.models import CNN, RandomSearch

parser = argparse.ArgumentParser()
parser.add_argument('features')
parser.add_argument('--gpu')
args = vars(parser.parse_args())


os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
RESULTS_PATH = 'logs/experiments'
FEATURES_PATH = 'logs/features'
features_file_name = args.get('features')

# Reading
features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name)).dropna(subset=['destroyed'])
#features_destroyed = features.loc[features['destroyed'] == 1].sample(500)
#features_non_destroyed = features.loc[features['destroyed'] == 0].sample(5000)
#features = pd.concat([features_destroyed, features_non_destroyed])

#### Modelling
sampler = RandomSearch()
Model = CNN
spaces = sampler.sample_cnn(50)
# Do splits
class_proportion = {
    1: 0.3,
}
batch_size = spaces[0]['batch_size']
test_batch_size = 500
data_stream = DataStream(batch_size=batch_size, train_proportion=0.7,
                         class_proportion=class_proportion, test_batch_size=test_batch_size)
num_batches = ceil(len(features) / batch_size)
num_batches_test = ceil(len(test_indices)/test_batch_size)
train_index_generator, test_index_generator = data_stream.split_by_patch_id(features[['image']],
                                                                            features[['destroyed']])
train_generator = data_stream.get_train_data_generator_from_index(
    [features['image'], features['destroyed']], train_index_generator)

train_dataset = Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int32))
test_indices = list(test_index_generator)
test_generator = data_stream.get_train_data_generator_from_index(
    [features['image'], features['destroyed']], test_indices)
test_dataset = Dataset.from_generator(lambda: test_generator, (tf.float32, tf.int32))
for space in spaces:
    space['batch_size'] = batch_size
    space['class_weight'] = {
        0: (class_proportion[1] +.1),
        1: 1 - (class_proportion[1] +.1),
    }
    print('Now validating:\n')
    print(space)
    model = Model(**space)
    losses = model.validate_generator(train_dataset, test_dataset,
                                      steps_per_epoch=num_batches,
                                      validation_steps=num_batches_test,
                                      **space)
    losses['model'] = str(Model)
    losses['space'] = space
    losses['features'] = features_file_name
    losses['num_batches_train'] = num_batches
    losses['num_batches_test'] = num_batches_test
    with open('{}/experiment_{}.json'.format(RESULTS_PATH, round(time())), 'w') as f:
        json.dump(str(losses), f)
