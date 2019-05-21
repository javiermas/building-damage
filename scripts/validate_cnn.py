import os
import argparse
import json
from math import ceil
from time import time
import pandas as pd

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
features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name))

#### Modelling
sampler = RandomSearch()
Model = CNN
spaces = sampler.sample_cnn(10)
for space in spaces:
    space['epochs'] = 1
    data_stream = DataStream(batch_size=space['batch_size'], train_proportion=0.8)
    num_batches = ceil(len(features) / space['batch_size'])
    train_index_generator, test_index_generator = data_stream.split_by_patch_id(features['image'])
    train_generator = data_stream.get_data_generator_from_index(
        [features['image'], features['destroyed']], train_index_generator)
    test_indices = list(test_index_generator)
    test_generator = data_stream.get_data_generator_from_index(
        [features['image'], features['destroyed']], test_indices)
    space['class_weight'] = {
        0: features['destroyed'].mean(),
        1: 1 - features['destroyed'].mean(),
    }
    model = Model(**space)
    losses = model.validate_generator(train_generator, test_generator,
                                      steps_per_epoch=num_batches,
                                      validation_steps=1,
                                      **space)
    losses['model'] = str(Model)
    losses['space'] = space
    losses['features'] = features_file_name
    with open('{}/experiment_{}.json'.format(RESULTS_PATH, round(time())), 'w') as f:
        json.dump(str(losses), f)
