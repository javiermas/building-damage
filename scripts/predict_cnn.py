import os
from math import ceil
from time import time
import argparse
import logging
from functools import reduce
import pandas as pd

from damage.models import CNN
from damage.data import DataStream, load_experiment_results
from damage import features

   
parser = argparse.ArgumentParser()
parser.add_argument('features')
parser.add_argument('--gpu')
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
RESULTS_PATH = 'logs/predictions'
FEATURES_PATH = 'logs/features'
features_file_name = args.get('features')

# Reading
features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name))

#### Modelling
Model = CNN

experiment_results = load_experiment_results()
experiment_results_single_model = experiment_results.loc[experiment_results['model'] == str(Model)]
space = experiment_results_single_model.loc[experiment_results_single_model['id'].idxmax(), 'space']
space['class_weight'] = {
    0: features['destroyed'].mean(),
    1: 1 - features['destroyed'].mean(),
}

data_stream = DataStream(batch_size=space['batch_size'], train_proportion=0.6)
train_index_generator, test_index_generator = data_stream.split_by_patch_id(features['image'])
train_generator = data_stream.get_data_generator_from_index([features['image'], features['destroyed']],
                                                            train_index_generator)
test_indices = list(test_index_generator)
test_generator = data_stream.get_data_generator_from_index([features['image']], test_indices)

num_batches = ceil(len(features) / space['batch_size'])
model = Model(**space)
model.fit_generator(train_generator,
                    steps_per_epoch=num_batches,
                    validation_steps=1,
                    **space)

predictions = model.predict_generator(test_generator, steps=len(test_indices))
predictions = pd.DataFrame({
    'prediction': predictions[:, 1],
}, index=reduce(lambda l, r: l.union(r), test_indices))
file_name = '{}/prediction_{}.p'.format(RESULTS_PATH, round(time()))
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
