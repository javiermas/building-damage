import os
import pickle
from math import ceil
from time import time
import argparse
import logging
from functools import reduce
import pandas as pd

from damage.models import CNN
from damage.data import DataStream, load_experiment_results


parser = argparse.ArgumentParser()
parser.add_argument('features')
parser.add_argument('--gpu')
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
RESULTS_PATH = 'logs/predictions'
FEATURES_PATH = 'logs/features'
features_file_name = args.get('features')

# Reading
#features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name))

#### Modelling
Model = CNN

# Choose space
experiment_results = load_experiment_results()
experiment_results_single_model = experiment_results.loc[experiment_results['model'] == str(Model)]
space = experiment_results_single_model.loc[experiment_results_single_model['id'].idxmax(), 'space']
class_proportion = {
    1: 0.4,
}
space['class_weight'] = {
    0: class_proportion[1],
    1: 1 - class_proportion[1],
}
space['epochs'] = 50
# Get data generators
'''
data_stream = DataStream(batch_size=space['batch_size'], train_proportion=0.6,
                         class_proportion=class_proportion)
train_index_generator, test_index_generator = data_stream.split_by_patch_id(features['image'], features['destroyed'])
train_generator = data_stream.get_data_generator_from_index([features['image'], features['destroyed']],
                                                            train_index_generator)
test_indices = list(test_index_generator)
test_generator = data_stream.get_data_generator_from_index([features['image']], test_indices)
'''
train_path = '{}/{}/train'.format(FEATURES_PATH, features_file_name)
test_path = '{}/{}/test'.format(FEATURES_PATH, features_file_name)
train_generator = DataStream.get_data_generator_from_path(train_path)
train_generator = (batch for batch in train_generator for epoch in range(space['epochs']))
test_generator = DataStream.get_data_generator_from_path(test_path)
with open('{}/{}/train/index.p'.format(FEATURES_PATH, features_file_name), 'rb') as f:
    train_indices = pickle.load(f)
with open('{}/{}/test/index.p'.format(FEATURES_PATH, features_file_name), 'rb') as f:
    test_indices = pickle.load(f)

# Fit model and predict
model = Model(**space)
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_indices),
                    validation_steps=1,
                    **space)

predictions = model.predict_generator(test_generator, steps=len(test_indices))
predictions = pd.DataFrame({
    'prediction': predictions[:, 1],
}, index=reduce(lambda l, r: l.union(r), test_indices))
file_name = '{}/prediction_{}.p'.format(RESULTS_PATH, round(time()))
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
