import os
from math import ceil
from time import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from damage.models import CNN, RandomSearch
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
features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name)).dropna(subset=['destroyed'])
#features_destroyed = features.loc[features['destroyed'] == 1].sample(1000)
#features_non_destroyed = features.loc[features['destroyed'] == 0].sample(1000)
#features = pd.concat([features_destroyed, features_non_destroyed])

#### Modelling
Model = CNN

# Choose space
experiment_results = load_experiment_results()
if not experiment_results.empty:
    experiment_results['val_loss_mean'] = experiment_results['val_loss'].apply(np.mean)
    experiment_results_single_model = experiment_results.loc[experiment_results['model'] == str(Model)]
    space = experiment_results_single_model.loc[
        experiment_results_single_model['val_loss_mean'].idxmax(), 'space']
else:
    space = RandomSearch._sample_single_cnn_space()

class_proportion = {
    1: 0.3,
}
space['class_weight'] = {
    0: (class_proportion[1] -0.1),
    1: 1 - (class_proportion[1] -0.1),
}
space['batch_size'] = 200
space['epochs'] = 10
# Get data generators
data_stream = DataStream(batch_size=space['batch_size'], train_proportion=0.7,
                         class_proportion=class_proportion)
train_index_generator, test_index_generator = data_stream.split_by_patch_id(features[['image']], features[['destroyed']])
train_generator = data_stream.get_train_data_generator_from_index([features['image'], features['destroyed']],
                                                            train_index_generator)
    
test_indices = list(test_index_generator)
test_generator = data_stream.get_test_data_generator_from_index(features['image'], test_indices)

num_batches = ceil(len(features) / space['batch_size'])
# Fit model and predict
train_dataset = Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int32))
model = Model(**space)
model.fit_generator(train_dataset,
                    steps_per_epoch=num_batches,
                    verbose=1,
                    **space)

test_dataset = Dataset.from_generator(lambda: test_generator, tf.float32)
predictions = model.predict_generator(test_dataset)

test_indices_flattened = test_indices[0]
for index in test_indices[1:]:
    test_indices_flattened = test_indices_flattened.append(index)

predictions = pd.DataFrame({
    'prediction': predictions.reshape(-1),
}, index=test_indices_flattened)
file_name = '{}/prediction_{}.p'.format(RESULTS_PATH, round(time()))
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
