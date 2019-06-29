import os
from math import ceil
from time import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from sklearn.model_selection import KFold

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
features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name))#.dropna(subset=['destroyed'])
#features_destroyed = features.loc[features['destroyed'] == 1].sample(500)
#features_non_destroyed = features.loc[features['destroyed'] == 0].sample(5000)
#features = pd.concat([features_destroyed, features_non_destroyed])

# Modelling
Model = CNN

# Choose space
experiment_results = load_experiment_results()
if not experiment_results.empty:
    experiment_results_single_model = experiment_results.loc[experiment_results['model'] == str(Model)]
    experiment_results['val_precision_positives_mean'] = experiment_results['val_precision_positives']\
        .apply(lambda x: np.nan if isinstance(x, float) else np.mean(x[-3:]))
    experiment_results['val_recall_positives_mean'] = experiment_results['val_recall_positives']\
        .apply(lambda x: np.nan if isinstance(x, float) else np.mean(x[-3:]))
    experiment_results_single_model = experiment_results.loc[
        experiment_results['val_recall_positives_mean'] > 0.4
    ]
    space = experiment_results_single_model.loc[
        experiment_results_single_model['val_precision_positives_mean'].idxmax(), 'space']
else:
    space = RandomSearch._sample_single_cnn_space()

space = {'class_weight': {0: 0.345, 1: 0.655}, 'batch_size': 30, 'layer_type': 'cnn', 'convolutional_layers': [{'dropout': 0.1, 'activation': 'relu', 'kernel_size': [5, 5], 'filters': 32, 'pool_size': [4, 4]}, {'dropout': 0.1, 'activation': 'relu', 'kernel_size': [5, 5], 'filters': 64, 'pool_size': [4, 4]}, {'dropout': 0.1, 'activation': 'relu', 'kernel_size': [5, 5], 'filters': 128, 'pool_size': [4, 4]}, {'dropout': 0.1, 'activation': 'relu', 'kernel_size': [5, 5], 'filters': 256, 'pool_size': [4, 4]}], 'dense_units': 128, 'epochs': 11}

space['epochs'] = min(space['epochs'], 5)
class_proportion = {
    1: 0.3,
}

# Get data generators
features_upsampled = DataStream._upsample_class_proportion(features.dropna(subset=['destroyed']), class_proportion).sample(frac=1)
train_generator = DataStream._get_index_generator(features_upsampled, space['batch_size'])
num_batches = len(train_generator)
train_generator = DataStream.get_train_data_generator_from_index(
    [features_upsampled['image'], features_upsampled['destroyed']],
    train_generator
)
train_dataset = Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int32))

test_generator = DataStream._get_index_generator(features, space['batch_size'], KFold)
num_batches_test = len(test_generator)
test_generator = DataStream.get_test_data_generator_from_index(features['image'], test_generator)
test_dataset = Dataset.from_generator(lambda: test_generator, tf.float32)

# Fit model and predict
print('Training with space: \n')
print(space)
model = Model(**space)
model.fit_generator(train_dataset,
                    steps_per_epoch=num_batches,
                    verbose=1,
                    **space)

predictions = model.predict_generator(test_dataset, steps=num_batches_test)
predictions = pd.DataFrame({
    'prediction': predictions.reshape(-1),
}, index=features.index)
file_name = '{}/prediction_{}.p'.format(RESULTS_PATH, round(time()))
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
