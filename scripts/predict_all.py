import os
from math import ceil
from time import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold

from damage.models import CNN, RandomSearch
from damage.data import DataStream, load_experiment_results
from damage import features


parser = argparse.ArgumentParser()
parser.add_argument('--features', required=True)
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

# Choose space and model
experiment_results = load_experiment_results()
if not experiment_results.empty:
    available_models = [m.split('_')[1].split('.')[0] for m in os.listdir('logs/models')]
    experiment_results = experiment_results.loc[
        (experiment_results['model'] == str(Model))
        & (experiment_results['features'] == features_file_name)
        & (experiment_results['id'].isin(available_models))
    ]
    experiment_results['val_precision_positives_last_epoch'] = experiment_results['val_precision_positives']\
        .apply(lambda x: np.nan if isinstance(x, float) else x[-1])
    experiment_results['val_recall_positives_last_epoch'] = experiment_results['val_recall_positives']\
        .apply(lambda x: np.nan if isinstance(x, float) else x[-1])
    experiment_results = experiment_results.loc[]
    identifier = experiment_results.loc[
        (experiment_results['val_precision_positives_last_epoch'].idxmax())
        & (experiment_results['val_recall_positives_last_epoch'] > 0.4),
        'id'
    ]
    space = experiment_results.loc[experiment_results['val_precision_positives_last_epoch'].idxmax(), 'space']
    try:
        print('Loading model {}'.format(identifier))
        print('With space {}'.format(space))
        model = load_model('logs/models/model_{}.h5'.format(identifier))
        print('Model loaded')
    except Exception as e:
        raise e('Error loading model')
else:
    print('Predicting with randomly sampled space. It is recommended to run some experiments first')
    space = RandomSearch._sample_single_cnn_space()

test_generator = DataStream._get_index_generator(features, space['batch_size'], KFold)
num_batches_test = len(test_generator)
test_generator = DataStream.get_test_data_generator_from_index(features['image'], test_generator)
test_dataset = Dataset.from_generator(lambda: test_generator, tf.float32)

# Predict
print('Generating predictions')
predictions = model.predict_generator(test_dataset, steps=num_batches_test)
predictions = pd.DataFrame({
    'prediction': predictions.reshape(-1),
}, index=features.index)
file_name = '{}/prediction_{}.p'.format(RESULTS_PATH, round(time()))
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
