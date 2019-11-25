import os
import random
import argparse
import json
from time import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from damage.data import DataStream, load_experiment_results, load_features_multiple_cities
from damage.models import CNN, RandomSearch, CNNPreTrained


parser = argparse.ArgumentParser()
parser.add_argument('--features', required=True)
parser.add_argument('--gpu')
args = vars(parser.parse_args())


os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
RESULTS_PATH = 'logs/experiments'
FEATURES_PATH = 'logs/features'
PREDICTIONS_PATH = 'logs/predictions'
TEST_MODE = os.environ.get('SYRIA_TEST', False)
features_file_name = args.get('features')
cities = features_file_name.split('.p,')

# Loading features
features = load_features_multiple_cities(cities, TEST_MODE)

# Best parameters
Model = CNN
experiment_results = load_experiment_results()
experiment_results = experiment_results.loc[
    (experiment_results['model'] == str(Model))
    & (experiment_results['features'] == 'aleppo.p')
]
experiment_results = experiment_results.loc[
    experiment_results['val_recall_positives_last_epoch'] > 0.5
]
if experiment_results.empty() or True:
    space = {
        'dense_units': 256,
        'prop_1_to_0': 0.3,
        'class_weight': {0: 0.1, 1: 0.7738110204081633},
        'learning_rate': 0.868511373751352,
        'convolutional_layers': [
            {'dropout': 0.25555555555555554, 'pool_size': [8, 8], 'kernel_size': [9, 9],
             'filters': 128, 'activation': 'relu'},
            {'dropout': 0.25555555555555554, 'pool_size': [8, 8], 'kernel_size': [9, 9],
             'filters': 256, 'activation': 'relu'}
        ], 
        'layer_type': 'cnn',
        'epochs': 8, 
        'prop_1_train': 0.2307688218957216, 
        'batch_size': 27
    }
    print('Running with default space', space)
else:
    best_experiment = experiment_results.loc[experiment_results['val_precision_positives_last_epoch'].idxmax()]
    space = best_experiment['space']
    print('Running with space', space)
    print('From best experiment')
    print(best_experiment)

# Modelling
RUNS = 50
for run in range(RUNS):
    class_proportion = {
        1: 0.3
    }
    test_batch_size = 200
    train_proportion = 0.7
    data_stream = DataStream(
        batch_size=space['batch_size'],
        train_proportion=train_proportion,
        class_proportion=class_proportion,
        test_batch_size=test_batch_size
    )
    unique_patches = features.index.get_level_values('patch_id').unique().tolist()
    train_patches = random.sample(unique_patches, round(len(unique_patches)*train_proportion))
    train_data = features.loc[features.index.get_level_values('patch_id').isin(train_patches)]
    train_data_upsampled = data_stream._upsample_class_proportion(train_data, class_proportion).sample(frac=1)
    test_patches = list(set(unique_patches) - set(train_patches))
    test_data = features.loc[features.index.get_level_values('patch_id').isin(test_patches)]

    train_indices = data_stream._get_index_generator(train_data_upsampled, space['batch_size'])
    test_indices = data_stream._get_index_generator(test_data, test_batch_size)
    train_generator = data_stream.get_train_data_generator_from_index(
        data=[train_data_upsampled['image'], train_data_upsampled['destroyed']],
        index=train_indices,
        augment_flip=best_experiment['augment_flip'],
        augment_brightness=best_experiment['augment_brightness'],
    )
    test_generator = data_stream.get_train_data_generator_from_index(
        data=[test_data['image'], test_data['destroyed']],
        index=test_indices,
        augment_flip=False,
        augment_brightness=False,
    )
    train_dataset = tf.data.Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int32))
    test_dataset = tf.data.Dataset.from_generator(lambda: test_generator, (tf.float32, tf.int32))

    num_batches = len(train_indices)
    num_batches_test = len(test_indices)
    if TEST_MODE:
        space['epochs'] = 1
        space['convolutional_layers'] = space['convolutional_layers'][:1]
        space['dense_units'] = 16
    
    space['prop_1_train'] = train_data_upsampled['destroyed'].mean()
    class_weight_0 = max(class_proportion[1] * space['class_weight'][0], 0.1)
    class_weight_1 = min((1 - (class_proportion[1] * space['class_weight'][1])), 0.9)
    space['class_weight'] = {
        0: class_weight_0,
        1: class_weight_1
    }
    print('Now validating:\n')
    print(space)
    try:
        model = Model(**space)
        losses = model.validate_generator(train_dataset, test_dataset,
                                          steps_per_epoch=num_batches,
                                          validation_steps=num_batches_test,
                                          **space)
    except Exception as e:
        losses = {'log': str(e)}

    losses['model'] = str(Model)
    losses['space'] = space
    losses['features'] = features_file_name
    losses['num_batches_train'] = num_batches
    losses['num_batches_test'] = num_batches_test
    losses['batch_size_test'] = test_batch_size
    identifier = round(time())
    test_indices = [i for batch in test_indices for i in batch]
    predictions_to_store = test_data.iloc[test_indices].drop('image', axis=1).copy()
    predictions_to_store['prediction'] = model.predict_generator(
        test_generator,
        steps=num_batches_test
    )
    losses['recall_val'] = (predictions_to_store.loc[
        predictions_to_store['destroyed'] == 1,
        'prediction'
    ] > 0.5).mean()
    losses['precision_val'] = predictions_to_store.loc[
        predictions_to_store['prediction'] >= 0.5,
        'destroyed'
    ].mean()
    losses['roc_auc_val'] = roc_auc_score(
        predictions_to_store['destroyed'], 
        predictions_to_store['prediction'],
    )
    with open('{}/experiment_{}.json'.format(RESULTS_PATH, identifier), 'w') as f:
        json.dump(str(losses), f)
        print('Experiment saved in experiment_{}.json'.format(identifier))
        
    if losses['roc_auc_val'] > 0.8:
        predictions_to_store.to_pickle(
            '{}/test_set_{}.p'.format(PREDICTIONS_PATH, identifier)
        )
        model.save('logs/models/model_{}.h5'.format(identifier))
