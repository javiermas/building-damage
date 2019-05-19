import os
from math import ceil
from time import time
import numpy as np
import pandas as pd

from damage.models import CNN
from damage.data import DataStream, load_data_multiple_cities, load_experiment_results
from damage import features


os.environ['CUDA_VISIBLE_DEVICES'] = '5'
RESULTS_PATH = 'logs/predictions'

## Reading
cities = ['daraa']
data = load_data_multiple_cities(cities)

### Processing
grid_size = 0.035
patch_size = 64*10
stride = patch_size#16
pipeline = features.Pipeline(
    preprocessors=[
        ('AnnotationPreprocessor', features.AnnotationPreprocessor(grid_size=grid_size)),
    ],
    features=[
        ('RasterSplitter', features.RasterSplitter(patch_size=patch_size, stride=stride, grid_size=grid_size)),
        ('AnnotationMaker', features.AnnotationMaker()),
        ('RasterPairMaker', features.RasterPairMaker()),
    ],

)
features = pipeline.transform(data)

#### Modelling
Model = CNN
experiment_results = load_experiment_results()
experiment_results_single_model = experiment_results.loc[experiment_results['model'] == 'ABCMeta']
space = experiment_results_single_model.loc[experiment_results_single_model['id'].idxmax(), 'space']
space['epochs'] = 1
space['class_weight'] = {
    0: features['destroyed'].mean(),
    1: 1 - features['destroyed'].mean(),
}

data_stream = DataStream(batch_size=space['batch_size'], train_proportion=0.6)
train_generator, test_generator, test_patches = data_stream.split_by_patch_id(features['image'],
                                                                              features['destroyed'])

num_batches = ceil(len(features) / space['batch_size'])
model = Model(**space)
model.fit_generator(train_generator,
                    steps_per_epoch=num_batches,
                    validation_steps=1,
                    **space)

import ipdb; ipdb.set_trace()
features_test = np.concatenate(list(test_generator))
predictions = model.predict_generator(features_test, steps=ceil(len(test_patches)/batch_size))
predictions = pd.DataFrame({
    'prediction': predictions,
}, features_test.index)
file_name = '{}/prediction_{}.p'.format(RESULTS_PATH, time())
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
