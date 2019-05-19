import os
import argparse
import json
from math import ceil
from time import time
import numpy as np

from damage.data import read_annotations, read_populated_areas, read_rasters, read_no_analysis_areas, DataStream
from damage.data.data_sources import DATA_SOURCES
from damage.models import CNN, RandomSearch
from damage import features

parser = argparse.ArgumentParser()
parser.add_argument('--gpu')
args = vars(parser.parse_args())

## Reading
RASTERS_PATH = 'data/city_rasters'
ANNOTATIONS_PATH = 'data/annotations'
POLYGONS_PATH = 'data/polygons'
RESULTS_PATH = 'logs'

cities = ['aleppo']
data = {}
for city in cities:
    annotation_files = ['{}/{}'.format(ANNOTATIONS_PATH, f) for f in DATA_SOURCES['aleppo']['annotations']]
    annotation_data = read_annotations(file_names=annotation_files)

    raster_files = ['{}/{}'.format(RASTERS_PATH, f) for f in DATA_SOURCES['aleppo']['rasters']]
    raster_data = read_rasters(file_names=raster_files)

    no_analysis_files = ['{}/{}'.format(POLYGONS_PATH, f) for f in DATA_SOURCES['aleppo']['no_analysis']]
    no_analysis_area = read_no_analysis_areas(file_names=no_analysis_files)

    data = {**data, **annotation_data, **raster_data, **no_analysis_area}

populated_areas = read_populated_areas(file_names=[
    '{}/populated_areas.shp'.format(POLYGONS_PATH),
])

data = {**populated_areas, **data}

### Processing
grid_size = 0.035
patch_size = 64
stride = patch_size
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
random_search = RandomSearch()
os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
Model = CNN
spaces = random_search.sample_cnn(10)
for space in spaces:
    data_stream = DataStream(batch_size=space['batch_size'], test_proportion=0.8)
    num_batches = ceil(len(features) / space['batch_size'])
    train_generator, test_generator = data_stream.split(np.stack(features['image'].values),
                                                        features['destroyed'].values)
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
    losses['patch_size'] = patch_size
    with open('{}/experiments/experiment_{}.json'.format(RESULTS_PATH, round(time())), 'w') as f:
        json.dump(str(losses), f)
