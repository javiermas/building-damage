import os
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from math import ceil
from time import time

from damage.data import DataStream, Raster
from damage.models import CNN, RandomSearch
from damage import features


def read_annotations(file_names):
    data = {}
    for file_name in file_names:
        data['annotation_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data


def read_rasters(file_names):
    data = {}
    for file_name in file_names:
        data['raster_'+file_name.split('/')[-1]] = Raster(path=file_name)

    return data


def read_populated_areas(file_names):
    data = {}
    for file_name in file_names:
        populated_area = gpd.read_file(file_name)
        populated_area['NAME_EN'] = populated_area['NAME_EN'].str.lower()
        data['populated_areas_'+file_name.split('/')[-1]] = populated_area

    return data

def read_no_analysis_areas(file_names):
    data = {}
    for file_name in file_names:
        data['no_analysis_areas_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data


## Reading
RASTERS_PATH = 'data/city_rasters'
ANNOTATIONS_PATH = 'data/annotations'
POLYGONS_PATH = 'data/polygons'
RESULTS_PATH = 'logs/experiments'

city = 'Daraa'
annotation_files = [
    '4_Damage_Sites_{}_CDA.shp'.format(city),
    #'6_Damage_Sites_{}_SDA.shp'.format(city),
]
annotation_data = read_annotations(file_names=['{}/{}'.format(ANNOTATIONS_PATH, f) for f in annotation_files])

raster_files = [
    'daraa_2011_10_17_zoom_19.tif',
    'daraa_2017_02_07_zoom_19.tif',
    #'homs_2011_05_21_zoom_19.tif',
    #'homs_2016_05_30_zoom_19.tif',
    #'raqqa_2013_01_17_zoom_19.tif',
    #'raqqa_2016_07_01_zoom_19.tif',
    #'aleppo_2011_06_26_zoom_19.tif',
    #'aleppo_2016_10_19_zoom_19.tif',
]
raster_data = read_rasters(file_names=['{}/{}'.format(RASTERS_PATH, f) for f in raster_files])

populated_areas = read_populated_areas(file_names=[
    '{}/populated_areas.shp'.format(POLYGONS_PATH),
])
no_analysis_area = read_no_analysis_areas(file_names=[
    '{}/5_No_Analysis_Areas_{}.shp'.format(POLYGONS_PATH, city),
])

data = {**annotation_data, **raster_data, **populated_areas, **no_analysis_area}
### Processing
grid_size = 0.035
patch_size = 64
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
random_search = RandomSearch()
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
Model = CNN
spaces = random_search.sample_cnn(10)
for space in spaces:
    cnn = Model(**space)
    data_stream = DataStream(batch_size=space['batch_size'], test_proportion=0.8)
    num_batches = ceil(len(features) / space['batch_size'])
    train_generator, test_generator = data_stream.split(np.stack(features['image'].values),
                                                        features['destroyed'].values)
    #try:
    losses = cnn.validate_generator(train_generator, test_generator,
                                    steps_per_epoch=num_batches,
                                    validation_steps=1,
                                    **space)
    '''
    except ValueError:
        losses = {} # In case of erroneous architecture
    '''
    losses['model'] = Model.__class__.__name__
    losses['space'] = space
    losses['patch_size'] = patch_size
    with open('{}/experiment_{}.json'.format(RESULTS_PATH, round(time())), 'w') as f:
        json.dump(losses, f)
