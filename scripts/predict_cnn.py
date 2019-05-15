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
RESULTS_PATH = 'logs'

city = 'Aleppo'
annotation_files = [
    #'4_Damage_Sites_{}_CDA.shp'.format(city),
    '6_Damage_Sites_{}_SDA.shp'.format(city),
]
annotation_data = read_annotations(file_names=['{}/{}'.format(ANNOTATIONS_PATH, f) for f in annotation_files])

raster_files = [
    #'daraa_2011_10_17_zoom_19.tif',
    #'daraa_2017_02_07_zoom_19.tif',
    #'homs_2011_05_21_zoom_19.tif',
    #'homs_2016_05_30_zoom_19.tif',
    #'raqqa_2013_01_17_zoom_19.tif',
    #'raqqa_2016_07_01_zoom_19.tif',
    'aleppo_2011_06_26_zoom_19.tif',
    'aleppo_2016_09_18_zoom_19.tif',
    'aleppo_2016_10_19_zoom_19.tif',
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
patch_size = 64*5
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
import ipdb; ipdb.set_trace()
experiment_files = os.listdir('{}/experiments'.format(RESULTS_PATH))
experiment_results = []
for file_name in experiment_files:
    with open('{}/experiments/{}'.format(RESULTS_PATH, file_name), 'r') as f:
        try:
            result = eval(json.load(f))
            result['id'] = int(file_name.split('_')[1].split('.')[0])
            experiment_results.append(result)
        except:
            continue

import ipdb; ipdb.set_trace()
experiment_results = pd.DataFrame(experiment_results)
space = experiment_results.loc[experiment_results['id'].idxmax(), 'space']
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
Model = CNN
data_stream = DataStream(batch_size=space['batch_size'], test_proportion=0.8)
num_batches = ceil(len(features) / space['batch_size'])
last_date = features.reset_index()['date'].max()
features_train = features.loc[features.index.get_level_values('date') < last_date]
features_predict = features.xs(last_date, level='date')
train_generator = data_stream.get_single_generator(np.stack(features_train['image'].values),
                                                   features_train['destroyed'].values)
space['class_weight'] = {
    0: features['destroyed'].mean(),
    1: 1 - features['destroyed'].mean(),
}
cnn = Model(**space)
losses = cnn.fit_generator(train_generator,
                           steps_per_epoch=num_batches,
                           validation_steps=1,
                           **space)

predictions = cnn.predict(features_predict['image'])
predictions = pd.DataFrame({
    'prediction': predictions,
}, features_predict.index)
predictions.to_pickle('{}/predictions/prediction_{}.p'.format(RESULTS_PATH, time()))
