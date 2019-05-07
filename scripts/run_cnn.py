import os
import geopandas as gpd
import numpy as np
import pandas as pd

from damage.data import DataStream, Raster
from damage.models import CNN, RandomSearch
from damage import features


def crop_annotation_to_image_dimensions(annotation_data, dimensions):
    mask = (annotation_data['row'] < dimensions['height'])\
        & (annotation_data['row'] >= 0)\
        & (annotation_data['col'] < dimensions['width'])\
        & (annotation_data['col'] >= 0)
    return annotation_data.loc[mask]

def read_annotations(file_names):
    path = 'data/annotations/'
    data = {}
    for file_name in file_names:
        data['annotation_'+file_name] = gpd.read_file(path+file_name)

    return data

def read_rasters(file_names):
    path = 'data/city_rasters/'
    data = {}
    for file_name in file_names:
        data['raster_'+file_name] = Raster(path=path+file_name)

    return data
## Reading
annotation_data = read_annotations(file_names=[
    '4_Damage_Sites_Daraa_CDA.shp',
])
raster_data = read_rasters(file_names=[
    'daraa_2011_10_17_zoom_19.tif',
    'daraa_2016_04_19_zoom_19.tif',
    #'homs_2011_05_21_zoom_19.tif',
    #'homs_2016_05_30_zoom_19.tif',
    #'raqqa_2013_01_17_zoom_19.tif',
    #'raqqa_2016_07_01_zoom_19.tif',
    #'aleppo_2011_06_26_zoom_19.tif',
    #'aleppo_2016_10_19_zoom_19.tif',
])
data = {**annotation_data, **raster_data}
### Processing
grid_size = 0.035
tile_size = 64
stride = 16
pipeline = features.Pipeline(
    preprocessors=[
        features.AnnotationPreprocessor(grid_size=grid_size),
    ],
    features=[
        features.RasterSplitter(tile_size=tile_size, stride=stride),
        features.RasterPairMaker(),
    ],

)
features = pipeline.transform(data)

import ipdb; ipdb.set_trace()
#### Modelling
random_search = RandomSearch()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
Model = CNN
spaces = random_search.sample_cnn(1)
for space in spaces:
    cnn = Model(**space)
    data_stream = DataStream(np.stack(feature_matrix['image'].values),
                             feature_matrix['target'].values,
                             space['batch_size'])
    train_generator, test_generator = data_stream.get_generators()
    losses = cnn.validate_generator(train_generator, test_generator,
                                    steps_per_epoch=data_stream.num_batches,
                                    validation_steps=1,
                                    **space)
    losses['model'] = Model.__class__.__name__
    losses['space'] = space
    losses['window'] = window
