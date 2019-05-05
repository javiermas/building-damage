import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd

from damage.data import DataStream, Raster
from damage.models import CNN, RandomSearch
from damage import features


def preprocess_damage_data(damage_data):
    damage_data['DmgCls'] = damage_data['DmgCls'].replace({None: 'No damage'})
    damage_data['geometry_tuple'] = damage_data['geometry'].apply(lambda point: (point.x, point.y))
    damage_data['latitude'] = damage_data['geometry'].apply(lambda point: point.y)
    damage_data['longitude'] = damage_data['geometry'].apply(lambda point: point.x)
    return damage_data

def get_image_index_from_annotation(annotation_data, city_data):
    return annotation_data.apply(lambda x:
        pd.Series({key: i for key, i in zip(['row', 'col'], city_data.index(x['longitude'], x['latitude']))}),
        axis=1)

def crop_annotation_to_image_dimensions(annotation_data, dimensions):
    mask = (annotation_data['row'] < dimensions['height'])\
        & (annotation_data['row'] >= 0)\
        & (annotation_data['col'] < dimensions['width'])\
        & (annotation_data['col'] >= 0)
    return annotation_data.loc[mask]

def raster_to_array(raster_data):
    raster_array = raster_data.read(indexes=[1,2,3])
    raster_array = np.swapaxes(np.swapaxes(raster_array, 1, 2), 0, 2)
    return raster_array

def create_feature_matrix(city_array, annotation_data, window=50):
    assert (window) % 2 == 0
    extra_pixels = int((window)/2)
    feature_data = []
    for ix, row in annotation_data.iterrows():
        pixel_row, pixel_col = row['row'], row['col']
        building = get_building_frame(city_array, pixel_row, pixel_col, extra_pixels)
        feature_data.append({'image': building, 'target': (row['DmgCls_2'] == 'Destroyed') * 1})
        
    return pd.DataFrame(feature_data)

def get_building_frame(city_array, row, col, extra_pixels):
    return city_array[(row-extra_pixels):(row+extra_pixels), (col-extra_pixels):(col+extra_pixels), :].astype(float)

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
    '6_Damage_Sites_Aleppo_SDA.shp',
])
raster_data = read_rasters(file_names=[
    'aleppo_2011_06_26_zoom_19.tif',
    'aleppo_2016_10_19_zoom_19.tif',
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
