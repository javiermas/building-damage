import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from damage.data import DataStream
from damage.models import CNN


def preprocess_damage_data(damage_data):
    damage_data['DmgCls'] = damage_data['DmgCls'].replace({None: 'No damage'})
    damage_data['geometry_tuple'] = damage_data['geometry'].apply(lambda point: (point.x, point.y))
    damage_data['latitude'] = damage_data['geometry'].apply(lambda point: point.y)
    damage_data['longitude'] = damage_data['geometry'].apply(lambda point: point.x)
    return damage_data

def get_image_index_from_annotation(annotation_data, city_data):
    return annotation_data.apply(lambda x: pd.Series({key: i for key, i in zip(['row', 'col'], city_data.index(x['longitude'], x['latitude']))}), axis=1)
    
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

file_name_annotation = 'data/annotations/Damage_Sites_Damascus_2017_Ex_Update.shp'
file_name_city = 'data/city_rasters/damascus_2017_01_22_zoom_19.tif'
annotation_data = gpd.read_file(file_name_annotation)
annotation_data = preprocess_damage_data(annotation_data)
damascus_raster = rasterio.open(file_name_city)
damascus_array = raster_to_array(damascus_raster)
image_index = get_image_index_from_annotation(annotation_data, damascus_raster)
annotation_data = pd.merge(annotation_data, image_index, left_index=True, right_index=True)
annotation_data = crop_annotation_to_image_dimensions(annotation_data,
        {'height': damascus_raster.height, 'width': damascus_raster.width})

window = 50
feature_matrix = create_feature_matrix(damascus_array, annotation_data, window)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
graph_config = {
    'num_classes': 2,
    'convolutional_layers': [{'filters': 64, 'kernel_size': [5, 5], 'pool_size': [2, 2]} for _ in range(2)],
    'weight_positives': 1 - feature_matrix['target'].mean(),
    'learning_rate': 0.1,
}
cnn = CNN(**graph_config)
batch_size = 100
data_stream = DataStream(np.stack(feature_matrix['image'].values), feature_matrix['target'].values, batch_size)
data_generator = data_stream.get_generator()
train_config = {
    'epochs': 3,
    'steps_per_epoch': data_stream.num_batches,
}
cnn.fit_generator(data_generator, **train_config)
