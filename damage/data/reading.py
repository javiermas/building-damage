import os
import json
import geopandas as gpd
import pandas as pd
import rasterio

from damage.data.data_sources import DATA_SOURCES


RASTERS_PATH = 'data/city_rasters'
ANNOTATIONS_PATH = 'data/annotations'
POLYGONS_PATH = 'data/polygons'
EXPERIMENTS_PATH = 'logs/experiments'


def load_data_multiple_cities(cities, rasters_path=RASTERS_PATH, annotations_path=ANNOTATIONS_PATH,
                              polygons_path=POLYGONS_PATH):
    data = {}
    for city in cities:
        data = {**data, **load_data_single_city(city, rasters_path, annotations_path, polygons_path)}

    populated_areas = read_populated_areas(file_names=[
        '{}/populated_areas.shp'.format(polygons_path),
    ])
    data = {**populated_areas, **data}
    return data


def load_data_single_city(city, rasters_path, annotations_path, polygons_path):
    annotation_files = ['{}/{}'.format(annotations_path, f) for f in DATA_SOURCES[city]['annotations']]
    annotation_data = read_annotations(file_names=annotation_files)

    raster_files = ['{}/{}'.format(rasters_path, f) for f in DATA_SOURCES[city]['rasters']]
    raster_data = read_rasters(file_names=raster_files)

    no_analysis_files = ['{}/{}'.format(polygons_path, f) for f in DATA_SOURCES[city]['no_analysis']]
    no_analysis_area = read_no_analysis_areas(file_names=no_analysis_files)

    data = {**annotation_data, **raster_data, **no_analysis_area}
    return data


def read_annotations(file_names):
    data = {}
    for file_name in file_names:
        data['annotation_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data


def read_rasters(file_names):
    data = {}
    for file_name in file_names:
        data['raster_'+file_name.split('/')[-1]] = rasterio.open(file_name)

    return data


def read_populated_areas(file_names):
    data = {}
    for file_name in file_names:
        data['populated_areas_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data

def read_no_analysis_areas(file_names):
    data = {}
    for file_name in file_names:
        data['no_analysis_areas_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data

def load_experiment_results(path=EXPERIMENTS_PATH):
    experiment_files = os.listdir(path)
    experiment_results = []
    for file_name in experiment_files:
        with open('{}/{}'.format(path, file_name), 'r') as f:
            try:
                result = eval(json.load(f))
                result['id'] = int(file_name.split('_')[1].split('.')[0])
                experiment_results.append(result)
            except:
                continue

    return pd.DataFrame(experiment_results)
