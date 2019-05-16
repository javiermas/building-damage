import geopandas as gpd
import rasterio


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
