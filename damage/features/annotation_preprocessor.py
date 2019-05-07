from damage.utils import geo_location_index
from damage.features.base import Preprocessor


class AnnotationPreprocessor(Preprocessor):

    def __init__(self, grid_size=0.035):
        self.grid_size = grid_size

    def transform(self, data):
        for key, value in data.items():
            if 'annotation' not in key:
                continue

            value['latitude'] = value['geometry'].apply(lambda point: point.y)
            value['longitude'] = value['geometry'].apply(lambda point: point.x)
            value['location_index'] = geo_location_index(value['latitude'], value['longitude'], self.grid_size)
            data[key] = value

        return data
