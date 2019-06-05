import argparse
from time import time

from damage.data import load_data_multiple_cities
from damage import features


parser = argparse.ArgumentParser()
parser.add_argument('--filename')
args = vars(parser.parse_args())

STORING_PATH = 'logs/features'
file_name = args.get('filename', None) or '{}.p'.format(str(round(time())))
## Reading
cities = ['aleppo']
data = load_data_multiple_cities(cities)

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
features.to_pickle('{}/{}'.format(STORING_PATH, file_name))
print('Features stored in {}/{}'.format(STORING_PATH, file_name))
