import argparse
import pickle
import sys
sys.path.insert(1, './')
from time import time
from datetime import timedelta

from damage.data import load_data_multiple_cities, save_as_pickle_to_path
from damage import features


parser = argparse.ArgumentParser()
parser.add_argument('--filename')
args = vars(parser.parse_args())
file_name = args.get('filename', None) or '{}.p'.format(str(round(time())))

# Constants
STORING_PATH = 'logs/features'
CITIES = ['aleppo']
PATCH_SIZE = 64
TIME_TO_ANNOTATION_THRESHOLD = timedelta(weeks=1)
STRIDE = PATCH_SIZE  # dont change

# Reading
data = load_data_multiple_cities(CITIES)

# Processing
pipeline = features.Pipeline(
    preprocessors=[
        ('AnnotationPreprocessor', features.AnnotationPreprocessor()),
    ],
    features=[
        ('RasterSplitter', features.RasterSplitter(patch_size=PATCH_SIZE, stride=STRIDE)),
        ('AnnotationMaker', features.AnnotationMaker(PATCH_SIZE, TIME_TO_ANNOTATION_THRESHOLD)),
        ('RasterPairMaker', features.RasterPairMaker()),
    ],
)
features = pipeline.transform(data)
save_as_pickle_to_path(features, '{}/{}'.format(STORING_PATH, file_name))
save_as_pickle_to_path(features[['destroyed', 'latitude', 'longitude']],
                       '{}/target_{}'.format(STORING_PATH, file_name))
print('Features stored in {}/{}'.format(STORING_PATH, file_name))
