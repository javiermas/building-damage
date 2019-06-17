import argparse
import pickle
import sys
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
patch_size = 128
stride = patch_size
pipeline = features.Pipeline(
    preprocessors=[
        ('AnnotationPreprocessor', features.AnnotationPreprocessor()),
    ],
    features=[
        ('RasterSplitter', features.RasterSplitter(patch_size=patch_size, stride=stride)),
        ('AnnotationMaker', features.AnnotationMaker(patch_size=patch_size)),
        ('RasterPairMaker', features.RasterPairMaker()),
    ],

)
features = pipeline.transform(data)
def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

save_as_pickled_object(features, '{}/{}'.format(STORING_PATH, file_name))
save_as_pickled_object(features['destroyed'], '{}/target_{}'.format(STORING_PATH, file_name))
print('Features stored in {}/{}'.format(STORING_PATH, file_name))
