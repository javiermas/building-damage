import argparse
import os
from time import time

from damage.data import load_data_multiple_cities, DataStream
from damage import features


STORING_PATH = 'logs/features'
## Reading
cities = ['daraa']
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
import ipdb; ipdb.set_trace()
# Store batches
BATCH_SIZE = 50
TRAIN_PROPORTION = 0.7
CLASS_PROPORTION = {
    1: 0.4,
}
FOLDER_NAME = '_'.join(c for c in cities)

data_stream = DataStream(BATCH_SIZE, TRAIN_PROPORTION, CLASS_PROPORTION)
train_index_generator, test_index_generator = data_stream.split_by_patch_id(features['image'], features['destroyed'])


if not os.path.exists('{}/{}'.format(STORING_PATH, FOLDER_NAME)):
    os.mkdir('{}/{}'.format(STORING_PATH, FOLDER_NAME))
    os.mkdir('{}/{}/train'.format(STORING_PATH, FOLDER_NAME))
    os.mkdir('{}/{}/test'.format(STORING_PATH, FOLDER_NAME))

data_stream.store_data_batches_from_index(
    path='{}/{}/train'.format(STORING_PATH, FOLDER_NAME),
    data=[features['image'], features['destroyed']],
    index=train_index_generator
)

data_stream.store_data_batches_from_index(
    path='{}/{}/test'.format(STORING_PATH, FOLDER_NAME),
    data=[features['image'], features['destroyed']],
    index=test_index_generator
)
