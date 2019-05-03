import numpy as np
from matplotlib import pyplot as plt
import rasterio

from damage.utils import update_progress


class Raster:
    """ This class is a wrapper over rasterio's Raster class
    to add extra functionalities and standardize the
    class methods to other data sources."""

    def __init__(self, path=None, raster=None):
        if path is not None and raster is not None:
            raise ValueError('Please provide either a path or a raster, not both')

        if path is not None:
            self.raster = self._read_raster(path)

        if raster is not None:
            self.raster = self._read_raster(raster)

        self.width = self.raster.width
        self.height = self.raster.height

    def _read_raster(self, path):
        return rasterio.open(path)

    @classmethod
    def from_raster(cls, raster):
        raster_self = cls(raster=raster)
        raster_self.raster = raster
        return raster_self

    def to_array(self):
        raster_array = self.raster.read(indexes=[1,2,3])
        raster_array = np.swapaxes(np.swapaxes(raster_array, 1, 2), 0, 2)
        return raster_array

    def plot(self):
        raster_array = self.to_array()
        _, ax = plt.subplots(figsize=(25, 25))
        ax.imshow(raster_array)

    def split(self, tile_size=64, stride=16): 
        array = self.to_array().astype(float)
        tiles = []
        number_of_iterations = len(range(tile_size//2, (self.width - tile_size//2), stride))
        for i, w in enumerate(range(tile_size//2, (self.width - tile_size//2), stride)):
            for h in range(tile_size//2, (self.height - tile_size//2), stride):
                longitude, latitude = self.raster.xy(h, w)
                left = (w - tile_size//2)
                right = (w + tile_size//2 + 1)
                top = (h - tile_size//2)
                bottom = (h + tile_size//2 + 1)
                tile = {
                    'image': array[top:bottom, left:right],
                    'longitude': longitude,
                    'latitude': latitude,
                }
                tiles.append(tile)

            update_progress(i / number_of_iterations)

        update_progress(1)
        return tiles
