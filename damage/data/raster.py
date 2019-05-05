import numpy as np
import rasterio


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
