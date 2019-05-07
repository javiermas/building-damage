from datetime import date
import pandas as pd
from tqdm import tqdm

from damage.utils import geo_location_index
from damage.features.base import Feature


class RasterSplitter(Feature):

    def __init__(self, tile_size, stride, grid_size=0.035):
        self.tile_size = tile_size
        self.stride = stride
        self.grid_size = grid_size

    def transform(self, data):
        rasters = [(key, value) for key, value in data.items() if 'raster' in key]
        tiles = []
        for name, raster in rasters:
            array = raster.to_array().astype(float)
            city, year, month, day = self.parse_filename(name)
            number_of_iterations = len(range(self.tile_size//2, (raster.width-self.tile_size//2), self.stride))
            for w in tqdm(range(self.tile_size//2, (raster.width - self.tile_size//2), self.stride)):
                for h in range(self.tile_size//2, (raster.height - self.tile_size//2), self.stride):
                    longitude, latitude = raster.raster.xy(h, w)
                    left = (w - self.tile_size//2)
                    right = (w + self.tile_size//2 + 1)
                    top = (h - self.tile_size//2)
                    bottom = (h + self.tile_size//2 + 1)
                    tile = {
                        'image': array[top:bottom, left:right],
                        'longitude': longitude,
                        'latitude': latitude,
                        'city': city,
                        'date': date(year=year, month=month, day=day),
                    }
                    tiles.append(tile)
    
            data.pop(name) # Hack to avoid memory problems

        tiles = pd.DataFrame(tiles)
        tiles['location_index'] = geo_location_index(tiles['longitude'], tiles['latitude'],
                                                     grid_size=self.grid_size) 
        return tiles

    def parse_filename(self, filename):
        """This method assumes the following format:
        'raster_city_year_month_day...'
        """
        filename_split = filename.split('_')
        city = filename_split[1]
        year = int(filename_split[2])
        month = int(filename_split[3])
        day = int(filename_split[4])
        return city, year, month, day
