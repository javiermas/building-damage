import pandas as pd
from numpy import concatenate


class RasterPairMaker:

    """ This class makes pairs of rasters in different points in time
    by concatenating the first raster to every posterior one in each city
    (going from 3 to 6 channels).
    """

    def transform(self, data):
        raster_pairs = self._make_raster_pairs_all_cities(data['RasterSplitter'])
        return raster_pairs

    def _make_raster_pairs_all_cities(self, raster_data):
        raster_pairs_all_cities = []
        for city in raster_data['city'].unique():
            raster_data_single_city = raster_data.loc[raster_data['city'] == city]
            raster_pairs_single_city = self._make_raster_pairs_single_city(raster_data_single_city)
            raster_pairs_all_cities.append(raster_pairs_single_city)

        return pd.concat(raster_pairs_all_cities)

    def _make_raster_pairs_single_city(self, raster_data):
        """ This method creates pairs of images
        (image_0, image_1), (image_0, image_2) by concatenating
        on the third axis (3 to 6 channel images).
        """
        dates = raster_data['date'].unique()
        first_date = raster_data['date'].min()
        first_raster = raster_data.loc[raster_data['date'] == first_date]
        combined_rasters = []
        for date in dates:
            if date == first_date:
                continue

            single_raster = raster_data.loc[raster_data['date'] == date]
            combined_raster = self._make_single_raster_pair(first_raster, single_raster)
            combined_rasters.append(combined_raster)

        combined_rasters = pd.concat(combined_rasters)
        return combined_rasters

    def _make_single_raster_pair(self, common_raster, variable_raster):
        """ This method takes two rasters and combined them
        by concatenating the images in them on the second axis.
        It keeps the variable_raster as the data structure and
        concatenates the images of the common_raster to the ones
        in the variable_raster.
        """
        combined_raster = pd.merge(variable_raster,
                                   common_raster[['location_index', 'image']],
                                   on='location_index')
        combined_raster['image'] = combined_raster.apply(
            lambda x: concatenate([x['image_x'], x['image_y']], axis=2), axis=1)
        combined_raster = combined_raster.drop(['image_x', 'image_y'], axis=1)
        return combined_raster




