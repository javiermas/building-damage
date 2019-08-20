from datetime import timedelta
import pandas as pd
import numpy as np

from damage.features.base import Feature


class AnnotationMaker(Feature):

    def __init__(self, patch_size, time_to_annotation_threshold):
        super().__init__()
        self.patch_size = patch_size
        self.time_to_annotation_threshold = time_to_annotation_threshold

    def transform(self, data):
        annotation_data = {key: value for key, value in data.items() if 'annotation' in key}
        annotation_data = self._combine_annotation_data(annotation_data)
        annotation_data = self._assign_patch_id_to_annotation(annotation_data, data)
        annotation_data = self._group_annotations_by_patch_id(annotation_data)
        annotation_data = self._combine_annotations_with_raster_dates(annotation_data, data['RasterSplitter'])
        annotation_data['destroyed'] = annotation_data['damage_num'].apply(self._damage_to_destruction)
        return annotation_data.set_index(['city', 'patch_id', 'date'])
    
    @staticmethod  
    def _damage_to_destruction(damage_num):
        if np.isnan(damage_num):
            return np.nan 
        elif damage_num == 3:
            return 1
        else:
            return 0

    @staticmethod
    def _combine_annotation_data(annotation_data):
        annotations = []
        for name, annotation in annotation_data.items():
            annotations.append(annotation)

        annotation_data = pd.concat(annotations).reset_index(drop=True)
        return annotation_data

    @staticmethod
    def _group_annotations_by_location_index(annotation_data):
        return annotation_data.groupby(['city', 'location_index', 'date'])['damage_num'].max()

    @staticmethod
    def _group_annotations_by_patch_id(annotation_data):
        return annotation_data.groupby(['city', 'patch_id', 'date'])['damage_num'].max()

    def _assign_patch_id_to_annotation(self, annotation_data, data):
        annotation_data_with_patch_id = []
        for city in annotation_data['city'].unique():
            raster_key_single_city = [key for key in data.keys() if 'raster' in key and city in key][0]
            raster_data_single_city = data[raster_key_single_city]
            annotation_data_single_city = annotation_data.loc[annotation_data['city'] == city]
            rows, cols = raster_data_single_city.index(
                annotation_data_single_city['longitude'],
                annotation_data_single_city['latitude']
            )
            rows_centroids = [int(self.patch_size * (r//self.patch_size) + (self.patch_size/2)) for r in rows]
            cols_centroids = [int(self.patch_size * (c//self.patch_size) + (self.patch_size/2)) for c in cols]
            patch_ids = ['{}-{}'.format(c, r) for r, c in zip(rows_centroids, cols_centroids)]
            annotation_data_single_city['patch_id'] = patch_ids
            annotation_data_with_patch_id.append(annotation_data_single_city)

        annotation_data_with_patch_id = pd.concat(annotation_data_with_patch_id)
        return annotation_data_with_patch_id

    def _combine_annotations_with_raster_dates(self, annotation_data, raster_data):
        annotation_dates = annotation_data.index.get_level_values('date').unique().tolist()
        cities = raster_data.index.get_level_values('city').unique()
        date_mappings = []
        for city in cities:
            raster_data_single_city = raster_data.xs(city, level='city')
            raster_dates = [d.date() for d in raster_data_single_city.index.get_level_values('date').unique()]
            for date in raster_dates:
                closest_previous_date = self._get_closest_previous_date(date, annotation_dates)
                date_mapping = {
                    'raster_date': date,
                    'annotation_date': closest_previous_date,
                    'city': city
                }
                date_mappings.append(date_mapping)

        raster_dates = pd.DataFrame(date_mappings)
        raster_dates_with_annotation_data = self._get_raster_dates_with_annotation_data(
            raster_dates,
            annotation_data.reset_index()
        )
        annotations_by_gap = self._get_long_and_short_gap_annotations(raster_dates_with_annotation_data)
        annotations_long_gap, annotations_short_gap = annotations_by_gap
        # Pandas seems to have a bug that changes the dtype of
        # a date column to datetime automatically when assigning to index
        raster_index = pd.DataFrame(raster_data.index.tolist(), columns=raster_data.index.names)
        raster_index['date'] = raster_index['date'].dt.date
        annotation_data = self._combine_long_and_short_gap_annotations_with_raster_data(
            annotations_long_gap,
            annotations_short_gap,
            raster_index,
        )
        return annotation_data

    def _get_raster_dates_with_annotation_data(self, raster_dates, annotation_data):
        raster_dates_with_annotation_data = pd.merge(
            raster_dates,
            annotation_data,
            left_on=['city', 'annotation_date'], right_on=['city', 'date']
        ).drop('date', axis=1).rename(columns={'raster_date': 'date'})
        return raster_dates_with_annotation_data

    def _get_long_and_short_gap_annotations(self, annotation_data):
        annotations_long_gap = annotation_data.loc[
            (annotation_data['date']\
             - annotation_data['annotation_date']) > self.time_to_annotation_threshold
        ]
        annotations_short_gap = annotation_data.loc[
            (annotation_data['date']\
             - annotation_data['annotation_date']) <= self.time_to_annotation_threshold
        ]
        return annotations_long_gap, annotations_short_gap

    def _get_closest_previous_date(self, date, pool_of_dates):
        previous_dates = [date_pool for date_pool in pool_of_dates
                          if self._is_date_previous_or_same_to_date(date_pool, date)]
        if not previous_dates:
            return None

        closest_previous_date = max(previous_dates)
        return closest_previous_date

    def _combine_long_and_short_gap_annotations_with_raster_data(
        self,
        annotations_long_gap,
        annotations_short_gap,
        raster_data
        ):
        raster_short_gap = raster_data.loc[raster_data['date'].isin(annotations_short_gap['date'].unique())]
        raster_long_gap = raster_data.loc[raster_data['date'].isin(annotations_long_gap['date'].unique())]
        # We take all raster data from the short gap ones, so we can fill it with 0 (no destruction).
        annotations_short_gap = pd.merge(raster_short_gap, annotations_short_gap,
                                         on=['city', 'patch_id', 'date'], how='left')
        annotations_short_gap['damage_num'] = annotations_short_gap['damage_num'].fillna(0)
        # We take only raster data from the long gap that matches with the annotations,
        # so we only keep the destroyed ones, and we backfill with 0 (if not destroyed in future, not destroyed 
        # in present, aka assume no reconstruction).
        annotations_long_gap = pd.merge(raster_long_gap, annotations_long_gap,
                                         on=['city', 'patch_id', 'date'], how='left')
        annotation_data = pd.concat([annotations_short_gap, annotations_long_gap], sort=False)\
            .sort_values(['city', 'patch_id', 'date'], ascending=True)\
            .reset_index(drop=True)
        annotation_data['damage_num_filled'] = annotation_data\
            .groupby(['city', 'patch_id'])['damage_num']\
            .bfill()
        # If we couldnt match the annotations with the raster, then the annotation date
        # will be missing (non-destroyed ones)

        annotation_data.loc[
            (annotation_data['annotation_date'].isnull())
             & (annotation_data['damage_num_filled'] == 3),
            'damage_num_filled'
        ] = np.nan
        annotation_data = annotation_data\
            .drop('damage_num', axis=1)\
            .rename(columns={'damage_num_filled': 'damage_num'})
        return annotation_data

    @staticmethod
    def _is_date_previous_or_same_to_date(date_0, date_1):
        return (date_0 - date_1) <= timedelta(0)
