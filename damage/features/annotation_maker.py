from datetime import timedelta
import pandas as pd

from damage.features.base import Feature


class AnnotationMaker(Feature):

    def transform(self, data):
        annotation_data = {key: value for key, value in data.items() if 'annotation' in key}
        annotation_data = self._combine_annotation_data(annotation_data)
        annotation_data = self._group_annotations_by_location_index(annotation_data)
        annotation_data = self._assign_patch_id_to_annotation(data['RasterSplitter'], annotation_data)
        annotation_data['destroyed'] = (annotation_data['damage_num'] == 3) * 1
        # We drop nans on date because those are the images that come before
        # any annotation, and cannot be used for training
        #annotation_data = annotation_data.dropna(subset=['date']).drop('location_index', axis=1)
        return annotation_data.set_index(['city', 'patch_id', 'date'])

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

    def _assign_patch_id_to_annotation(self, raster_data, annotation_data):
        annotation_dates = annotation_data.index.get_level_values('date').unique().tolist()
        cities = raster_data.index.get_level_values('city').unique()
        date_mappings = []
        for city in cities:
            raster_data_single_city = raster_data.xs(city, level='city')
            raster_dates = [d.date() for d in raster_data_single_city.index.get_level_values('date').unique()]
            threshold = timedelta(days=30*6)
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
        threshold = timedelta(days=30*6)
        annotations_by_gap = self._get_long_and_short_gap_annotations(raster_dates_with_annotation_data,
                                                                      threshold)
        annotations_long_gap, annotations_short_gap = annotations_by_gap
        # Pandas seems to have a bug that changes the dtype of
        # a date column to datetime automatically when assigning to index
        raster_data_no_index = raster_data.reset_index()
        raster_locations_no_index = raster_data_no_index[['city', 'patch_id', 'location_index', 'date']]
        raster_locations_no_index['date'] = raster_locations_no_index['date'].dt.date
        annotation_data = self._combine_long_and_short_gap_annotations_with_raster_data(
            annotations_long_gap,
            annotations_short_gap,
            raster_locations_no_index
        )
        return annotation_data

    def _get_raster_dates_with_annotation_data(self, raster_dates, annotation_data):
        raster_dates_with_annotation_data = pd.merge(
            raster_dates,
            annotation_data,
            left_on=['city', 'annotation_date'], right_on=['city', 'date']
        ).drop('date', axis=1).rename(columns={'raster_date': 'date'})
        return raster_dates_with_annotation_data

    def _get_long_and_short_gap_annotations(self, annotation_data, threshold):
        annotations_long_gap = annotation_data.loc[
            (annotation_data['date']\
             - annotation_data['annotation_date']) > threshold
        ]
        annotations_short_gap = annotation_data.loc[
            (annotation_data['date']\
             - annotation_data['annotation_date']) <= threshold
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
                                         on=['city', 'location_index', 'date'], how='left')
        # We take only raster data from the long gap that matches with the annotations,
        # so we only keep the destroyed ones gap ones.
        annotations_long_gap = pd.merge(raster_long_gap, annotations_long_gap,
                                         on=['city', 'location_index', 'date'], how='inner')
        annotation_data = pd.concat([annotations_short_gap, annotations_long_gap], sort=False)
        # If there's no annotation, we assume it is not destroyed
        annotation_data['damage_num'] = annotation_data['damage_num'].fillna(0)
        return annotation_data

    @staticmethod
    def _is_date_previous_or_same_to_date(date_0, date_1):
        return (date_0 - date_1) <= timedelta(0)
