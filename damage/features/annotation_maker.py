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
        annotation_data = annotation_data.dropna(subset=['date']).drop('location_index', axis=1)
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

    @staticmethod
    def _assign_patch_id_to_annotation(raster_data, annotation_data):
        # Pandas seems to have a bug that changes the dtype of
        # a date column to datetime automatically when assigning to index
        raster_locations_no_index = raster_data['location_index'].reset_index()
        raster_locations_no_index['date'] = raster_locations_no_index['date'].dt.date
        # Left join on raster data because we are not interested
        # on annotations that do not match with any raster patch
        annotation_data = pd.merge(raster_locations_no_index, annotation_data.reset_index(),
                                   on=['city', 'location_index', 'date'], how='left')
        # If there's no annotation, we assume it is not destroyed
        annotation_data['damage_num'] = annotation_data['damage_num'].fillna(0)
        return annotation_data
