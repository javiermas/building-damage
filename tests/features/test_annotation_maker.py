import unittest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from damage.features import AnnotationMaker
from tests.utils import assertFrameEqual


class TestAnnotationMaker(unittest.TestCase):

    def setUp(self):
        self.annotation_maker = AnnotationMaker()

    def test_combine_annotation_data(self):
        city_a_data = pd.DataFrame({
            'city': ['a'],
            'dmg': ['Destroyed'],
        })
        city_b_data = pd.DataFrame({
            'city': ['b'],
            'dmg': ['Severe Damage'],
        })
        annotation_data = {
            'annotation_city_a': city_a_data,
            'annotation_city_b': city_b_data,
        }
        expected_output = pd.DataFrame({
            'city': ['a', 'b'],
            'dmg': ['Destroyed', 'Severe Damage'],
        })
        output = self.annotation_maker._combine_annotation_data(annotation_data)
        assertFrameEqual(output, expected_output)

    def test_is_date_previous_or_same_to_date_previous_date(self):
        date_0, date_1 = date(2016, 1, 1), date(2016, 1, 2)
        output = self.annotation_maker._is_date_previous_or_same_to_date(date_0, date_1)
        self.assertTrue(output)

    def test_is_date_previous_or_same_to_date_posterior_date(self):
        date_0, date_1 = date(2016, 1, 3), date(2016, 1, 2)
        output = self.annotation_maker._is_date_previous_or_same_to_date(date_0, date_1)
        self.assertFalse(output)

    def test_get_closest_previous_data(self):
        pool_of_dates = [date(2016, 1, 1), date(2017, 1, 1), date(2018, 1, 1)]
        single_date = date(2016, 12, 31)
        output = self.annotation_maker._get_closest_previous_date(single_date, pool_of_dates)
        expected_output = date(2016, 1, 1)
        self.assertEqual(expected_output, output)

    def test_get_raster_dates_with_annotation_data(self):
        raster_dates = pd.DataFrame({
            'city': ['a'] * 3,
            'raster_date': [date(2015, 1, 1), date(2016, 2, 1), date(2017, 9, 1)],
            'annotation_date': [None, date(2016, 1, 1), date(2017, 1, 1)],
        })
        annotation_data = pd.DataFrame({
            'date': [date(2016, 1, 1), date(2017, 1, 1), date(2017, 10, 1), date(2018, 1, 1)],
            'city': ['a'] * 4,
            'damage': [1] * 4,
        })
        output = self.annotation_maker._get_raster_dates_with_annotation_data(raster_dates, annotation_data)
        expected_output = pd.DataFrame({
            'city': ['a'] * 2,
            'date': [date(2016, 2, 1), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), date(2017, 1, 1)],
            'damage': [1, 1],
        })
        assertFrameEqual(output, expected_output)
    
    def test_get_long_and_short_gap_annotations(self):
        annotation_data = pd.DataFrame({
            'city': ['a'] * 2,
            'date': [date(2016, 2, 1), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), date(2017, 1, 1)],
            'damage': [1, 1],
        })
        threshold = timedelta(days=30*6)
        output = self.annotation_maker._get_long_and_short_gap_annotations(annotation_data, threshold)
        expected_output_0 = pd.DataFrame({
            'city': ['a'],
            'date': [date(2017, 9, 1)],
            'annotation_date': [date(2017, 1, 1)],
            'damage': [1],
        })
        expected_output_1 = pd.DataFrame({
            'city': ['a'],
            'date': [date(2016, 2, 1)],
            'annotation_date': [date(2016, 1, 1)],
            'damage': [1],
        })
        assertFrameEqual(output[0], expected_output_0)
        assertFrameEqual(output[1], expected_output_1)
    
    def test_combine_long_and_short_gap_annotations_with_raster_data(self):
        short_gap = pd.DataFrame({
            'city': ['a']*2,
            'date': [date(2016, 2, 1), date(2016, 2, 1)],
            'annotation_date': [date(2016, 1, 1), date(2016, 1, 1)],
            'location_index': [1, 2],
            'damage_num': [1, 1],
        })
        long_gap = pd.DataFrame({
            'city': ['a']*2,
            'date': [date(2017, 9, 1), date(2017, 9, 1)],
            'annotation_date': [date(2017, 1, 1), date(2017, 1, 1)],
            'location_index': [0, 1],
            'damage_num': [1, 1],
        })
        raster_data = pd.DataFrame({
            'city': ['a']*4,
            'date': [date(2016, 2, 1), date(2016, 2, 1), date(2017, 9, 1), date(2017, 9, 1)],
            'location_index': [1, 3, 1, 2],
            'patch_id': ['a', 'b', 'a', 'b'],
        })
        expected_output = pd.DataFrame({
            'city': ['a']*3,
            'date': [date(2016, 2, 1), date(2016, 2, 1), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), np.nan, date(2017, 1, 1)],
            'location_index': [1, 3, 1],
            'damage_num': [1., 0., 1.],
            'patch_id': ['a', 'b', 'a'],
        })
        output = self.annotation_maker._combine_long_and_short_gap_annotations_with_raster_data(
            long_gap,
            short_gap,
            raster_data
        )
        assertFrameEqual(output, expected_output)


    """
    def test_make_single_raster_pair_expected_input_returns_expected_output(self):
        first_raster = pd.DataFrame({
            'location_index': [0],
            'SensDt': [date(2017, 1, 1)],
            'date': ['a'],
        })
        previous_raster = pd.DataFrame({
            'location_index': [0],
            'image': [np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])],
            'date': ['b'],
        })
        raster_pair = self.annotation_maker._make_single_raster_pair(first_raster, previous_raster)
        expected_output = pd.DataFrame({
            'location_index': [0],
            'image': [np.array([[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]])],
            'date': ['b'],
        })
        assertFrameEqual(raster_pair, expected_output)

    def test_make_raster_pairs_single_city_expected_input_returns_expected_output(self):
        raster_data = pd.DataFrame({
            'location_index': [0, 0],
            'image': [np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
                      np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])],
            'date': [date(2017, 1, 1), date(2018, 1, 1)],
        })
        output = self.annotation_maker._make_raster_pairs_single_city(raster_data)
        expected_output = pd.DataFrame({
            'location_index': [0],
            'image': [np.array([[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]])],
            'date': [date(2018, 1, 1)],
        })
        assertFrameEqual(output, expected_output) 

    def test_make_raster_pairs_multiple_cities_expected_input_returns_expected_output(self):
        raster_data = pd.DataFrame({
            'location_index': [0, 0],
            'image': [np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
                      np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])],
            'date': [date(2017, 1, 1), date(2018, 1, 1)],
            'city': ['a', 'a'],
        })
        output = self.annotation_maker._make_raster_pairs_all_cities(raster_data)
        expected_output = pd.DataFrame({
            'location_index': [0],
            'image': [np.array([[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]])],
            'date': [date(2018, 1, 1)],
            'city': ['a'],
        })
        assertFrameEqual(output, expected_output)
    """
