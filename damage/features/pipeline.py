from functools import reduce
import pandas as pd

from damage.features.base import Transformer


class Pipeline(Transformer):

    def __init__(self, features, preprocessors):
        self.features = features
        self.feature_names = [feature_name for feature_name, _ in self.features]
        self.preprocessors = preprocessors

    def transform(self, data):
        for preprocessor_name, preprocessor in self.preprocessors:
            data = preprocessor(data)

        for feature_name, feature in self.features:
            data[feature_name] = feature(data)

        feature_data = [data[name] for name in self.feature_names if name in data.keys()]
        feature_data = self._merge_feature_data(feature_data)
        return feature_data

    def _merge_feature_data(self, feature_data):
        return reduce(lambda l, r: pd.merge(l, r, left_index=True, right_index=True, how='outer'), feature_data)
