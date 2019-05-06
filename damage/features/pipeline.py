class Pipeline:

    def __init__(self, features, preprocessors):
        self.features = features
        self.preprocessors = preprocessors

    def transform(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)

        for feature in self.features:
            data[feature.__class__.__name__] = feature.transform(data)

        return {feature.__class__.__name__: data[feature.__class__.__name__] for feature in self.features}
