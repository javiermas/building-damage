from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self):
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self):
        pass

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
