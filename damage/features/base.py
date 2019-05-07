from abc import ABC, abstractmethod


class Transformer(ABC):

    def __call__(self, *args, **kwargs):
        return self.transform(*args. **kwargs)

    @abstractmethod
    def transform(self, data):
        pass


class Feature(Transformer):
    pass


class Preprocessor(Transformer):
    pass
