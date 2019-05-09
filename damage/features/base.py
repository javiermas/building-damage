from abc import ABC, abstractmethod


class Transformer(ABC):
    """ A Transformer class is a class with a public method transform
    that takes a dictionary of data and returns some transformation
    of that data.
    """
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, data):
        pass


class Preprocessor(Transformer):
    """ A Preprocessor class is a Transformer class whose transform
    method takes a dictionary of data and returns that same
    dictionary after some modification. The Preprocessor overwrites.
    """
    pass


class Feature(Transformer):
    """ A Feature class is a Transformer class whose transform
    method takes a dictionary of data and returns a new
    data object. The Feature creates something new respecting
    the existing data.
    """
    pass
