import abc

import numpy as np


class Featuriser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def featurise(self, state: np.ndarray):
        pass


class TileCoder(Featuriser):
    def featurise(self, state: np.ndarray):
        pass
