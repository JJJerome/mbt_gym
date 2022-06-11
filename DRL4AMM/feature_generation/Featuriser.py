import abc
from typing import List

import numpy as np

from DRL4AMM.feature_generation.SuttonTileCoder import IHT, tiles


class Featuriser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def featurise(self, state: np.ndarray, action: np.ndarray):
        pass

    @property
    @abc.abstractmethod
    def n_features(self):
        pass


class TileCoder(Featuriser):
    def __int__(self, n_tilings: int = 1, max_size: int = 2048, num_actions: int = 20, tile_size: float = 0.1):
        self.n_tilings = n_tilings
        self.max_size = max_size
        self.num_actions = num_actions
        self.iht = IHT(max_size)

    def featurise(self, state: np.ndarray, action: np.ndarray) -> List[int]:
        assert len(action) == 1, "Tile coding requires discrete 1 dimensional actions."
        return tiles(self.iht, self.n_tilings, action[0], state)

    @property
    def n_features(self):
        return self.n_tilings
