import abc
from math import sqrt
from typing import Optional

import numpy as np
from numpy.random import default_rng

from pydantic import NonNegativeFloat

from stochastic.processes.base import BaseProcess
from stochastic.processes.continuous.brownian_motion import BrownianMotion


class MidpriceModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_next_price(self, current_midprice: float):
        pass

    @property
    @abc.abstractmethod
    def max_value(self):
        pass


class FillProbabilityFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate_fill_probability(self, spread: NonNegativeFloat) -> NonNegativeFloat:
        pass


class ArrivalModel(metaclass=abc.ABCMeta):  # TODO: generalise
    @abc.abstractmethod
    def calculate_next_arrival_rate(self) -> float:
        pass

    @abc.abstractmethod
    def get_arrival(self):
        pass


class StochasticMidpriceModel(MidpriceModel):
    """A wrapper for the python package 'stochastic' to generate trajectories for a large class of processes"""

    def __init__(
        self, process: BaseProcess = None, initial_value: float = 1.0, is_multiplicative: bool = False, seed: Optional[int] = None
    ):
        self.process = process or BrownianMotion(rng=np.random.default_rng(seed), scale=2.0)  # As in Avellaneda-Stoikov
        self.initial_value = initial_value
        self.is_multiplicative = is_multiplicative

    def get_next_price(self, current_midprice: np.ndarray):
        if self.is_multiplicative:
            return self.process.sample_at(current_midprice, initial=self.initial_value)
        else:
            return self.process.sample_at(current_midprice) + self.initial_value


class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 1.0,
        initial_price: float = 100,
        dt: float = 0.1,
        terminal_time: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.initial_price = initial_price
        self.dt = dt
        self.terminal_time = terminal_time
        self.rng = default_rng(seed)

    def get_next_price(self, current_midprice: np.ndarray):
        return current_midprice + self.drift * self.dt + self.volatility * sqrt(self.dt) * self.rng.normal()

    @property
    def max_value(self):
        return self.initial_price + 4 * self.volatility * self.terminal_time


class ExponentialFillFunction(FillProbabilityFunction):
    def __init__(self, fill_exponent: float = 1.5, seed: Optional[int] = None):
        self.fill_exponent = fill_exponent
        self.rng = default_rng(seed)

    def calculate_fill_probability(self, half_spread: NonNegativeFloat) -> NonNegativeFloat:
        return np.exp(-self.fill_exponent * half_spread)

    def get_fill(self, half_spread: NonNegativeFloat) -> NonNegativeFloat:
        unif = self.rng.uniform()
        return unif < self.calculate_fill_probability(half_spread)


class PoissonArrivalModel(ArrivalModel):

    def __init__(self, intensity: float = 140, seed: Optional[int] = None):
        self.intensity = intensity
        self.rng = default_rng(seed)

    def calculate_next_arrival_rate(self) -> float:
        return self.intensity

    def get_arrival(self, interval_length:float):
        unif = self.rng.uniform()
        return unif < self.intensity * interval_length
