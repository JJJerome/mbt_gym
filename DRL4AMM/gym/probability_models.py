import abc
import numpy as np

from pydantic import NonNegativeFloat

from stochastic.processes.base import BaseProcess
from stochastic.processes.continuous.brownian_motion import BrownianMotion


class MidpriceModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_trajectory(self, timestamps: np.ndarray):
        pass


class FillProbabilityFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate_fill_probability(self, spread: NonNegativeFloat) -> NonNegativeFloat:
        pass


class ArrivalModel(metaclass=abc.ABCMeta):  # TODO: generalise
    @abc.abstractmethod
    def calculate_arrival_rate(self, interval_length: float) -> float:
        pass


class StochasticMidpriceModel(MidpriceModel):
    """A wrapper for the python package 'stochastic' to generate trajectories for a large class of processes"""

    def __init__(
        self, process: BaseProcess = None, initial_value: float = 1.0, is_multiplicative: bool = False, seed: int = None
    ):
        self.process = process or BrownianMotion(rng=np.random.default_rng(seed), scale=2.0)  # As in Avellaneda-Stoikov
        self.initial_value = initial_value
        self.is_multiplicative = is_multiplicative

    def generate_trajectory(self, timestamps: np.ndarray):
        if self.is_multiplicative:
            return self.process.sample_at(timestamps, initial=self.initial_value)
        else:
            return self.process.sample_at(timestamps) + self.initial_value


class ExponentialFillFunction(FillProbabilityFunction):
    def __init__(self, fill_exponent: float = 1.5):
        self.fill_exponent = fill_exponent

    def calculate_fill_probability(self, spread: NonNegativeFloat) -> NonNegativeFloat:
        return np.exp(-self.fill_exponent * spread)


class PoissonArrivalModel(ArrivalModel):
    def __init__(self, intensity: float = 140):
        self.intensity = intensity

    def calculate_arrival_rate(self, interval_length: float) -> float:
        return self.intensity * interval_length
