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
    def calculate_next_arrival_rates(self) -> float:
        pass

    @abc.abstractmethod
    def get_arrivals(self) -> float:
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


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
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
        # Euler: current_midprice + self.drift * current_midprice * self.dt + self.volatility * current_midprice * sqrt(self.dt) * self.rng.normal()
        return current_midprice*np.exp( (self.drift - self.volatility**2/2)*self.dt + self.volatility * sqrt(self.dt) * self.rng.normal() )

    @property
    def max_value(self):
        stdev = sqrt(self.initial_price**2 * np.exp(2*self.drift*self.terminal_time)*(np.exp(self.volatility**2*self.terminal_time) -1) )
        return self.initial_price*np.exp(self.drift*self.terminal_time) + 4 * stdev


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
    def __init__(
        self, intensity: float = 100, 
        seed: Optional[int] = None
    ):
        self.intensity = intensity
        self.rng = default_rng(seed)

    def calculate_next_arrival_rates(self) -> float:
        return self.intensity

    def get_arrivals(self, interval_length:float) -> float:
        unif = self.rng.uniform(size=2)
        return unif < self.intensity * interval_length


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self, 
        baseline_arrival_rate: float = 100, 
        alpha: float = 2, 
        beta: float = 0.5, 
        time_horizon: float = 1,
        seed: Optional[int] = None
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.alpha = alpha  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.beta = beta
        self.time_horizon = time_horizon
        self.rng = default_rng(seed)

    def calculate_next_arrival_rates(self, intensities:np.ndarray, arrivals:float, interval_length:float) -> float:
        next_intensities = ( intensities
            + self.beta * (self.baseline_arrival_rate - intensities) * interval_length
            + self.alpha * arrivals )
        return next_intensities

    def get_arrivals(self, intensities:np.ndarray, interval_length:float) -> float:
        unif = self.rng.uniform(size=2)
        return unif < intensities * interval_length
    
    @property
    def get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10 # TODO: Improve this with 4*std
