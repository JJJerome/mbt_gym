import abc
from math import sqrt
from typing import Optional

import numpy as np
from numpy.random import default_rng

from pydantic import NonNegativeFloat


class StochasticProcessModel(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def min_value(self):
        pass

    @property
    @abc.abstractmethod
    def max_value(self):
        pass

    @property
    @abc.abstractmethod
    def step_size(self):
        pass

    @property
    @abc.abstractmethod
    def initial_state(self):
        pass

    @property
    @abc.abstractmethod
    def current_state(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray) -> float:
        pass


class MidpriceModel(StochasticProcessModel):
    @abc.abstractmethod
    def update_midprice(self):
        pass


class FillProbabilityModel(StochasticProcessModel):
    @abc.abstractmethod
    def get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def rng(self):
        pass

    @abc.abstractmethod
    def get_fills(self, depths: np.ndarray) -> np.ndarray:
        unif = self.rng.uniform()
        return unif < self.get_fill_probabilities(depths)

    @property
    @abc.abstractmethod
    def max_depth(self) -> float:
        pass


class ArrivalModel(StochasticProcessModel):
    @abc.abstractmethod
    def get_arrivals(self, interval_length: float) -> float:
        pass


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
        self.current_midprice = initial_price
        self.dt = dt
        self.terminal_time = terminal_time
        self.rng = default_rng(seed)

    def reset(self):
        self.current_midprice = self.initial_price

    def update_midprice(self):
        self.current_midprice = (
            self.current_midprice + self.drift * self.dt + self.volatility * sqrt(self.dt) * self.rng.normal()
        )
        return self.current_midprice

    @property
    def max_value(self):
        return self.initial_price + 4 * self.volatility * self.terminal_time

    @property
    def initial_state(self):
        return self.initial_price


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
        self.current_midprice = initial_price
        self.dt = dt
        self.terminal_time = terminal_time
        self.rng = default_rng(seed)

    def reset(self):
        self.current_midprice = self.initial_price

    def update_midprice(self):
        # Euler: current_midprice + self.drift * current_midprice * self.dt + self.volatility * current_midprice * sqrt(self.dt) * self.rng.normal()
        self.current_midprice = self.current_midprice * np.exp(
            (self.drift - self.volatility**2 / 2) * self.dt + self.volatility * sqrt(self.dt) * self.rng.normal()
        )
        return self.current_midprice

    @property
    def max_value(self):
        stdev = sqrt(
            self.initial_price**2
            * np.exp(2 * self.drift * self.terminal_time)
            * (np.exp(self.volatility**2 * self.terminal_time) - 1)
        )
        return self.initial_price * np.exp(self.drift * self.terminal_time) + 4 * stdev

    @property
    def initial_state(self):
        return self.initial_price


class ExponentialFillFunction(FillProbabilityModel):
    def __init__(self, fill_exponent: float = 1.5, step_size: float = 0.1, seed: Optional[int] = None):
        self.fill_exponent = fill_exponent
        self._step_size = step_size
        self.rng = default_rng(seed)

    def get_fill_probabilities(self, half_spread: NonNegativeFloat) -> NonNegativeFloat:
        return np.exp(-self.fill_exponent * half_spread)

    def get_fill(self, half_spread: NonNegativeFloat) -> NonNegativeFloat:
        unif = self.rng.uniform()
        return unif < self.get_fill_probabilities(half_spread)

    @property
    def min_value(self):
        return np.array([])

    @property
    def max_value(self):
        return np.array([])

    @property
    def step_size(self):
        return self._step_size

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent

    def reset(self):
        pass

    def update(self) -> float:
        pass

    @property
    def initial_state(self):
        return np.array([])


class PoissonArrivalModel(ArrivalModel):
    def __init__(self, intensity: float = 100, seed: Optional[int] = None):
        self.intensity = intensity
        self.rng = default_rng(seed)

    def update_arrival_rates(self, arrivals: float, interval_length: float) -> float:
        return self.intensity

    def get_arrivals(self, interval_length: float) -> float:
        unif = self.rng.uniform(size=2)
        return unif < self.intensity * interval_length

    @property
    def max_arrival_rate(self):
        return self.intensity

    @property
    def initial_state(self):
        return np.array([])  # We do not give the fixed intensity or the cumulative arrivals to the RL agent


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: float = 100,
        alpha: float = 2,
        beta: float = 0.5,
        time_horizon: float = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.current_arrival_rates = baseline_arrival_rate
        self.alpha = alpha  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.beta = beta
        self.time_horizon = time_horizon
        self.rng = default_rng(seed)

    def reset(self):
        self.current_arrival_rates = self.baseline_arrival_rate

    def update_arrival_rates(self, arrivals: float, interval_length: float) -> float:
        self.current_arrival_rates = (
            self.current_arrival_rates
            + self.beta * (self.baseline_arrival_rate - self.current_arrival_rates) * interval_length
            + self.alpha * arrivals
        )
        return self.current_arrival_rates

    def get_arrivals(self, interval_length: float) -> float:
        unif = self.rng.uniform(size=2)
        return unif < self.current_arrival_rates * interval_length

    @property
    def max_arrival_rate(self):
        return self.baseline_arrival_rate * 10  # TODO: Improve this with 4*std

    # TODO: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel

    @property
    def initial_state(self):
        return self.baseline_arrival_rate
