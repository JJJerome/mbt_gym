import abc
from math import sqrt
from typing import Optional

import numpy as np
from numpy.random import default_rng

from pydantic import NonNegativeFloat


class StochasticProcessModel(metaclass=abc.ABCMeta):
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        seed: int = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.terminal_time = terminal_time
        self.initial_state = initial_state
        self.current_state = initial_state
        self.rng = default_rng(seed)

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass


MidpriceModel = StochasticProcessModel


class FillProbabilityModel(StochasticProcessModel):
    def __init__(
        self, min_value: float, max_value: float, step_size: float, terminal_time: float, initial_state: float, seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, seed)

    @abc.abstractmethod
    def get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_hypothetical_fills(self, depths: np.ndarray) -> np.ndarray:
        unif = self.rng.uniform(size=2)
        return unif < self.get_fill_probabilities(depths)

    @property
    @abc.abstractmethod
    def max_depth(self) -> float:
        pass


class ArrivalModel(StochasticProcessModel):
    def __init__(
        self, min_value: float, max_value: float, step_size: float, terminal_time: float, initial_state: float, seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, seed)

    @abc.abstractmethod
    def get_arrivals(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray) -> np.ndarray:
        pass


########################################################################################################################
#             EXAMPLES                                                                                                 #
########################################################################################################################


class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 10.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
           min_value = initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price), 
           max_value = self._get_max_value(initial_price, terminal_time), 
           step_size = step_size, 
           terminal_time = terminal_time, 
           initial_state = initial_price, 
           seed = seed
        )
        
    def reset(self):
        self.current_state = self.initial_state

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray) -> float:
        self.current_state = (
            self.current_state
            + self.drift * self.step_size
            + self.volatility * sqrt(self.step_size) * self.rng.normal()
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        super().__init__(
           min_value = initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price), 
           max_value = self._get_max_value(initial_price, terminal_time), 
           step_size = step_size, 
           terminal_time = terminal_time, 
           initial_state = initial_price, 
           seed = seed
        )
        
    def reset(self):
        self.current_state = self.initial_state

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray) -> float:
        # Euler: current_midprice + self.drift * current_midprice * self.dt + self.volatility * current_midprice * sqrt(self.dt) * self.rng.normal()
        self.current_state = self.current_state * np.exp(
            (self.drift - self.volatility ** 2 / 2) * self.step_size
            + self.volatility * sqrt(self.step_size) * self.rng.normal()
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price ** 2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility ** 2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev


class ExponentialFillFunction(FillProbabilityModel):
    def __init__(self, fill_exponent: float = 1.5, step_size: float = 0.1, seed: Optional[int] = None):
        self.fill_exponent = fill_exponent
        super().__init__(
            min_value = np.array([]), 
            max_value = np.array([]), 
            step_size = step_size, 
            terminal_time = np.array([]), 
            initial_state = np.array([]), 
            seed = seed
        )

    def get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return np.exp(-self.fill_exponent * depths)

    def get_hypothetical_fills(self, depths: np.ndarray) -> np.ndarray:
        unif = self.rng.uniform(size=2)
        return unif < self.get_fill_probabilities(depths)

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent

    def reset(self):
        pass

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass



class PoissonArrivalModel(ArrivalModel):
    def __init__(self, intensity: np.ndarray = np.array([100., 100.]), step_size: float = 0.01, seed: Optional[int] = None):
        self.intensity = intensity
        super().__init__(
            min_value = np.array([]), 
            max_value = np.array([]), 
            step_size = step_size, 
            terminal_time = np.array([]), 
            initial_state = np.array([]), 
            seed = seed
        )

    def update(
        self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass

    def reset(self):
        pass       

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=2)
        return unif < self.intensity * self.step_size


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: np.ndarray = np.array([100., 100.]),
        step_size: float = 0.01,
        alpha: float = 2,
        beta: float = 0.5,
        terminal_time: float = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.alpha = alpha  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.beta = beta
        super().__init__(
            min_value = np.array([0, 0]), 
            max_value = self._get_max_arrival_rate(), 
            step_size = step_size, 
            terminal_time = terminal_time, 
            initial_state = baseline_arrival_rate, 
            seed = seed
        )

    def reset(self):
        self.current_state = self.baseline_arrival_rate

    def update(
        self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.beta * (self.baseline_arrival_rate - self.current_state) * self.step_size
            + self.alpha * arrivals
        )
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=2)
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10  # TODO: Improve this with 4*std

    # TODO: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel