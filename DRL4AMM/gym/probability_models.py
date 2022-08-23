import abc
from copy import copy
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
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.terminal_time = terminal_time
        self.num_trajectories = num_trajectories
        self.initial_state = initial_state
        self.initial_vector_state = self._get_initial_vector_state()
        self._check_attribute_shapes()
        self.current_state = copy(self.initial_vector_state)
        self.rng = default_rng(seed)

    def reset(self):
        self.current_state = self.initial_vector_state

    @abc.abstractmethod
    def update(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass

    def _check_attribute_shapes(self):
        for name in ["initial_state", "min_value", "max_value"]:
            attribute = getattr(self, name)
            assert (
                len(attribute.shape) == 2 and attribute.shape[0] == 1
            ), f"Attribute {name} must be a vector of shape (1, state_size)."

    def _get_initial_vector_state(self) -> np.ndarray:
        initial_state = self.initial_state
        if isinstance(initial_state, list):
            initial_state = np.array([[self.initial_state]])
        return np.repeat(initial_state, self.num_trajectories, axis=0)


MidpriceModel = StochasticProcessModel


class FillProbabilityModel(StochasticProcessModel):
    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        """Note that _get_fill_probabilities can return a 'probability' greater than one. However, this is not an issue
        for it is only use is in `get_hypothetical_fills` below."""
        pass

    def get_hypothetical_fills(self, depths: np.ndarray) -> np.ndarray:
        assert depths.shape == (self.num_trajectories, 2), (
            "Depths must be a numpy array of shape "
            + f"({self.num_trajectories},2). Instead it is a numpy array of shape {depths.shape}."
        )
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self._get_fill_probabilities(depths)

    @property
    @abc.abstractmethod
    def max_depth(self) -> float:
        pass


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the order book. The first entry of arrivals represents an arrival
    of an exogenous SELL order (arriving on the buy side of the book) and the second entry represents an arrival of an
    exogenous BUY order (arriving on the sell side of the book).
    """

    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def get_arrivals(self) -> np.ndarray:
        pass


########################################################################################################################
#             EXAMPLES                                                                                                 #
########################################################################################################################


class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
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
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.current_state = self.current_state * np.exp(
            (self.drift - self.volatility**2 / 2) * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price**2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility**2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev


class ExponentialFillFunction(FillProbabilityModel):
    def __init__(
        self, fill_exponent: float = 1.5, step_size: float = 0.1, num_trajectories: int = 1, seed: Optional[int] = None
    ):
        self.fill_exponent = fill_exponent
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def _get_fill_probabilities(self, depths: np.ndarray) -> np.ndarray:
        return np.exp(-self.fill_exponent * depths)

    @property
    def max_depth(self) -> float:
        return -np.log(0.01) / self.fill_exponent

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass


class PoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        pass

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: np.ndarray = np.array([[10.0, 10.0]]),
        step_size: float = 0.01,
        jump_size: float = 40.0,
        mean_reversion_speed: float = 60.0,
        terminal_time: float = 1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.jump_size = jump_size  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.mean_reversion_speed = mean_reversion_speed
        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_arrival_rate(),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.ones((num_trajectories, 2)) * baseline_arrival_rate,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.mean_reversion_speed
            * (self.baseline_arrival_rate - self.current_state)
            * self.step_size
            * np.ones((self.num_trajectories, 2))
            + self.jump_size * arrivals
        )
        return self.current_state

    def get_arrivals(self) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10

    # TODO: Improve this with 4*std
    # See: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel
