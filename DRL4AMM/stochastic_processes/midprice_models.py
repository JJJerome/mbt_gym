from math import sqrt
from typing import Optional

import numpy as np

from DRL4AMM.stochastic_processes.StochasticProcessModel import StochasticProcessModel

MidpriceModel = StochasticProcessModel


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


class OuMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
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
        self.current_state += -self.mean_reversion_speed * (
            self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1))
        ) + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: What should this be?


class ShortTermOuAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_process: OuMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_process = ou_process or OuMidpriceModel(initial_price = 0.0)
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_process.min_value,
                    ]
                ]
            ),
            max_value=np.array([[self._get_max_asset_price(initial_price, terminal_time), self.ou_process.max_value]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price,self.ou_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.current_state[:, 0] = self.current_state[:, 0] + self.ou_process.current_state * self.step_size * np.ones(
            (self.num_trajectories, 1)
        ) + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        self.ou_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_process.current_state

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?
