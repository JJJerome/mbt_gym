import abc
from typing import Union

import numpy as np
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
        pass

    @abc.abstractmethod
    def reset(self, initial_state: np.ndarray):
        pass


class PnL(RewardFunction):
    """A simple profit and loss reward function of the 'mark-to-market' value of the agent's portfolio."""

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        assert len(current_state.shape) > 1, "Reward functions must be calculated on state matrices."
        current_market_value = (
            current_state[:, CASH_INDEX] + current_state[:, INVENTORY_INDEX] * current_state[:, ASSET_PRICE_INDEX]
        )
        next_market_value = (
            next_state[:, CASH_INDEX] + next_state[:, INVENTORY_INDEX] * next_state[:, ASSET_PRICE_INDEX]
        )
        return next_market_value - current_market_value

    def reset(self, initial_state: np.ndarray):
        pass


class CjOeCriterion(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
        terminal_time: float = 1.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent
        self.terminal_time = terminal_time
        self.initial_inventory = None
        self.episode_length = None

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.per_step_inventory_aversion * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
            - dt
            * self.terminal_inventory_aversion
            * (
                self.inventory_exponent
                * np.squeeze(action)
                * (current_state[:, INVENTORY_INDEX]) ** (self.inventory_exponent - 1)
                + self.initial_inventory**self.inventory_exponent * self.episode_length
            )
        )

    def reset(self, initial_state: np.ndarray):
        self.initial_inventory = initial_state[:, INVENTORY_INDEX]
        self.episode_length = self.terminal_time - initial_state[:, TIME_INDEX]


class CjMmCriterion(RewardFunction):
    """A version of the Cartea-Jaimungal criterion which uses Ito's lemma for Poisson processes to split the negative
    reward attributed to terminal inventory aversion over the trajectory of the inventory."""

    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
        terminal_time: float = 1.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent
        self.terminal_time = terminal_time
        self.initial_inventory = None
        self.episode_length = None

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.per_step_inventory_aversion * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
            - self.terminal_inventory_aversion
            * (
                next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
                - current_state[:, INVENTORY_INDEX] ** self.inventory_exponent
                + dt / self.episode_length * self.initial_inventory**self.inventory_exponent
            )
        )

    def reset(self, initial_state: np.ndarray):
        self.initial_inventory = initial_state[:, INVENTORY_INDEX]
        self.episode_length = self.terminal_time - initial_state[:, TIME_INDEX]


class RunningInventoryPenalty(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: float = 0.01,
        terminal_inventory_aversion: float = 0.0,
        inventory_exponent: float = 2.0,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, TIME_INDEX] - current_state[:, TIME_INDEX]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.per_step_inventory_aversion * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
            - self.terminal_inventory_aversion
            * int(is_terminal_step)
            * next_state[:, INVENTORY_INDEX] ** self.inventory_exponent
        )

    def reset(self, initial_state: np.ndarray):
        pass


# Cartea and Jaimungal criterion is the same as inventory adjusted PnL

CjCriterion = RunningInventoryPenalty


class ExponentialUtility(RewardFunction):
    def __init__(self, risk_aversion: float = 0.1):
        self.risk_aversion = risk_aversion

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        return (
            -np.exp(
                -self.risk_aversion
                * (next_state[:, CASH_INDEX] + next_state[:, INVENTORY_INDEX] * next_state[:, ASSET_PRICE_INDEX])
            )
            if is_terminal_step
            else 0
        )

    def reset(self, initial_state: np.ndarray):
        pass
