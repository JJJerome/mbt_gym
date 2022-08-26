import abc
from typing import Union

import numpy as np

from pydantic import NonNegativeFloat, PositiveFloat
from DRL4AMM.gym.models import Action

CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> Union[float, np.ndarray]:
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


class InventoryAdjustedPnL(RewardFunction):
    def __init__(
        self,
        per_step_inventory_aversion: NonNegativeFloat = 0.01,
        terminal_inventory_aversion: NonNegativeFloat = 0.0,
        inventory_exponent: PositiveFloat = 2.0,
        step_size: float = 1.0 / 200,
    ):
        self.per_step_inventory_aversion = per_step_inventory_aversion
        self.terminal_inventory_aversion = terminal_inventory_aversion
        self.pnl = PnL()
        self.inventory_exponent = inventory_exponent
        self.step_size = step_size

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


# Cartea and Jaimungal criterion is the same as inventory adjusted PnL

CjCriterion = InventoryAdjustedPnL


class TerminalExponentialUtility(RewardFunction):
    def __init__(self, risk_aversion: NonNegativeFloat = 0.1):
        self.risk_aversion = risk_aversion

    def calculate(
        self, current_state: np.ndarray, action: Action, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        return (
            -np.exp(
                -self.risk_aversion
                * (next_state[:, CASH_INDEX] + next_state[:, INVENTORY_INDEX] * next_state[:, ASSET_PRICE_INDEX])
            )
            if is_terminal_step
            else 0
        )
