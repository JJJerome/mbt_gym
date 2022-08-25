import abc
from typing import Union

import numpy as np

from pydantic import NonNegativeFloat, PositiveFloat
from DRL4AMM.gym.models import Action


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
        current_market_value = current_state[:, 0] + current_state[:, 1] * current_state[:, 3]
        next_market_value = next_state[:, 0] + next_state[:, 1] * next_state[:, 3]
        return next_market_value - current_market_value


# observation space is ([[stock_price, cash, inventory, time]])
class CjCriterion(RewardFunction):
    def __init__(self, phi: NonNegativeFloat = 0.01, alpha: NonNegativeFloat = 0.01):
        self.phi = phi
        self.alpha = alpha
        self.pnl = PnL()

    """Cartea-Jaimungal type performance."""

    def calculate(
        self, current_state: np.ndarray, action: np.ndarray, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        dt = next_state[:, 2] - current_state[:, 2]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt * self.phi * (next_state[:, 1] - current_state[:, 1]) ** 2
            - self.alpha * int(is_terminal_step) * (next_state[:, 1] - current_state[:, 1]) ** 2
        )


class TerminalExponentialUtility(RewardFunction):
    def __init__(self, risk_aversion: NonNegativeFloat = 0.1):
        self.risk_aversion = risk_aversion

    def calculate(
        self, current_state: np.ndarray, action: Action, next_state: np.ndarray, is_terminal_step: bool = False
    ) -> float:
        return (
            -np.exp(-self.risk_aversion * (next_state[:, 0] + next_state[:, 1] * next_state[:, 3]))
            if is_terminal_step
            else 0
        )


# TODO: note that CJ_criterion is just InventoryAdjustedPnL with inventory_exponent = 2
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
        dt = next_state[:, 2] - current_state[:, 2]
        return (
            self.pnl.calculate(current_state, action, next_state, is_terminal_step)
            - dt
            * self.per_step_inventory_aversion
            * (next_state[:, 1] - current_state[:, 1]) ** self.inventory_exponent
            - self.terminal_inventory_aversion
            * int(is_terminal_step)
            * (next_state[:, 1] - current_state[:, 1]) ** self.inventory_exponent
        )
