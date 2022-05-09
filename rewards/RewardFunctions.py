import abc

import numpy as np

from pydantic import NonNegativeFloat, PositiveFloat

from RL4MM.base import State
from RL4MM.gym.models import Action


class RewardFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self, current_state: State, action: Action, next_state: State, is_terminal_step: bool = False
    ) -> float:
        pass


class PnL(RewardFunction):
    """A simple profit and loss reward function of the 'mark-to-market' value of the agent's portfolio."""

    def calculate(self, current_state, action, next_state, is_terminal_step=False) -> float:
        current_market_value = current_state[1] + current_state[0] * current_state[2]
        next_market_value = next_state[1] + next_state[0] * next_state[2]
        return next_market_value - current_market_value


class TerminalExponentialUtility(RewardFunction):
    def __init__(self, risk_aversion: NonNegativeFloat = 0.1):
        self.risk_aversion = risk_aversion

    def calculate(
        self, current_state: State, action: Action, next_state: State, is_terminal_step: bool = False
    ) -> float:
        return -np.exp(-self.risk_aversion * (next_state[1] + next_state[0] * next_state[2])) if is_terminal_step else 0


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
        self.pnl_reward = PnL()
        self.inventory_exponent = inventory_exponent
        self.step_size = step_size

    def calculate(self, current_state, action, next_state, is_terminal_step=False) -> float:
        inventory_aversion = (
            is_terminal_step * self.terminal_inventory_aversion
            + (1 - is_terminal_step) * self.step_size * self.per_step_inventory_aversion
        )

        return (
            self.pnl_reward.calculate(current_state, action, next_state)
            - inventory_aversion * abs(next_state[2]) ** self.inventory_exponent
        )


# class ExpectedPnL(RewardFunction):
#     def __init__(self):
#         drift: float = (0.0,)
#         volatility: float = (2.0,)
#         arrival_rate: float = (140.0,)
#         fill_exponent: float = (1.5,)
#
#     def calculate(self, current_state, action, next_state, is_terminal_step=False) -> float:
#         current_market_value = current_state[1] + current_state[0] * current_state[2]
#         next_market_value = next_state[1] + next_state[0] * next_state[2]
#         return next_market_value - current_market_value
