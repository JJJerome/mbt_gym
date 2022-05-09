import numpy as np
import warnings

from pydantic import NonNegativeFloat, PositiveInt

from RL4MM.agents.Agent import Agent


class AvellanedaStoikovAgent(Agent):
    def __init__(
        self,
        risk_aversion: NonNegativeFloat = 0.1,
        terminal_time: NonNegativeFloat = 1.0,
        volatility: NonNegativeFloat = 2.0,
        dt: NonNegativeFloat = None,
        rate_of_arrival: NonNegativeFloat = 140,
        fill_exponent: NonNegativeFloat = 1.5,
        n_steps: PositiveInt = 200,
    ):
        self.risk_aversion = risk_aversion
        self.terminal_time = terminal_time
        self.volatility = volatility
        self.dt = dt or 1 / n_steps
        self.rate_of_arrival = rate_of_arrival
        self.fill_exponent = fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[2]
        time = state[3] * self.dt
        action = self._get_action(inventory, time)
        if min(action) < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action

    def _get_price_adjustment(self, inventory: int, time: NonNegativeFloat) -> float:
        return inventory * self.risk_aversion * self.volatility ** 2 * (self.terminal_time - time)

    def _get_spread(self, time: NonNegativeFloat) -> float:
        if self.risk_aversion == 0:
            return 2 / self.fill_exponent  # Limit as risk aversion -> 0
        volatility_aversion_component = self.risk_aversion * self.volatility ** 2 * (self.terminal_time - time)
        fill_exponent_component = 2 / self.risk_aversion * np.log(1 + self.risk_aversion / self.fill_exponent)
        return volatility_aversion_component + fill_exponent_component

    def _get_action(self, inventory: int, time: NonNegativeFloat):
        bid_half_spread = self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2
        ask_half_spread = -self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2
        return np.array([bid_half_spread, ask_half_spread])
