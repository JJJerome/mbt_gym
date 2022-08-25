import gym
import numpy as np
import warnings
from scipy.linalg import expm

from pydantic import NonNegativeFloat

from DRL4AMM.agents.Agent import Agent
from DRL4AMM.gym.TradingEnvironment import TradingEnvironment


class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.env = env
        self.env.action_space.seed(seed)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.env.action_space.sample().reshape(1, -1), self.env.num_trajectories, axis = 0)


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: tuple, env:gym.Env):
        self.fixed_action = fixed_action
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.fixed_action.reshape(1, -1), self.env.num_trajectories, axis=0)


class FixedSpreadAgent(Agent):
    def __init__(self, half_spread: float = 1.0, offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.half_spread - self.offset, self.half_spread + self.offset])


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        bid = float(input(f"Current state is {state}. How large do you want to set midprice-bid half spread? "))
        ask = float(input(f"Current state is {state}. How large do you want to set ask-midprice half spread? "))
        return np.array([bid, ask])


class AvellanedaStoikovAgent(Agent):
    def __init__(self, risk_aversion: NonNegativeFloat = 0.1, env: TradingEnvironment = None):
        self.risk_aversion = risk_aversion
        self.env = env or TradingEnvironment()
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.midprice_model.volatility
        self.rate_of_arrival = self.env.arrival_model.intensity
        self.fill_exponent = self.env.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[1]
        time = state[2]
        action = self._get_action(inventory, time)
        if min(action) < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action

    def _get_price_adjustment(self, inventory: int, time: NonNegativeFloat) -> float:
        return inventory * self.risk_aversion * self.volatility**2 * (self.terminal_time - time)

    def _get_spread(self, time: NonNegativeFloat) -> float:
        if self.risk_aversion == 0:
            return 2 / self.fill_exponent  # Limit as risk aversion -> 0
        volatility_aversion_component = self.risk_aversion * self.volatility**2 * (self.terminal_time - time)
        fill_exponent_component = 2 / self.risk_aversion * np.log(1 + self.risk_aversion / self.fill_exponent)
        return volatility_aversion_component + fill_exponent_component

    def _get_action(self, inventory: int, time: NonNegativeFloat):
        bid_half_spread = self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2
        ask_half_spread = -self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2
        return np.array([bid_half_spread, ask_half_spread])


class CarteaJaimungalAgent(Agent):
    def __init__(
        self,
        phi: float = 2 * 10 ** (-4),
        alpha: float = 0.0001,
        env: TradingEnvironment = None,
        max_inventory: int = 10,
    ):
        self.phi = phi
        self.alpha = alpha
        self.env = env or TradingEnvironment()
        assert self.env.action_type == "limit"
        self.terminal_time = self.env.terminal_time
        self.lambdas = self.env.arrival_model.intensity
        self.kappa = self.env.fill_probability_model.fill_exponent
        self.max_inventory = max_inventory
        self.a_matrix, self.z_vector = self._calculate_a_and_z()
        self.large_depth = 10_000
        self.num_trajectories = self.env.num_trajectories

    def get_action(self, state: np.ndarray):
        action = np.zeros(shape=(self.num_trajectories,2))
        for iq, q in enumerate(state[:,1]):
            inventory = q
            current_time = state[iq,2]
            aux_action = np.array(self._calculate_deltas(current_time, inventory))
            action[iq,:] = aux_action[:,0]
        return action

    def _calculate_deltas(self, current_time: float, inventory: int):
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        index = np.clip(self.max_inventory - inventory, 0, 2 * self.max_inventory)
        index = int(index)
        h_0 = h_t[index]
        if inventory >= self.max_inventory:
            delta_minus = self.large_depth
        else:
            h_plus_one = h_t[index - 1]
            delta_minus = 1 / self.kappa - h_plus_one + h_0
        if inventory <= -self.max_inventory:
            delta_plus = self.large_depth
        else:
            h_minus_one = h_t[index + 1]
            delta_plus = 1 / self.kappa - h_minus_one + h_0
        return delta_minus, delta_plus

    def _calculate_ht(self, current_time: float) -> float:
        omega_function = self._calculate_omega(current_time)
        return 1 / self.kappa * np.log(omega_function)

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = 2 * self.max_inventory + 1
        Amatrix = np.zeros(shape=(matrix_size, matrix_size))
        z_vector = np.zeros(shape=(matrix_size,1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i
            Amatrix[i, i] = -self.phi * self.kappa * inventory**2
            z_vector[i,0] = np.exp(-self.alpha * self.kappa * inventory**2)
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[0] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[1] * np.exp(-1)
        return Amatrix, z_vector
