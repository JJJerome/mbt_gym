from copy import deepcopy

import gym
import numpy as np
import warnings
from scipy.linalg import expm

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment, INVENTORY_INDEX, TIME_INDEX
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel, TemporaryAndPermanentPriceImpact


class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = deepcopy(env.action_space)
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: tuple, env: gym.Env):
        self.fixed_action = fixed_action
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.fixed_action.reshape(1, -1), self.env.num_trajectories, axis=0)


class FixedSpreadAgent(Agent):
    def __init__(self, env: gym.Env, half_spread: float = 1.0, offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.array([[self.half_spread - self.offset, self.half_spread + self.offset]])
        return np.repeat(action, self.env.num_trajectories, axis=0)


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        bid = float(input(f"Current state is {state}. How large do you want to set midprice-bid half spread? "))
        ask = float(input(f"Current state is {state}. How large do you want to set ask-midprice half spread? "))
        return np.array([bid, ask])


class AvellanedaStoikovAgent(Agent):
    def __init__(self, risk_aversion: float = 0.1, env: TradingEnvironment = None):
        self.risk_aversion = risk_aversion
        self.env = env or TradingEnvironment()
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.midprice_model.volatility
        self.rate_of_arrival = self.env.arrival_model.intensity
        self.fill_exponent = self.env.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[:, 1]
        time = state[:, 2]
        action = self._get_action(inventory, time)
        if action.min() < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action

    def _get_price_adjustment(self, inventory: int, time: float) -> float:
        return inventory * self.risk_aversion * self.volatility**2 * (self.terminal_time - time)

    def _get_spread(self, time: float) -> float:
        if self.risk_aversion == 0:
            return 2 / self.fill_exponent  # Limit as risk aversion -> 0
        volatility_aversion_component = self.risk_aversion * self.volatility**2 * (self.terminal_time - time)
        fill_exponent_component = 2 / self.risk_aversion * np.log(1 + self.risk_aversion / self.fill_exponent)
        return volatility_aversion_component + fill_exponent_component

    def _get_action(self, inventory: int, time: float):
        bid_half_spread = (self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        ask_half_spread = (-self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        return np.append(bid_half_spread, ask_half_spread, axis=1)


class CarteaJaimungalMmAgent(Agent):
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
        action = np.zeros(shape=(self.num_trajectories, 2))
        for iq, q in enumerate(state[:, INVENTORY_INDEX]):
            inventory = q
            current_time = state[iq, TIME_INDEX]
            aux_action = np.array(self._calculate_deltas(current_time, inventory))
            action[iq, :] = aux_action[:, 0]
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
        z_vector = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i
            Amatrix[i, i] = -self.phi * self.kappa * inventory**2
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa * inventory**2)
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[0] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[1] * np.exp(-1)
        return Amatrix, z_vector


class CarteaJaimungalOeAgent(Agent):
    def __init__(
        self,
        phi: float = 2 * 10 ** (-4),
        alpha: float = 0.0001,
        env: TradingEnvironment = None,
    ):
        self.phi = phi
        self.alpha = alpha
        self.env = env or TradingEnvironment()
        self.price_impact_model = env.price_impact_model
        assert self.env.action_type == "speed"
        self.terminal_time = self.env.terminal_time
        self.temporary_price_impact = self.price_impact_model.temporary_impact_coefficient
        self.permanent_price_impact = self.price_impact_model.permanent_impact_coefficient
        self.num_trajectories = self.env.num_trajectories

    def get_action(self, state: np.ndarray):
        action = np.zeros(shape=(self.num_trajectories, 1))
        # The formulae below is in page 146 of Cartea, Jaimungal, Penalva (2015)
        # Algorithmic and High-Frequency Trading
        # Cambridge University Press
        gamma = np.sqrt(self.phi / self.temporary_price_impact)
        zeta = self.alpha - 0.5 * self.permanent_price_impact + np.sqrt(self.temporary_price_impact * self.phi)
        zeta /= self.alpha - 0.5 * self.permanent_price_impact - np.sqrt(self.temporary_price_impact * self.phi)
        initial_inventory = self.env.initial_inventory

        time_left = self.terminal_time - state[0, TIME_INDEX]
        action[:, :] = (
            gamma
            * initial_inventory
            * (
                (zeta * np.exp(gamma * time_left) + np.exp(-gamma * time_left))
                / (zeta * np.exp(gamma * self.terminal_time) - np.exp(-gamma * self.terminal_time))
            )
        )
        return action
