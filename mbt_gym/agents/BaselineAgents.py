from copy import deepcopy

import gym
import numpy as np
import warnings
from scipy.linalg import expm

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.index_names import INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX, CASH_INDEX, BID_INDEX, ASK_INDEX
from mbt_gym.rewards.RewardFunctions import CjMmCriterion, PnL
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel, TemporaryAndPermanentPriceImpact
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics, TradinghWithSpeedModelDynamics

class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = deepcopy(env.action_space)
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: np.ndarray, env: gym.Env):
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
        self.volatility = self.env.model_dynamics.midprice_model.volatility
        self.rate_of_arrival = self.env.model_dynamics.arrival_model.intensity
        self.fill_exponent = self.env.model_dynamics.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[:, INVENTORY_INDEX]
        time = state[:, TIME_INDEX]
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
        env: TradingEnvironment = None,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.model_dynamics, LimitOrderModelDynamics), "Trader must be type LimitOrderTrader"
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for CjMmAgent is incorrect."
        self.kappa = self.env.model_dynamics.fill_probability_model.fill_exponent
        self.num_trajectories = self.env.num_trajectories
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.model_dynamics.arrival_model.intensity
            self.max_inventory = env.max_inventory
            self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        indices_minus_one = np.clip(indices - 1, 0, 2 * self.max_inventory)
        indices_plus_one = np.clip(indices + 1, 0, 2 * self.max_inventory)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = h_plus_one == h_0
        max_inventory_ask = h_minus_one == h_0
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask).reshape(-1)
        return deltas

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
                Amatrix[i, i + 1] = self.lambdas[BID_INDEX] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[ASK_INDEX] * np.exp(-1)
        return Amatrix, z_vector
    
    def calculate_true_value_function(self, state: np.ndarray):
        current_time = state[0, TIME_INDEX]
        inventories = state[:, INVENTORY_INDEX]
        value_fct = np.zeros(shape=(self.num_trajectories, 1))
        h_t = self._calculate_ht(current_time)
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        h_0 = h_t[indices]
        value_fct = h_0 + state[:, CASH_INDEX] + state[:, INVENTORY_INDEX] * state[:, ASSET_PRICE_INDEX]
        return value_fct


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
        self.price_impact_model = env.model_dynamics.price_impact_model
        assert isinstance(self.env.model_dynamics, TradinghWithSpeedModelDynamics), "Trader must be type TradinghWithSpeedTrader"
        self.terminal_time = self.env.terminal_time
        self.temporary_price_impact = self.price_impact_model.temporary_impact_coefficient
        self.permanent_price_impact = self.price_impact_model.permanent_impact_coefficient
        self.num_trajectories = self.env.num_trajectories

    def get_action(self, state: np.ndarray):
        action = np.zeros(shape=(self.num_trajectories, 1))
        # The formulae below is in page 147 of Cartea, Jaimungal, Penalva (2015)
        # Algorithmic and High-Frequency Trading
        # Cambridge University Press
        gamma = np.sqrt(self.phi / self.temporary_price_impact)
        zeta = (self.alpha - 0.5 * self.permanent_price_impact + np.sqrt(self.temporary_price_impact * self.phi)) / (
            self.alpha - 0.5 * self.permanent_price_impact - np.sqrt(self.temporary_price_impact * self.phi)
        )
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
        return -np.sign(initial_inventory) * action
