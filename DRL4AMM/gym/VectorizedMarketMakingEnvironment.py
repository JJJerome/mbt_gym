from typing import List, Any, Type, Sequence, Optional, Union

from copy import copy

import gym
import numpy as np
from gym.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from DRL4AMM.gym.probability_models import (
    MidpriceModel,
    ArrivalModel,
    FillProbabilityModel,
    BrownianMotionMidpriceModel,
    PoissonArrivalModel,
    ExponentialFillFunction,
)
from DRL4AMM.rewards.RewardFunctions import RewardFunction, PnL


class VectorizedMarketMakingEnvironment(VecEnv):
    def __init__(
        self,
        terminal_time: float = 30.0,
        n_steps: int = 30 * 10,
        reward_function: RewardFunction = None,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        action_type: str = "limit",
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        max_inventory: int = 100,
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        market_order_penalty: float = None,
        seed: int = None,
        num_trajectories: int = 1000,
    ):
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or PnL()
        self.midprice_model: MidpriceModel = midprice_model or BrownianMotionMidpriceModel(
            step_size=self.terminal_time / self.n_steps, num_trajectories=num_trajectories
        )
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel(
            step_size=self.terminal_time / self.n_steps, num_trajectories=num_trajectories
        )
        self.fill_probability_model: FillProbabilityModel = fill_probability_model or ExponentialFillFunction(
            step_size=self.terminal_time / self.n_steps, num_trajectories=num_trajectories
        )
        self.action_type = action_type
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self.fill_probability_model.max_depth
        self.rng = np.random.default_rng(seed)
        self.num_trajectories = num_trajectories
        self.dt = self.terminal_time / self.n_steps
        self.book_half_spread = market_order_penalty
        self.initial_state = self._get_initial_state()
        self.state = self.initial_state
        self.actions = np.zeros((self.num_trajectories, 2))
        self._check_model_params()
        self.empty_infos = [{} for _ in range(self.num_trajectories)]

        super().__init__(self.num_trajectories, self._get_observation_space(), self._get_action_space())

    def reset(self) -> VecEnvObs:
        self.midprice_model.reset()
        self.arrival_model.reset()
        self.fill_probability_model.reset()
        self.state = self._get_initial_state()
        return copy(self.state)[:, 1:3]

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        current_state = copy(self.state)
        next_state = self._update_state(self.actions)
        done = self.state[0, 2] >= self.terminal_time - self.dt / 2
        rewards = self.reward_function.calculate(current_state, self.actions, next_state, done)
        infos = self.empty_infos
        return copy(self.state)[:, 1:3], rewards, done, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.num_trajectories)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass

    def _update_state(self, action: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = action[:, 0:2]
            fills = self.fill_probability_model.get_hypothetical_fills(depths)
        else:
            fills = action[:, 0:2]
        self.arrival_model.update(arrivals, fills, action)  # TODO
        self.midprice_model.update(arrivals, fills, action)  # TODO
        self.fill_probability_model.update(arrivals, fills, action)  # TODO
        self._update_agent_state(arrivals, fills, action)  # TODO
        self._update_market_state()
        return copy(self.state)

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        multiplier = np.concatenate((np.ones((self.num_trajectories, 1)), -np.ones((self.num_trajectories, 1))), axis=1)
        if self.action_type == "limit_and_market":
            raise NotImplementedError
        self.state[:, 1] += np.sum(arrivals * fills * -multiplier, axis=1)  # Update inventory
        if self.action_type == "touch":
            self.state[:, 0] += np.sum(
                multiplier
                * arrivals
                * fills
                * (self.state[:, 3].reshape(-1, 1).repeat(2, axis=1) + self.book_half_spread * multiplier),
                axis=1,
            )
        else:
            depths = action[:, 0:2]
            self.state[:, 0] += np.sum(
                multiplier
                * arrivals
                * fills
                * (self.state[:, 3].reshape(-1, 1).repeat(2, axis=1) + depths * multiplier),
                axis=1,
            )
        self._clip_inventory_and_cash()
        self.state[:, 2] += self.dt * np.ones((self.num_trajectories,))

    def _update_market_state(self):
        len_midprice_state = self.midprice_model.current_state.shape[1]
        len_arrival_state = self.arrival_model.current_state.shape[1]
        len_fill_prob_state = self.fill_probability_model.current_state.shape[1]
        index = 3  # length of the agent_state
        self.state[:, index : index + len_midprice_state] = self.midprice_model.current_state
        index += len_midprice_state
        self.state[:, index : index + len_arrival_state] = self.arrival_model.current_state
        index += len_arrival_state
        self.state[:, index : index + len_fill_prob_state] = self.fill_probability_model.current_state

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    def _clip_inventory_and_cash(self):
        self.state[:, 1] = self._clip(self.state[:, 1], -self.max_inventory, self.max_inventory, cash_flag=False)
        self.state[:, 0] = self._clip(self.state[:, 0], -self.max_cash, self.max_cash, cash_flag=True)

    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped).all() and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped).all() and ~cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    def _get_initial_state(self):
        scalar_initial_state = np.array([[self.initial_cash, self.initial_inventory, 0.0]])
        initial_state = np.repeat(scalar_initial_state, self.num_trajectories, axis=0)
        initial_state = np.append(initial_state, self.midprice_model.current_state, axis=1)
        initial_state = np.append(initial_state, self.arrival_model.current_state, axis=1)
        initial_state = np.append(initial_state, self.fill_probability_model.current_state, axis=1)
        return initial_state

    def _check_model_params(self):
        for model_name in ["midprice_model", "arrival_model", "fill_probability_model"]:
            model = getattr(self, model_name)
            assert self.num_trajectories == model.num_trajectories, (
                f"Environement num trajectories = {self.num_trajectories},"
                + f"but {model_name}.num_trajectories = {model.num_trajectories}."
            )

    # observation space is (cash, inventory, time, stock price)
    def _get_observation_space(self):
        return Box(
            low=np.array([[-self.max_cash, -self.max_inventory, 0, 0]]),
            high=np.array([[self.max_cash, self.max_inventory, self.terminal_time, self.max_stock_price]]),
            dtype=np.float64,
        )

    def _get_action_space(self):
        return Box(low=0.0, high=self.max_depth, shape=(1, 2))  # agent chooses spread on bid and ask


# import torch
#
# from torch import sqrt, tensor
#
# from gym.spaces import Box
# from ray.rllib.env.vector_env import VectorEnv
# from ray.rllib.utils.annotations import override
#
# from DRL4AMM.rewards.RewardFunctions import RewardFunction, CjCriterion
#
#
# class VectorizedAvellanedaStoikov(VectorEnv):
#     def __init__(
#         self,
#         terminal_time: float = 30.0,
#         n_steps: int = 30 * 10,
#         reward_function: RewardFunction = None,
#         drift: float = 0.0,
#         volatility: float = 0.01,
#         arrival_rate: float = 1.0,
#         fill_exponent: float = 100.0,
#         initial_cash: float = 0.0,
#         initial_inventory: int = 0,
#         initial_stock_price: float = 100.0,
#         max_inventory: int = 100,
#         max_cash: float = None,
#         max_stock_price: float = None,
#         max_half_spread: float = 4.0,
#         seed: int = None,
#         num_envs: int = 1000,
#     ):
#         self.terminal_time = terminal_time
#         self.n_steps = n_steps
#         self.reward_function = reward_function or CjCriterion(phi=2 * 10 ** (-4), alpha=0.0001)
#         self.drift = drift
#         self.volatility = tensor(volatility)
#         self.arrival_rate = arrival_rate
#         self.fill_exponent = fill_exponent
#         self.initial_cash = initial_cash
#         self.initial_inventory = initial_inventory
#         self.initial_stock_price = initial_stock_price
#         self.max_inventory = max_inventory
#         self.max_cash = max_cash or self.get_max_cash(initial_cash, initial_stock_price, arrival_rate, terminal_time)
#         self.max_stock_price = max_stock_price or self.get_max_stock_price(
#             initial_stock_price, volatility, terminal_time
#         )
#         self.max_half_spread = max_half_spread
#         self.rng = np.random.default_rng(seed)
#         self.num_envs = num_envs
#         self.dt = tensor(self.terminal_time / self.n_steps)
#         self.max_inventory_exceeded_penalty = self.initial_stock_price * self.volatility * self.dt * 10
#
#         # observation space is (stock price, cash, inventory, step_number)
#         observation_space = Box(
#             low=np.array([0, -self.max_cash, -self.max_inventory, 0]),
#             high=np.array([self.max_stock_price, self.max_cash, self.max_inventory, terminal_time]),
#             dtype=np.float64,
#         )
#         action_space = Box(low=0.0, high=self.max_half_spread, shape=(2,))  # agent chooses spread on bid and ask
#         self.state = tensor()
#
#         super().__init__(observation_space=observation_space, action_space=action_space, num_envs=self.num_envs)
#
#     @override(VectorEnv)
#     def vector_reset(self):
#         self.state = torch.cat(torch.ones(n))
#         paths, bid_arrivals, ask_arrivals, rand_fill = self.get_randomness()
#
#         pass
#
#     def reset(self):
#         self.state = np.array([self.initial_stock_price, self.initial_cash, self.initial_inventory, 0])
#         return self.state
#
#     @override(VectorEnv)
#     def vector_step(self, actions):
#         pass
#
#     def generate_prices(self):
#         brownian_motion = torch.cat(
#             (
#                 torch.zeros((1, self.num_envs)),
#                 sqrt(self.volatility * self.dt) * torch.randn((self.n - 1, self.num_envs)).cumsum(dim=0),
#             ),
#             dim=0,
#         )
#
#         bm = torch.reshape(bm, (self.n, self.num_envs))
#         time = torch.reshape(torch.linspace(0, self.T, self.n), (self.n, 1))
#         path = torch.cat((time, bm), dim=1)
#         path[:, 1] += self.drift * path[:, 0]
#         return path
#
#     def get_randomness(self):
#         dt = torch.tensor(self.sigma ** 2 * self.T / self.n)
#         paths = self.generate_price(self.num_envs)
#         bid_arrival_dist = torch.distributions.Bernoulli(torch.min(1. - torch.exp(-self.rate_b * dt), torch.ones([1])))
#         ask_arrival_dist = torch.distributions.Bernoulli(torch.min(1. - torch.exp(-self.rate_a * dt), torch.ones([1])))
#         bid_arrivals = bid_arrival_dist.sample([self.n, self.num_envs]).squeeze()
#         ask_arrivals = ask_arrival_dist.sample([self.n, self.num_envs]).squeeze()
#         rand_fill = torch.rand([self.n, self.num_envs, 2])
#         return paths, bid_arrivals, ask_arrivals, rand_fill
