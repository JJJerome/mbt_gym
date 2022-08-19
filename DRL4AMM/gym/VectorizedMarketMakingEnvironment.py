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
from DRL4AMM.rewards.RewardFunctions import RewardFunction


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
        initial_stock_price: float = 100.0,
        max_inventory: int = 100,
        max_cash: float = None,
        max_stock_price: float = None,
        max_half_spread: float = 4.0,
        seed: int = None,
        num_trajectories: int = 1000,
    ):
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function
        self.midprice_model: MidpriceModel = midprice_model or BrownianMotionMidpriceModel(
            step_size=self.terminal_time / self.n_steps
        )
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel(
            step_size=self.terminal_time / self.n_steps
        )
        self.fill_probability_model: FillProbabilityModel = fill_probability_model or ExponentialFillFunction(
            step_size=self.terminal_time / self.n_steps
        )
        self.action_type = action_type
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.initial_stock_price = initial_stock_price
        self.max_inventory = max_inventory
        self.max_stock_price = max_stock_price or self._get_max_stock_price()
        self.max_cash = max_cash or self._get_max_cash()
        self.max_half_spread = max_half_spread
        self.rng = np.random.default_rng(seed)
        self.num_trajectories = num_trajectories
        self.dt = self.terminal_time / self.n_steps
        self.max_inventory_exceeded_penalty = self.initial_stock_price * self.volatility * self.dt * 10
        self.vec_env_step_return = self.initial_vec_env_step_return
        self.states = self.initial_state
        self.actions = np.zeros((self.num_trajectories, 2))

        # observation space is (cash, inventory, time, stock price)
        observation_space = Box(
            low=np.array([-self.max_cash, -self.max_inventory, 0, 0]),
            high=np.array([self.max_cash, self.max_inventory, terminal_time, self.max_stock_price]),
            dtype=np.float64,
        )
        action_space = Box(low=0.0, high=self.max_half_spread, shape=(2,))  # agent chooses spread on bid and ask

        super().__init__(self.num_trajectories, observation_space, action_space)

    def reset(self) -> VecEnvObs:
        self.states = self.initial_state
        return self.states

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        current_state = copy(self.states)
        self._update_states(self.actions)
        return self.vec_env_step_return

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

    def _update_state(self, action: np.ndarray) -> None:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            raise NotImplementedError
        else:
            fills = np.array([self.post_buy_at_touch(action), self.post_sell_at_touch(action)])
        self.arrival_model.update(arrivals, fills, action)  # TODO
        self.midprice_model.update(arrivals, fills, action)  # TODO
        self.fill_probability_model.update(arrivals, fills, action)  # TODO
        self._update_agent_state(arrivals, fills, action)  # TODO

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    def _get_max_stock_price(self) -> float:
        return self.initial_stock_price * 4  # Update!

    @property
    def initial_vec_env_step_return(self):
        return None

    @property
    def initial_state(self):
        scalar_initial_state = np.array([[self.initial_cash, self.initial_inventory, 0.0, self.initial_stock_price]])
        return np.repeat(scalar_initial_state, self.num_trajectories, axis=0)


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
