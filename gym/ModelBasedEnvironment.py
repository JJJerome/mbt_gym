from typing import Iterator

import gym
import numpy as np

from copy import deepcopy
from gym.spaces import Box
from math import isclose

from RL4MM.gym.models import Action
from RL4MM.rewards.RewardFunctions import RewardFunction, PnL
from RL4MM.gym.probability_models import (
    MidpriceModel,
    StochasticMidpriceModel,
    FillProbabilityFunction,
    ExponentialFillFunction,
    ArrivalModel,
    PoissonArrivalModel,
)


class ModelBasedEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 100,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_function: FillProbabilityFunction = None,
        reward_function: RewardFunction = None,
        initial_cash: float = 1000,
        initial_inventory: int = 0,
        max_inventory: int = 20,
        seed: int = None,
    ):
        super(ModelBasedEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.dt = self.terminal_time / self.n_steps
        self.timestamps: np.ndarray = np.linspace(0, terminal_time, n_steps + 1)
        self.midprice_model: MidpriceModel = midprice_model or StochasticMidpriceModel()
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel()
        self.fill_probability_function = fill_probability_function or ExponentialFillFunction()
        self.reward_function = reward_function or PnL()
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.rng = np.random.default_rng(seed)

        self.action_space = Box(low=0.0, high=np.inf, shape=(2,))  # agent chooses spread on bid and ask
        # observation space is (stock_price, cash, inventory, time)
        self.observation_space = Box(
            low=np.zeros(4),
            high=np.array([np.inf, np.inf, self.max_inventory, terminal_time]),
            dtype=np.float64,
        )
        self.obs: np.ndarray = self.observation_space.sample()
        self.trajectory: Iterator = iter([])

    def reset(self):
        self.trajectory = iter(self.midprice_model.generate_trajectory(self.timestamps))
        self.obs = np.array([next(self.trajectory), self.initial_cash, self.initial_inventory, 0])
        return self.obs

    def step(self, action: Action):
        next_obs = self._get_next_obs(action)
        done = isclose(next_obs[3], self.terminal_time)  # due to floating point arithmetic
        reward = self.reward_function.calculate(self.obs, action, next_obs, done)
        self.obs = next_obs
        return next_obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def _get_next_obs(self, action: Action) -> np.ndarray:
        next_obs = deepcopy(self.obs)
        next_obs[0] = next(self.trajectory)
        next_obs[3] += self.dt
        fill_prob_bid, fill_prob_ask = [
            self.arrival_model.calculate_arrival_rate(self.dt)
            * self.fill_probability_function.calculate_fill_probability(a)
            for a in action
        ]
        unif_bid, unif_ask = self.rng.random(2)
        if unif_bid > fill_prob_bid and unif_ask > fill_prob_ask:  # neither the agent's bid nor their ask is filled
            pass
        if unif_bid < fill_prob_bid and unif_ask > fill_prob_ask:  # only bid filled
            # Note that market order gets filled THEN asset midprice changes
            next_obs[1] -= self.obs[0] - action[0]
            next_obs[2] += 1
        if unif_bid > fill_prob_bid and unif_ask < fill_prob_ask:  # only ask filled
            next_obs[1] += self.obs[0] + action[1]
            next_obs[2] -= 1
        if unif_bid < fill_prob_bid and unif_ask < fill_prob_ask:  # both bid and ask filled
            next_obs[1] += action[0] + action[1]
        return next_obs
