import gym
import numpy as np

from copy import deepcopy
from gym.spaces import Box
from math import sqrt, isclose

from RL4MM.gym.models import Action
from RL4MM.rewards.RewardFunctions import RewardFunction, PnL


class AvellanedaStoikovEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 200,
        reward_function: RewardFunction = None,
        drift: float = 0.0,
        volatility: float = 2.0,
        arrival_rate: float = 140.0,
        fill_exponent: float = 1.5,
        max_inventory: int = 100,
        max_cash: float = None,
        max_stock_price: float = None,
        max_half_spread: float = 10,
        initial_cash: float = 100.0,
        initial_inventory: int = 0,
        initial_stock_price: float = 100.0,
        continuous_observation_space: bool = True,  # This permits us to use out of the box algos from Stable-baselines
        seed: int = None,
    ):
        super(AvellanedaStoikovEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or PnL()
        self.drift = drift
        self.volatility = volatility
        self.arrival_rate = arrival_rate
        self.fill_exponent = fill_exponent
        self.max_inventory = max_inventory
        self.max_cash = max_cash or initial_cash * 100
        self.max_stock_price = max_stock_price or initial_stock_price * 100
        self.max_half_spread = max_half_spread
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.initial_stock_price = initial_stock_price
        self.continuous_observation_space = continuous_observation_space
        self.rng = np.random.default_rng(seed)
        self.dt = self.terminal_time / self.n_steps
        self.max_inventory_exceeded_penalty = self.initial_stock_price * self.volatility * self.dt * 10

        self.action_space = Box(low=0.0, high=self.max_half_spread, shape=(2,))  # agent chooses spread on bid and ask
        # observation space is (stock price, cash, inventory, step_number)
        self.observation_space = Box(
            low=np.array([0, -self.max_cash, -self.max_inventory, 0]),
            high=np.array([self.max_stock_price, self.max_cash, self.max_inventory, terminal_time]),
            dtype=np.float64,
        )
        self.state: np.ndarray = np.array([])

    def reset(self):
        self.state = np.array([self.initial_stock_price, self.initial_cash, self.initial_inventory, 0])
        return self.state

    def step(self, action: Action):
        next_state = self._get_next_state(action)
        done = isclose(next_state[3], self.terminal_time)  # due to floating point arithmetic
        reward = self.reward_function.calculate(self.state, action, next_state, done)
        if abs(next_state[2]) > self.max_inventory:
            reward -= self.max_inventory_exceeded_penalty
        self.state = next_state
        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass

    def _get_next_state(self, action: Action) -> np.ndarray:
        action = Action(*action)  # for SB learning algo
        next_state = deepcopy(self.state)
        next_state[0] += self.drift * self.dt + self.volatility * sqrt(self.dt) * self.rng.normal()
        next_state[3] += self.dt
        fill_prob_bid, fill_prob_ask = self.fill_prob(action.bid), self.fill_prob(action.ask)
        unif_bid, unif_ask = self.rng.random(2)
        if unif_bid > fill_prob_bid and unif_ask > fill_prob_ask:  # neither the agent's bid nor their ask is filled
            pass
        if unif_bid < fill_prob_bid and unif_ask > fill_prob_ask:  # only bid filled
            # Note that market order gets filled THEN asset midprice changes
            next_state[1] -= self.state[0] - action.bid
            next_state[2] += 1
        if unif_bid > fill_prob_bid and unif_ask < fill_prob_ask:  # only ask filled
            next_state[1] += self.state[0] + action.ask
            next_state[2] -= 1
        if unif_bid < fill_prob_bid and unif_ask < fill_prob_ask:  # both bid and ask filled
            next_state[1] += action.bid + action.ask
        return next_state

    def fill_prob(self, half_spread: float) -> float:
        return min(self.arrival_rate * np.exp(-self.fill_exponent * half_spread) * self.dt, 1)
