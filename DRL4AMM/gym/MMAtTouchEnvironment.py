import gym
import numpy as np

from copy import deepcopy
from gym.spaces import Box, MultiBinary
from math import sqrt, isclose

from DRL4AMM.gym.models import Action
from DRL4AMM.rewards.RewardFunctions import RewardFunction, CJ_criterion


class MMAtTouchEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 300.0,
        n_steps: int = int(10 * 300 / 50),
        reward_function: RewardFunction = None,
        drift: float = 0.0,
        volatility: float = 0.001,
        arrival_rate_ask: float = 50.0 / 300,
        arrival_rate_bid: float = 50.0 / 300,
        half_spread: float = 0.01,
        mean_jump_size: float = 0.02,
        max_inventory: int = 20,
        max_cash: float = None,
        max_stock_price: float = None,
        max_inventory_exceeded_penalty: float = None,  # typing: ignore
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        initial_stock_price: float = 100.0,
        continuous_observation_space: bool = True,  # This permits us to use out of the box algos from Stable-baselines
        seed: int = None,
    ):
        super(MMAtTouchEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or CJ_criterion(phi=0.01, alpha=10 * 0.01)
        self.drift = drift
        self.volatility = volatility
        self.arrival_rate_ask = arrival_rate_ask
        self.arrival_rate_bid = arrival_rate_bid
        self.half_spread = half_spread
        self.mean_jump_size = mean_jump_size
        self.max_inventory = max_inventory
        self.max_cash = max_cash or initial_cash + max(arrival_rate_bid, arrival_rate_ask) * initial_stock_price * 5.0
        self.max_stock_price = max_stock_price or initial_stock_price * 2.0
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.initial_stock_price = initial_stock_price
        self.continuous_observation_space = continuous_observation_space
        self.rng = np.random.default_rng(seed)
        self.dt = self.terminal_time / self.n_steps
        self.max_inventory_exceeded_penalty = (
            max_inventory_exceeded_penalty or self.initial_stock_price * self.volatility * self.dt * 10
        )
        self.action_space = MultiBinary(2)  # agent chooses spread on bid and ask
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

    # state[0]=stock_price, state[1]=cash, state[2]=inventory, state[3]=time
    def _get_next_state(self, action: Action) -> np.ndarray:
        action = Action(*action)
        next_state = deepcopy(self.state)
        next_state[3] += self.dt
        unif_bid, unif_ask = self.rng.random(2)
        bid_arrival = True if unif_bid < self.arrival_prob_ask else False
        ask_arrival = True if unif_ask < self.arrival_prob_bid else False
        if bid_arrival:
            if action.bid:
                next_state[1] -= self.state[0] - self.half_spread * action.bid
                next_state[2] += 1
        if ask_arrival:
            if action.ask:
                next_state[1] += self.state[0] + self.half_spread * action.ask
                next_state[2] -= 1
        next_state[0] = self.get_next_asset_price()
        return next_state

    def get_next_asset_price(self):
        return self.state[0] + self.drift * self.dt + self.volatility * sqrt(self.dt) * self.rng.normal()

    def get_next_asset_price_pure_jump(self, bid_arrival: bool, ask_arrival: bool):
        # Midprice model driven purely by Poisson arrivals
        next_price = self.state[0]
        if bid_arrival:
            next_price += 2 * self.mean_jump_size - self.rng.random()
        if ask_arrival:
            next_price -= 2 * self.mean_jump_size - self.rng.random()
        return next_price

    def fill_prob(self, action: float) -> float:
        prob_market_arrival = 1.0 - np.exp(-self.arrival_rate * self.dt)
        return prob_market_arrival * action

    @property
    def arrival_prob_ask(self):
        return 1.0 - np.exp(-self.arrival_rate_ask * self.dt)

    @property
    def arrival_prob_bid(self):
        return 1.0 - np.exp(-self.arrival_rate_bid * self.dt)
