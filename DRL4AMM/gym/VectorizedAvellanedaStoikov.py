import numpy as np
import torch

from torch import sqrt, tensor

from gym.spaces import Box
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override

from DRL4AMM.rewards.RewardFunctions import RewardFunction, CjCriterion


class VectorizedAvellanedaStoikov(VectorEnv):
    def __init__(
        self,
        terminal_time: float = 30.0,
        n_steps: int = 30 * 10,
        reward_function: RewardFunction = None,
        drift: float = 0.0,
        volatility: float = 0.01,
        arrival_rate: float = 1.0,
        fill_exponent: float = 100.0,
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        initial_stock_price: float = 100.0,
        max_inventory: int = 100,
        max_cash: float = None,
        max_stock_price: float = None,
        max_half_spread: float = 4.0,
        seed: int = None,
        num_envs: int = 1000,
    ):
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or CjCriterion(phi=2 * 10 ** (-4), alpha=0.0001)
        self.drift = drift
        self.volatility = tensor(volatility)
        self.arrival_rate = arrival_rate
        self.fill_exponent = fill_exponent
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.initial_stock_price = initial_stock_price
        self.max_inventory = max_inventory
        self.max_cash = max_cash or self.get_max_cash(initial_cash, initial_stock_price, arrival_rate, terminal_time)
        self.max_stock_price = max_stock_price or self.get_max_stock_price(
            initial_stock_price, volatility, terminal_time
        )
        self.max_half_spread = max_half_spread
        self.rng = np.random.default_rng(seed)
        self.num_envs = num_envs
        self.dt = tensor(self.terminal_time / self.n_steps)
        self.max_inventory_exceeded_penalty = self.initial_stock_price * self.volatility * self.dt * 10

        # observation space is (stock price, cash, inventory, step_number)
        observation_space = Box(
            low=np.array([0, -self.max_cash, -self.max_inventory, 0]),
            high=np.array([self.max_stock_price, self.max_cash, self.max_inventory, terminal_time]),
            dtype=np.float64,
        )
        action_space = Box(low=0.0, high=self.max_half_spread, shape=(2,))  # agent chooses spread on bid and ask
        self.state = tensor()

        super().__init__(observation_space=observation_space, action_space=action_space, num_envs=self.num_envs)

    @override(VectorEnv)
    def vector_reset(self):
        # self.state = torch.cat(torch.ones(n))
        # paths, bid_arrivals, ask_arrivals, rand_fill = get_randomness(env, nsims)

        pass

    def reset(self):
        self.state = np.array([self.initial_stock_price, self.initial_cash, self.initial_inventory, 0])
        return self.state

    @override(VectorEnv)
    def vector_step(self, actions):
        pass

    def generate_prices(self):
        brownian_motion = torch.cat(
            (
                torch.zeros((1, self.num_envs)),
                sqrt(self.volatility * self.dt) * torch.randn((self.n - 1, nsims)).cumsum(dim=0),
            ),
            dim=0,
        )

        bm = torch.reshape(bm, (self.n, nsims))
        time = torch.reshape(torch.linspace(0, self.T, self.n), (self.n, 1))
        path = torch.cat((time, bm), dim=1)
        path[:, 1] += self.drift * path[:, 0]
        return path

        @staticmethod
        def get_max_cash(
            initial_cash: float, initial_stock_price: float, arrival_rate: float, terminal_time: float
        ) -> float:
            return initial_cash + arrival_rate * terminal_time * initial_stock_price * 5.0

        @staticmethod
        def get_max_stock_price(initial_stock_price: float, volatility: float, terminal_time: float) -> float:
            return initial_stock_price + 4 * terminal_time * volatility
