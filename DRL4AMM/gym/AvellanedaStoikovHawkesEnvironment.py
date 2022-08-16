import gym
import numpy as np

from copy import deepcopy
from gym.spaces import Box
from math import sqrt, isclose

from DRL4AMM.rewards.RewardFunctions import RewardFunction, CjCriterion


class AvellanedaStoikovEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 30.0,
        n_steps: int = 30 * 10,
        reward_function: RewardFunction = None,
        drift: float = 0.0,
        volatility: float = 0.01,
        baseline_arrival_rate: float = 1.0,
        alpha: float = 1.0,  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        beta: float = 1.0,
        fill_exponent: float = 100.0,
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        initial_stock_price: float = 100.0,
        max_inventory: int = 100,
        max_cash: float = None,
        max_stock_price: float = None,
        max_hawkes_intensity: float = None,
        max_half_spread: float = 4.0,  # TODO: think about
        seed: int = None,
    ):
        super(AvellanedaStoikovEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or CjCriterion(phi=2 * 10 ** (-4), alpha=0.0001)
        self.drift = drift
        self.volatility = volatility
        self.baseline_arrival_rate = baseline_arrival_rate
        self.alpha = alpha
        self.beta = beta
        self.fill_exponent = fill_exponent
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.initial_stock_price = initial_stock_price
        self.max_inventory = max_inventory
        self.max_cash = max_cash or self._get_max_cash(
            initial_cash, baseline_arrival_rate, terminal_time, initial_stock_price
        )
        self.max_stock_price = max_stock_price or self._get_max_stock_price(
            initial_stock_price, volatility, terminal_time
        )
        self.max_hawkes_intensity = max_hawkes_intensity or 10 * self.baseline_arrival_rate  # TODO: think about
        self.max_half_spread = max_half_spread
        self.rng = np.random.default_rng(seed)
        self.dt = self.terminal_time / self.n_steps

        # observation space is (stock price, cash, inventory, step_number, hawkes intensity)
        self.observation_space = Box(
            low=np.array([0, -self.max_cash, -self.max_inventory, 0, 0, 0]),
            high=np.array(
                [
                    self.max_stock_price,
                    self.max_cash,
                    self.max_inventory,
                    terminal_time,
                    self.max_hawkes_intensity,
                    self.max_hawkes_intensity,
                ]
            ),
            dtype=np.float64,
        )
        self.action_space = Box(low=0.0, high=self.max_half_spread, shape=(2,))  # agent chooses spread on bid and ask
        self.state: np.ndarray = np.array([])

    def reset(self):
        self.state = np.array(
            [
                self.initial_stock_price,
                self.initial_cash,
                self.initial_inventory,
                0,
                self.baseline_arrival_rate,
                self.baseline_arrival_rate,
            ]
        )
        return self.state

    def step(self, action: np.ndarray):
        next_state = self._get_next_state(action)
        done = isclose(next_state[3], self.terminal_time)  # due to floating point arithmetic
        reward = self.reward_function.calculate(self.state, action, next_state, done)
        self.state = next_state
        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass

    # state[0]=stock_price, state[1]=cash, state[2]=inventory, state[3]=time, state[4]=hawkes intensity
    def _get_next_state(self, action: np.ndarray) -> np.ndarray:
        next_state = deepcopy(self.state)
        next_state[0] += self.drift * self.dt + self.volatility * sqrt(self.dt) * self.rng.normal()
        next_state[3] += self.dt
        next_state[3] = np.round(
            next_state[3], decimals=3
        )  # due to floating point arithmetic in self.dt TODO: sort me out
        jump_probs = np.array([self._clamp(self.state[4] * self.dt), self._clamp(self.state[5] * self.dt)])
        jumps_occurred = self.rng.random(2) <= jump_probs
        next_state[[4, 5]] = (
            self.state[[4, 5]]
            + self.beta * (self.baseline_arrival_rate - self.state[[4, 5]]) * self.dt
            + self.alpha * jumps_occurred
        )
        fill_prob_bid, fill_prob_ask = self.fill_prob(action[0]), self.fill_prob(action[1])
        unif_bid, unif_ask = self.rng.random(2)
        if unif_bid > fill_prob_bid and unif_ask > fill_prob_ask:  # neither the agent's bid nor their ask is filled
            pass
        if unif_bid < fill_prob_bid and unif_ask > fill_prob_ask:  # only bid filled
            # Note that market order gets filled THEN asset midprice changes
            next_state[1] -= self.state[0] - action[0]
            next_state[2] += 1
        if unif_bid > fill_prob_bid and unif_ask < fill_prob_ask:  # only ask filled
            next_state[1] += self.state[0] + action[1]
            next_state[2] -= 1
        if unif_bid < fill_prob_bid and unif_ask < fill_prob_ask:  # both bid and ask filled
            next_state[1] += action[0] + action[1]
        return next_state

    def fill_prob(self, half_spread: float) -> float:
        prob_market_arrival = 1.0 - np.exp(-self.baseline_arrival_rate * self.dt)
        fill_prob = np.exp(-self.fill_exponent * half_spread)
        return min(prob_market_arrival * fill_prob, 1)

    @staticmethod
    def _get_max_stock_price(initial_stock_price, volatility, terminal_time):
        return initial_stock_price + 4 * volatility * terminal_time

    @staticmethod
    def _get_max_cash(
        initial_cash, baseline_arrival_rate, terminal_time, initial_stock_price
    ):  # TODO: update for Hawkes.
        return (
            initial_cash + 3 * baseline_arrival_rate * terminal_time * initial_stock_price
        )  # TODO:https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel

    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)
