from typing import Union, Tuple

import gym
import numpy as np

from gym.spaces import Box

from DRL4AMM.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from DRL4AMM.stochastic_processes.fill_probability_models import FillProbabilityModel, ExponentialFillFunction
from DRL4AMM.stochastic_processes.midprice_models import MidpriceModel, BrownianMotionMidpriceModel
from DRL4AMM.gym.tracking.InfoCalculator import InfoCalculator, ActionInfoCalculator
from DRL4AMM.rewards.RewardFunctions import RewardFunction, PnL

ACTION_SPACES = ["touch", "limit", "limit_and_market"]

CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
        reward_function: RewardFunction = None,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        action_type: str = "limit",
        initial_cash: float = 0.0,
        initial_inventory: Union[int, Tuple[float, float]] = 0,  # Either a deterministic initial inventory, or a tuple
        max_inventory: int = 10_000,  # representing the mean and variance of it.
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        minimum_tick_size: float = None,
        info_calculator: InfoCalculator = None,
        seed: int = None,
        num_trajectories: int = 1,
    ):
        super(TradingEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.num_trajectories = num_trajectories
        self.n_steps = n_steps
        self.step_size = self.terminal_time / self.n_steps
        self.reward_function = reward_function or PnL()
        self.midprice_model: MidpriceModel = midprice_model or BrownianMotionMidpriceModel(
            step_size=self.step_size, num_trajectories=num_trajectories
        )
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel(
            step_size=self.step_size, num_trajectories=num_trajectories
        )
        self.fill_probability_model: FillProbabilityModel = fill_probability_model or ExponentialFillFunction(
            step_size=self.step_size, num_trajectories=num_trajectories
        )
        self.action_type = action_type
        self.rng = np.random.default_rng(seed)
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.state = self.initial_state
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self.fill_probability_model.max_depth
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.minimum_tick_size = minimum_tick_size
        self.info_calculator = info_calculator or ActionInfoCalculator()
        self.midprice_index_range = self._get_midprice_index_range()
        self.arrival_index_range = self._get_fill_index_range()
        self.fill_index_range = self._get_fill_index_range()
        self._check_params()
        self.empty_infos = [{} for _ in range(self.num_trajectories)]
        ones = np.ones((self.num_trajectories, 1))
        self.multiplier = np.append(-ones, ones, axis=1)

    def reset(self):
        self.midprice_model.reset()
        self.arrival_model.reset()
        self.fill_probability_model.reset()
        self.state = self.initial_state
        return self.state.copy()

    def step(self, action: np.ndarray):
        current_state = self.state.copy()
        next_state = self._update_state(action)
        done = self.state[0, 2] >= self.terminal_time - self.step_size / 2
        dones = np.full((self.num_trajectories,), done, dtype=bool)
        rewards = self.reward_function.calculate(current_state, action, next_state, done)
        infos = self.empty_infos
        return self.state.copy(), rewards, dones, infos

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    # action = [bid_depth, ask_depth, MO_buy, MO_sell]
    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = self.limit_depths(action)
            fills = self.fill_probability_model.get_fills(depths)
        else:
            fills = self.post_at_touch(action)
        self._update_agent_state(arrivals, fills, action)
        self._update_market_state(arrivals, fills, action)
        return self.state

    def _update_market_state(self, arrivals, fills, action):
        self.arrival_model.update(arrivals, fills, action)
        self.midprice_model.update(arrivals, fills, action)
        self.fill_probability_model.update(arrivals, fills, action)
        self.state[:, self.midprice_index_range[0] : self.midprice_index_range[1]] = self.midprice_model.current_state
        self.state[:, self.arrival_index_range[0] : self.arrival_index_range[1]] = self.arrival_model.current_state
        self.state[:, self.fill_index_range[0] : self.fill_index_range[1]] = self.fill_probability_model.current_state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        if self.action_type == "limit_and_market":
            mo_buy = np.single(self.market_order_buy(action) > 0.5)
            mo_sell = np.single(self.market_order_sell(action) > 0.5)
            best_bid = self.midprice - self.minimum_tick_size
            best_ask = self.midprice + self.minimum_tick_size
            self.state[:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
            self.state[:, INVENTORY_INDEX] += mo_buy - mo_sell
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.multiplier, axis=1)
        if self.action_type == "touch":
            self.state[:, CASH_INDEX] += np.sum(
                self.multiplier * arrivals * fills * (self.midprice + self.minimum_tick_size * self.multiplier), axis=1
            )
        else:
            self.state[:, CASH_INDEX] += np.sum(
                self.multiplier * arrivals * fills * (self.midprice + self.limit_depths(action) * self.multiplier),
                axis=1,
            )
        self._clip_inventory_and_cash()
        self.state[:, TIME_INDEX] += self.step_size

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)

    def _clip_inventory_and_cash(self):
        self.state[:, 1] = self._clip(self.state[:, 1], -self.max_inventory, self.max_inventory, cash_flag=False)
        self.state[:, 0] = self._clip(self.state[:, 0], -self.max_cash, self.max_cash, cash_flag=True)

    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped).any() and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped).any() and ~cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    def limit_depths(self, action: np.ndarray):
        if self.action_type in ["limit", "limit_and_market"]:
            return action[:, 0:2]
        else:
            raise Exception('Bid depth only exists for action_type in ["limit", "limit_and_market"].')

    def market_order_buy(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[:, 2]
        else:
            raise Exception('Market order buy action only exists for action_type == "limit_and_market".')

    def market_order_sell(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[:, 3]
        else:
            raise Exception('Market order sell action only exists for action_type == "limit_and_market".')

    def post_at_touch(self, action: np.ndarray):
        if self.action_type == "touch":
            return action[:, 0:2]
        else:
            raise Exception('Post buy at touch action only exists for action_type == "touch".')

    @property
    def initial_state(self) -> np.ndarray:
        scalar_initial_state = np.array([[self.initial_cash, 0, 0.0]])
        initial_state = np.repeat(scalar_initial_state, self.num_trajectories, axis=0)
        if isinstance(self.initial_inventory, tuple) and len(self.initial_inventory) == 2:
            # initial_inventories = self.rng.normal(*self.initial_inventory, size=self.num_trajectories).round()
            initial_inventories = self.rng.integers(*self.initial_inventory, size=self.num_trajectories)
        elif isinstance(self.initial_inventory, int):
            initial_inventories = self.initial_inventory * np.ones((self.num_trajectories,))
        else:
            raise Exception("Initial inventory must be a tuple of length 2 or an int.")
        initial_state[:, 1] = initial_inventories
        initial_state = np.append(initial_state, self.midprice_model.initial_vector_state, axis=1)
        initial_state = np.append(initial_state, self.arrival_model.initial_vector_state, axis=1)
        initial_state = np.append(initial_state, self.fill_probability_model.initial_vector_state, axis=1)
        return initial_state

    def _get_current_state(self) -> np.ndarray:
        state = self.state[:, 0:3]
        state = np.append(state, self.midprice_model.current_state, axis=1)
        state = np.append(state, self.arrival_model.current_state, axis=1)
        state = np.append(state, self.fill_probability_model.current_state, axis=1)
        return state

    def _get_observation_space(self) -> gym.spaces.Space:
        """The observation space consists of a numpy array containg the agent's cash, the agent's inventory and the
        current time. It also contains the states of the arrival model, the midprice model and the fill probability
        model in that order."""
        low = np.array([-self.max_cash, -self.max_inventory, 0])
        low = np.append(low, self.arrival_model.min_value)
        low = np.append(low, self.midprice_model.min_value)
        low = np.append(low, self.fill_probability_model.min_value)
        high = np.array([self.max_cash, self.max_inventory, self.terminal_time])
        high = np.append(high, self.arrival_model.max_value)
        high = np.append(high, self.midprice_model.max_value)
        high = np.append(high, self.fill_probability_model.max_value)
        return Box(
            low=low,
            high=high,
            dtype=np.float64,
        )

    def _get_action_space(self) -> gym.spaces.Space:
        assert self.action_type in ACTION_SPACES, f"Action type {self.action_type} is not in {ACTION_SPACES}."
        if self.action_type == "touch":
            return gym.spaces.MultiBinary(2)  # agent chooses spread on bid and ask
        if self.action_type == "limit":
            return gym.spaces.Box(low=0.0, high=self.max_depth, shape=(2,))  # agent chooses spread on bid and ask
        if self.action_type == "limit_and_market":
            return gym.spaces.Box(
                low=np.zeros(
                    4,
                ),
                high=np.array(self.max_depth, self.max_depth, 1, 1),
                shape=(2,),
            )

    def _get_midprice_index_range(self):
        min_midprice_index = 3
        max_midprice_index = 3 + self.midprice_model.initial_vector_state.shape[1]
        return min_midprice_index, max_midprice_index

    def _get_arrival_index_range(self):
        min_arrival_index = self._get_midprice_index_range()[1]
        max_arrival_index = self._get_midprice_index_range()[1] + self.arrival_model.initial_vector_state.shape[1]
        return min_arrival_index, max_arrival_index

    def _get_fill_index_range(self):
        min_fill_index = self._get_arrival_index_range()[1]
        max_fill_index = self._get_arrival_index_range()[1] + self.fill_probability_model.initial_vector_state.shape[1]
        return min_fill_index, max_fill_index

    @staticmethod
    def _get_max_depth():
        return 4.0  # TODO: improve

    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)

    def _check_params(self):
        assert self.action_type in ["limit", "limit_and_market", "touch"]
        for stochastic_process in [self.midprice_model, self.arrival_model, self.fill_probability_model]:
            assert np.isclose(stochastic_process.step_size, self.step_size, 2), (
                f"{type(self.midprice_model).__name__}.step_size = {stochastic_process.step_size}, "
                + f" but env.step_size = {self.terminal_time/self.n_steps}"
            )
