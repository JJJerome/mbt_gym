from typing import Union, Tuple

import gym
import numpy as np

from gym.spaces import Box

from mbt_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel, ExponentialFillFunction
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel, BrownianMotionMidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel, TemporaryPowerPriceImpact
from mbt_gym.gym.info_calculation.InfoCalculator import InfoCalculator, ActionInfoCalculator
from mbt_gym.rewards.RewardFunctions import RewardFunction, PnL

MARKET_MAKING_ACTION_TYPES = ["touch", "limit", "limit_and_market"]
EXECUTION_ACTION_TYPES = ["speed"]
ACTION_TYPES = MARKET_MAKING_ACTION_TYPES + EXECUTION_ACTION_TYPES

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
        price_impact_model: PriceImpactModel = None,
        action_type: str = "limit",
        initial_cash: float = 0.0,
        initial_inventory: Union[int, Tuple[float, float]] = 0,  # Either a deterministic initial inventory, or a tuple
        max_inventory: int = 10_000,  # representing the mean and variance of it.
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        max_speed: float = None,
        half_spread: float = None,
        info_calculator: InfoCalculator = None,
        seed: int = None,
        num_trajectories: int = 1,
    ):
        super().__init__() #TradingEnvironment, self
        self.terminal_time = terminal_time
        self.num_trajectories = num_trajectories
        self.n_steps = n_steps
        self.step_size = self.terminal_time / self.n_steps
        self.reward_function = reward_function or PnL()
        self.midprice_model = midprice_model or BrownianMotionMidpriceModel(step_size=self.step_size, num_trajectories=num_trajectories, seed=seed)
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model 
        self.stochastic_processes = self._get_stochastic_processes()
        self.stochastic_process_indices = self._get_stochastic_process_indices()
        self.action_type = action_type
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self._check_params()
        self.rng = np.random.default_rng(seed)
        self.state = self.initial_state
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self._get_max_depth() 
        self.max_speed = max_speed or self._get_max_speed() 
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.half_spread = half_spread
        self.info_calculator = info_calculator or ActionInfoCalculator()
        self.empty_infos = [{} for _ in range(self.num_trajectories)] if self.num_trajectories > 1 else {}
        ones = np.ones((self.num_trajectories, 1))
        self.multiplier = np.append(-ones, ones, axis=1)
        #self._check_stochastic_processes # TODO: Implement

    def _get_stochastic_processes(self):
        stochastic_processes = []
        for process in [self.midprice_model, self.arrival_model, self.fill_probability_model, self.price_impact_model]:
            if process is not None and process.initial_vector_state.shape[1]>0:
                stochastic_processes.append(process)
        return stochastic_processes

    def _get_stochastic_process_indices(self):
        number_of_processes = len(self.stochastic_processes)
        indices = np.zeros((number_of_processes, 2))
        count = 3
        for i, process in enumerate(self.stochastic_processes):
            indices[i, 0] = count
            dimension = process.initial_vector_state.shape[1]
            indices[i, 1] = count + dimension
            count += dimension + 1
        return indices.astype(int)

    def reset(self):
        for process in self.stochastic_processes:
            process.reset()
        self.state = self.initial_state
        return self.state.copy()

    def step(self, action: np.ndarray):
        if action.shape != (self.num_trajectories, self.action_space.shape[0]):
            action = action.reshape(self.num_trajectories, self.action_space.shape[0])
        current_state = self.state.copy()
        next_state = self._update_state(action)
        done = self.state[0, 2] >= self.terminal_time - self.step_size / 2
        dones = np.full((self.num_trajectories,), done, dtype=bool)
        rewards = self.reward_function.calculate(current_state, action, next_state, done)
        infos = self.empty_infos
        return self.state.copy(), rewards, dones, infos

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    def _get_max_depth(self) -> float:
        if self.fill_probability_model is not None:
            return self.fill_probability_model.max_depth
        else:
            return None

    def _get_max_speed(self) -> float:
        if self.price_impact_model is not None:
            return self.price_impact_model.max_speed
        else:
            return None

    def _get_arrivals_and_fills(self, action: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = self.limit_depths(action)
            fills = self.fill_probability_model.get_fills(depths)
        elif self.action_type == "touch":
            fills = self.post_at_touch(action)
        return arrivals, fills

    # The action space depends on the action_type but bids always precede asks for limit and market order actions.
    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        if self.action_type in MARKET_MAKING_ACTION_TYPES:
            arrivals, fills = self._get_arrivals_and_fills(action)
        else:
            arrivals, fills = None, None
        self._update_agent_state(arrivals, fills, action)
        self._update_market_state(arrivals, fills, action)
        return self.state

    def _update_market_state(self, arrivals, fills, action):
        for i, process in enumerate(self.stochastic_processes):
            process.update(arrivals, fills, action)
            lower_index = self.stochastic_process_indices[i, 0]
            upper_index = self.stochastic_process_indices[i, 1]
            self.state[:, lower_index : upper_index] = process.current_state    
  

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        if self.action_type == "limit_and_market":
            mo_buy = np.single(self.market_order_buy(action) > 0.5)
            mo_sell = np.single(self.market_order_sell(action) > 0.5)
            best_bid = self.midprice - self.half_spread
            best_ask = self.midprice + self.half_spread
            self.state[:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
            self.state[:, INVENTORY_INDEX] += mo_buy - mo_sell
        if self.action_type == "touch":
            self.state[:, CASH_INDEX] += np.sum(
                self.multiplier * arrivals * fills * (self.midprice + self.half_spread * self.multiplier), axis=1
            )
            self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.multiplier, axis=1)
        elif self.action_type in MARKET_MAKING_ACTION_TYPES:
            self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.multiplier, axis=1)
            self.state[:, CASH_INDEX] += np.sum(
                self.multiplier * arrivals * fills * (self.midprice + self.limit_depths(action) * self.multiplier),
                axis=1,
            )
        if self.action_type in EXECUTION_ACTION_TYPES:
            price_impact = self.price_impact_model.get_impact(action)
            execution_price = self.midprice[0] + price_impact
            volume = action * self.step_size
            self.state[:, CASH_INDEX] -= np.squeeze(volume * execution_price)
            self.state[:, INVENTORY_INDEX] += np.squeeze(volume)
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
            initial_inventories = self.rng.integers(*self.initial_inventory, size=self.num_trajectories)
        elif isinstance(self.initial_inventory, int):
            initial_inventories = self.initial_inventory * np.ones((self.num_trajectories,))
        else:
            raise Exception("Initial inventory must be a tuple of length 2 or an int.")
        initial_state[:, 1] = initial_inventories
        for process in self.stochastic_processes:
            initial_state = np.append(initial_state, process.initial_vector_state, axis=1)
        return initial_state

    def _get_observation_space(self) -> gym.spaces.Space:
        """The observation space consists of a numpy array containg the agent's cash, the agent's inventory and the
        current time. It also contains the states of the arrival model, the midprice model and the fill probability
        model in that order."""
        low = np.array([-self.max_cash, -self.max_inventory, 0])
        high = np.array([self.max_cash, self.max_inventory, self.terminal_time])
        for process in self.stochastic_processes:
            low = np.append(low, process.min_value)
            high = np.append(high, process.max_value)
        return Box(
            low=low,
            high=high,
            dtype=np.float64,
        )

    def _get_action_space(self) -> gym.spaces.Space:
        assert self.action_type in ACTION_TYPES, f"Action type {self.action_type} is not in {ACTION_TYPES}."
        if self.action_type == "touch":
            return gym.spaces.MultiBinary(2)  # agent chooses spread on bid and ask
        if self.action_type == "limit":
            assert self.max_depth is not None, "For limit orders max_depth cannot be NoneType"
            return gym.spaces.Box(low=0.0, high=self.max_depth, shape=(2,))  # agent chooses spread on bid and ask
        if self.action_type == "limit_and_market":
            return gym.spaces.Box(
                low=np.zeros(
                    4,
                ),
                high=np.array(self.max_depth, self.max_depth, 1, 1),
                shape=(2,),
            )
        if self.action_type == "speed":
            return gym.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(1,))  # agent chooses speed of trading: positive buys, negative sells



    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)

    def _check_params(self):
        assert self.action_type in ACTION_TYPES
        for stochastic_process in self.stochastic_processes:
            assert np.isclose(stochastic_process.step_size, self.step_size, atol=0.0, rtol=0.01), (
                f"{type(self.midprice_model).__name__}.step_size = {stochastic_process.step_size}, "
                + f" but env.step_size = {self.terminal_time/self.n_steps}"
            )
            assert stochastic_process.num_trajectories == self.num_trajectories, (
                "The stochastic processes given to an instance of TradingEnvironment must match the number of "
                "trajectories specified."
            )

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        for process in self.stochastic_processes:
            process.seed(seed)
