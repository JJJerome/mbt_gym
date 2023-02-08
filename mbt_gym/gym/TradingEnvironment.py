from collections import OrderedDict
from copy import copy, deepcopy
from typing import Union, Tuple, Callable, Optional

import gym
import numpy as np

from gym.spaces import Box

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from mbt_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel, ExponentialFillFunction
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel, BrownianMotionMidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel
from mbt_gym.gym.info_calculators import InfoCalculator
from mbt_gym.rewards.RewardFunctions import RewardFunction, PnL

MARKET_MAKING_ACTION_TYPES = ["touch", "limit", "limit_and_market"]
EXECUTION_ACTION_TYPES = ["speed"]
ACTION_TYPES = MARKET_MAKING_ACTION_TYPES + EXECUTION_ACTION_TYPES

CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3

BID_INDEX = 0
ASK_INDEX = 1


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
        fixed_market_half_spread: float = None,
        start_time: Union[float, int, Callable] = 0.0,
        info_calculator: InfoCalculator = None,  # episode given as a proportion.
        seed: int = None,
        num_trajectories: int = 1,
        normalise_action_space: bool = True,
        normalise_observation_space: bool = True,
        normalise_rewards: bool = False,
    ):
        super(TradingEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or PnL()
        self.midprice_model = midprice_model or BrownianMotionMidpriceModel(
            step_size=self._step_size, num_trajectories=num_trajectories
        )
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.action_type = action_type
        self._check_required_stochastic_processes()
        self.stochastic_processes = self._get_stochastic_processes()
        self.stochastic_process_indices = self._get_stochastic_process_indices()
        self.num_trajectories = num_trajectories
        self.step_size = self.terminal_time / self.n_steps
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        if seed:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.start_time = start_time
        self.state = self.initial_state
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self._get_max_depth()
        self.max_speed = max_speed or self._get_max_speed()
        self.fixed_market_half_spread = fixed_market_half_spread
        self.info_calculator = info_calculator
        self._empty_infos = self._get_empty_infos()
        self._fill_multiplier = self._get_fill_multiplier()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.normalise_action_space_ = normalise_action_space
        self.normalise_observation_space_ = normalise_observation_space
        self.normalise_rewards_ = normalise_rewards
        if self.normalise_observation_space_:
            self.original_observation_space = copy(self.observation_space)
            self.observation_space = self._get_normalised_observation_space()
        if self.normalise_action_space_:
            self.original_action_space = copy(self.action_space)
            self.action_space = self._get_normalised_action_space()
        if self.normalise_rewards_:
            assert isinstance(self.arrival_model, PoissonArrivalModel) and isinstance(
                self.fill_probability_model, ExponentialFillFunction
            ), "Arrival model must be Poisson and fill probability model must be exponential to scale rewards"
            self.reward_scaling = 1 / self._get_inventory_neutral_rewards()

    def reset(self):
        for process in self.stochastic_processes.values():
            process.reset()
        self.state = self.initial_state
        self.reward_function.reset(self.state.copy())
        return self.normalise_observation(self.state.copy())

    def step(self, action: np.ndarray):
        if action.shape != (self.num_trajectories, self.action_space.shape[0]):
            action = action.reshape(self.num_trajectories, self.action_space.shape[0])
        action = self.normalise_action(action, inverse=True)
        current_state = self.state.copy()
        next_state = self._update_state(action)
        done = self.state[0, TIME_INDEX] >= self.terminal_time - self.step_size / 2
        dones = np.full((self.num_trajectories,), done, dtype=bool)
        rewards = self.reward_function.calculate(current_state, action, next_state, done)
        infos = (
            self.info_calculator.calculate(current_state, action, rewards)
            if self.info_calculator is not None
            else self._empty_infos
        )
        return self.normalise_observation(next_state.copy()), self.normalise_rewards(rewards), dones, infos

    def normalise_observation(self, obs: np.ndarray, inverse: bool = False):
        if self.normalise_observation_space_ and not inverse:
            return (obs - self._intercept_obs_norm) / self._gradient_obs_norm - 1
        elif self.normalise_observation_space_ and inverse:
            return (obs + 1) * self._gradient_obs_norm + self._intercept_obs_norm
        else:
            return obs

    def normalise_action(self, action: np.ndarray, inverse: bool = False):
        if self.normalise_action_space_ and not inverse:
            return (action - self._intercept_action_norm) / self._gradient_action_norm - 1
        elif self.normalise_action_space_ and inverse:
            return (action + 1) * self._gradient_action_norm + self._intercept_action_norm
        else:
            return action

    def normalise_rewards(self, rewards: np.ndarray):
        return self.reward_scaling * rewards if self.normalise_rewards_ else rewards

    @property
    def initial_state(self) -> np.ndarray:
        scalar_initial_state = np.array([[self.initial_cash, 0, 0.0]])
        initial_state = np.repeat(scalar_initial_state, self.num_trajectories, axis=0)
        start_time = self._get_start_time()
        initial_state[:, TIME_INDEX] = start_time * np.ones((self.num_trajectories,))
        initial_state[:, INVENTORY_INDEX] = self._get_initial_inventories()
        for process in self.stochastic_processes.values():
            initial_state = np.append(initial_state, process.initial_vector_state, axis=1)
        return initial_state

    @property
    def is_at_max_inventory(self):
        return self.state[:, INVENTORY_INDEX] >= self.max_inventory

    @property
    def is_at_min_inventory(self):
        return self.state[:, INVENTORY_INDEX] <= -self.max_inventory

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, step_size: float):
        self._step_size = step_size
        for process_name, process in self.stochastic_processes.items():
            if process.step_size != step_size:
                process.step_size = step_size
        if hasattr(self.reward_function, "step_size"):
            self.reward_function.step_size = step_size

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @num_trajectories.setter
    def num_trajectories(self, num_trajectories: float):
        self._num_trajectories = num_trajectories
        for process_name, process in self.stochastic_processes.items():
            if process.num_trajectories != num_trajectories:
                process.num_trajectories = num_trajectories
        self._empty_infos = self._get_empty_infos()
        self._fill_multiplier = self._get_fill_multiplier()

    @property
    def _intercept_obs_norm(self):
        return self.original_observation_space.low

    @property
    def _gradient_obs_norm(self):
        return (self.original_observation_space.high - self.original_observation_space.low) / 2

    @property
    def _intercept_action_norm(self):
        return self.original_action_space.low

    @property
    def _gradient_action_norm(self):
        return (self.original_action_space.high - self.original_action_space.low) / 2

    # The action space depends on the action_type but bids always precede asks for limit and market order actions.
    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        if self.action_type in MARKET_MAKING_ACTION_TYPES:
            arrivals, fills = self._get_arrivals_and_fills(action)
        else:
            arrivals, fills = None, None
        if self.action_type in EXECUTION_ACTION_TYPES:
            action = self._get_execution_action(action)
        self._update_agent_state(arrivals, fills, action)
        self._update_market_state(arrivals, fills, action)
        return self.state

    def _update_market_state(self, arrivals, fills, action):
        for process_name, process in self.stochastic_processes.items():
            process.update(arrivals, fills, action)
            lower_index = self.stochastic_process_indices[process_name][0]
            upper_index = self.stochastic_process_indices[process_name][1]
            self.state[:, lower_index:upper_index] = process.current_state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        if self.action_type == "limit_and_market":
            mo_buy = np.single(self._market_order_buy(action) > 0.5)
            mo_sell = np.single(self._market_order_sell(action) > 0.5)
            best_bid = self.midprice - self.fixed_market_half_spread
            best_ask = self.midprice + self.fixed_market_half_spread
            self.state[:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
            self.state[:, INVENTORY_INDEX] += mo_buy - mo_sell
        elif self.action_type == "touch":
            self.state[:, CASH_INDEX] += np.sum(
                self._fill_multiplier * arrivals * fills * (self.midprice + self.fixed_market_half_spread * self._fill_multiplier),
                axis=1,
            )
            self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self._fill_multiplier, axis=1)
        elif self.action_type in ["limit", "limit_and_market"]:
            self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self._fill_multiplier, axis=1)
            self.state[:, CASH_INDEX] += np.sum(
                self._fill_multiplier * arrivals * fills * (self.midprice + self._limit_depths(action) * self._fill_multiplier),
                axis=1,
            )
        if self.action_type in EXECUTION_ACTION_TYPES:
            price_impact = self.price_impact_model.get_impact(action)
            execution_price = self.midprice + price_impact
            volume = action * self.step_size
            self.state[:, CASH_INDEX] -= np.squeeze(volume * execution_price)
            self.state[:, INVENTORY_INDEX] += np.squeeze(volume)
        self._clip_inventory_and_cash()
        self.state[:, TIME_INDEX] += self.step_size

    def _get_arrivals_and_fills(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = self._limit_depths(action)
            fills = self.fill_probability_model.get_fills(depths)
        elif self.action_type == "touch":
            fills = self._post_at_touch(action)
        else:
            raise NotImplementedError
        fills = self._remove_max_inventory_fills(fills)
        return arrivals, fills

    def _remove_max_inventory_fills(self, fills: np.ndarray) -> np.ndarray:
        fill_multiplier = np.concatenate(
            ((1 - self.is_at_max_inventory).reshape(-1, 1), (1 - self.is_at_min_inventory).reshape(-1, 1)), axis=1
        )
        return fill_multiplier * fills

    def _limit_depths(self, action: np.ndarray):
        if self.action_type in ["limit", "limit_and_market"]:
            return action[:, 0:2]
        else:
            raise Exception('Bid depth only exists for action_type in ["limit", "limit_and_market"].')

    def _market_order_buy(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[:, 2]
        else:
            raise Exception('Market order buy action only exists for action_type == "limit_and_market".')

    def _market_order_sell(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[:, 3]
        else:
            raise Exception('Market order sell action only exists for action_type == "limit_and_market".')

    def _post_at_touch(self, action: np.ndarray):
        if self.action_type == "touch":
            return action[:, 0:2]
        else:
            raise Exception('Post buy at touch action only exists for action_type == "touch".')

    def _get_max_cash(self) -> float:
        return self.n_steps * self.max_stock_price  # TODO: make this a tighter bound

    def _get_max_depth(self) -> Optional[float]:
        if self.fill_probability_model is not None:
            return self.fill_probability_model.max_depth
        else:
            return None

    def _get_max_speed(self) -> float:
        if self.price_impact_model is not None:
            return self.price_impact_model.max_speed
        else:
            return None

    def _get_observation_space(self) -> gym.spaces.Space:
        """The observation space consists of a numpy array containg the agent's cash, the agent's inventory and the
        current time. It also contains the states of the arrival model, the midprice model and the fill probability
        model in that order."""
        low = np.array([-self.max_cash, -self.max_inventory, 0])
        high = np.array([self.max_cash, self.max_inventory, self.terminal_time])
        for process in self.stochastic_processes.values():
            low = np.append(low, process.min_value)
            high = np.append(high, process.max_value)
        return Box(low=np.float32(low), high=np.float32(high))

    def _get_action_space(self) -> gym.spaces.Space:
        if self.action_type == "touch":
            return gym.spaces.MultiBinary(2)  # agent chooses spread on bid and ask
        elif self.action_type == "limit":
            assert self.max_depth is not None, "For limit orders max_depth cannot be None."
            # agent chooses spread on bid and ask
            return gym.spaces.Box(low=np.float32(0.0), high=np.float32(self.max_depth), shape=(2,))
        elif self.action_type == "limit_and_market":
            return gym.spaces.Box(
                low=np.zeros(4),
                high=np.array([self.max_depth, self.max_depth, 1, 1], dtype=np.float32),
            )
        elif self.action_type == "speed":
            # agent chooses speed of trading: positive buys, negative sells
            return gym.spaces.Box(low=np.float32([-self.max_speed]), high=np.float32([self.max_speed]))

    def _get_normalised_observation_space(self):
        # Linear normalisation of the gym.Box space so that the domain of the observation space is [-1,1].
        return gym.spaces.Box(
            low=-np.ones_like(self.observation_space.low, dtype=np.float32),
            high=np.ones_like(self.observation_space.high, dtype=np.float32),
        )

    def _get_normalised_action_space(self):
        # Linear normalisation of the gym.Box space so that the domain of the action space is [-1,1].
        return gym.spaces.Box(
            low=-np.ones_like(self.action_space.low, dtype=np.float32),
            high=np.ones_like(self.action_space.high, dtype=np.float32),
        )

    def _get_execution_action(self, action: np.ndarray) -> np.ndarray:
        return np.sign(action) * min(np.abs(action), np.abs(self.state[:, INVENTORY_INDEX]))

    def _get_start_time(self):
        if isinstance(self.start_time, (float, int)):
            random_start = self.start_time
        elif isinstance(self.start_time, Callable):
            random_start = self.start_time()
        else:
            raise NotImplementedError
        return self._quantise_time_to_step(random_start)

    def _quantise_time_to_step(self, time: float):
        assert (time >= 0.0) and (time < self.terminal_time), "Start time is not within (0, env.terminal_time)."
        return np.round(time / self.step_size) * self.step_size

    def _get_initial_inventories(self) -> np.ndarray:
        if isinstance(self.initial_inventory, tuple) and len(self.initial_inventory) == 2:
            return self.rng.integers(*self.initial_inventory, size=self.num_trajectories)
        elif isinstance(self.initial_inventory, int):
            return self.initial_inventory * np.ones((self.num_trajectories,))
        elif isinstance(self.initial_inventory, Callable):
            initial_inventory = self.initial_inventory()
            if self.action_type not in EXECUTION_ACTION_TYPES:
                initial_inventory = int(np.round(initial_inventory))
            return initial_inventory
        else:
            raise Exception("Initial inventory must be a tuple of length 2 or an int.")

    def _clip_inventory_and_cash(self):
        self.state[:, INVENTORY_INDEX] = self._clip(
            self.state[:, INVENTORY_INDEX], -self.max_inventory, self.max_inventory, cash_flag=False
        )
        self.state[:, CASH_INDEX] = self._clip(self.state[:, CASH_INDEX], -self.max_cash, self.max_cash, cash_flag=True)

    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped).any() and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped).any() and not cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)

    def _check_required_stochastic_processes(self) -> None:
        assert self.action_type in ACTION_TYPES, f"Action type '{self.action_type}' is not in {ACTION_TYPES}."
        if self.action_type == "touch":
            processes = ["arrival_model"]
        elif self.action_type in ["limit", "limit_and_market"]:
            processes = ["arrival_model", "fill_probability_model"]
        elif self.action_type == "speed":
            processes = ["price_impact_model"]
        else:
            raise NotImplementedError
        for process in processes:
            self._check_process_is_not_none(process)

    def _check_process_is_not_none(self, process: str):
        assert getattr(self, process) is not None, f"Action type is '{self.action_type}' but env.{process} is None."

    def _get_stochastic_processes(self):
        stochastic_processes = dict()
        for process_name in ["midprice_model", "arrival_model", "fill_probability_model", "price_impact_model"]:
            process: StochasticProcessModel = getattr(self, process_name)
            if process is not None:
                stochastic_processes[process_name] = process
        return OrderedDict(stochastic_processes)

    def _get_stochastic_process_indices(self):
        process_indices = dict()
        count = 3
        for process_name, process in self.stochastic_processes.items():
            dimension = int(process.initial_vector_state.shape[1])
            process_indices[process_name] = (count, count + dimension)
            count += dimension
        return OrderedDict(process_indices)

    def _get_empty_infos(self):
        return [{} for _ in range(self.num_trajectories)] if self.num_trajectories > 1 else {}

    def _get_fill_multiplier(self):
        ones = np.ones((self.num_trajectories, 1))
        return np.append(-ones, ones, axis=1)

    def _get_inventory_neutral_rewards(self, num_total_trajectories = 100_000):
        fixed_action = 1 / self.fill_probability_model.fill_exponent
        full_trajectory_env = deepcopy(self)
        full_trajectory_env.start_time = 0.0
        full_trajectory_env.num_trajectories = num_total_trajectories
        full_trajectory_env.normalise_rewards_ = False
        class FixedAgent(Agent):
            def get_action(self, obs:np.ndarray) -> np.ndarray:
                return np.ones((num_total_trajectories, 2)) * fixed_action
        fixed_agent = FixedAgent()
        _, _, rewards = generate_trajectory(full_trajectory_env, fixed_agent)
        mean_rewards = np.mean(rewards) * self.n_steps
        return mean_rewards

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        for i, process in enumerate(self.stochastic_processes.values()):
            process.seed(seed + i + 1)
