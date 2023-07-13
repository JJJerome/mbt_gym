from collections import OrderedDict
from copy import copy, deepcopy
from typing import Union, Tuple, Callable

import gym
import numpy as np

from gym.spaces import Box

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.ModelDynamics import ModelDynamics, LimitOrderModelDynamics
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel
from mbt_gym.stochastic_processes.arrival_models import ArrivalModel, PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel, ExponentialFillFunction
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel, BrownianMotionMidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel
from mbt_gym.gym.info_calculators import InfoCalculator
from mbt_gym.rewards.RewardFunctions import RewardFunction, PnL

from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX


class TradingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 1.0,
        n_steps: int = 20 * 10,
        reward_function: RewardFunction = None,
        model_dynamics: ModelDynamics = None,
        initial_cash: float = 0.0,
        initial_inventory: Union[int, Tuple[float, float]] = 0,  # Either a deterministic initial inventory, or a tuple
        max_inventory: int = 10_000,  # representing the mean and variance of it.
        max_cash: float = None,
        max_stock_price: float = None,
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
        self._step_size = self.terminal_time / self.n_steps
        self.reward_function = reward_function or PnL()
        self.model_dynamics = model_dynamics or LimitOrderModelDynamics(
            midprice_model=BrownianMotionMidpriceModel(
                step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ),
            arrival_model=PoissonArrivalModel(
                intensity=np.array([100, 100]), step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ),
            fill_probability_model=ExponentialFillFunction(
                step_size=self._step_size, num_trajectories=num_trajectories, seed=seed
            ),
            num_trajectories=num_trajectories,
            seed=seed,
        )
        self.stochastic_processes = self._get_stochastic_processes()
        self.stochastic_process_indices = self._get_stochastic_process_indices()
        self.num_trajectories = num_trajectories
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        if seed:
            self.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.start_time = start_time
        self.model_dynamics.state = self.initial_state
        self.max_stock_price = max_stock_price or self.model_dynamics.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.info_calculator = info_calculator
        self._empty_infos = self._get_empty_infos()
        self.observation_space = self._get_observation_space()
        self.action_space = self.model_dynamics.get_action_space()
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
            assert isinstance(self.model_dynamics.arrival_model, PoissonArrivalModel) and isinstance(
                self.model_dynamics.fill_probability_model, ExponentialFillFunction
            ), "Arrival model must be Poisson and fill probability model must be exponential to scale rewards"
            self.reward_scaling = 1 / self._get_inventory_neutral_rewards()

    def reset(self):
        for process in self.stochastic_processes.values():
            process.reset()
        self.model_dynamics.state = self.initial_state
        self.reward_function.reset(self.model_dynamics.state.copy())
        return self.normalise_observation(self.model_dynamics.state.copy())

    def step(self, action: np.ndarray):
        action = self.normalise_action(action, inverse=True)
        current_state = self.model_dynamics.state.copy()
        next_state = self._update_state(action)
        dones = self._get_dones()
        rewards = self.reward_function.calculate(current_state, action, next_state, dones[0])
        infos = self._calculate_infos(current_state, action, rewards)
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
    def state(self):
        return self.model_dynamics.state

    @property
    def is_at_max_inventory(self):
        return self.state[:, INVENTORY_INDEX] >= self.max_inventory

    @property
    def is_at_min_inventory(self):
        return self.state[:, INVENTORY_INDEX] <= -self.max_inventory

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
    def num_trajectories(self, num_trajectories: int):
        self._num_trajectories = num_trajectories
        for process_name, process in self.stochastic_processes.items():
            if process.num_trajectories != num_trajectories:
                process.num_trajectories = num_trajectories
        self._empty_infos = self._get_empty_infos()
        self.model_dynamics.fill_multiplier = self.model_dynamics._get_fill_multiplier()

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

    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        arrivals, fills = self.model_dynamics.get_arrivals_and_fills(action)
        if fills is not None:
            fills = self._remove_max_inventory_fills(fills)
        self._update_agent_state(arrivals, fills, action)
        self._update_market_state(arrivals, fills, action)
        return self.model_dynamics.state

    def _update_market_state(self, arrivals, fills, action):
        for process_name, process in self.stochastic_processes.items():
            process.update(arrivals, fills, action, self.model_dynamics.state)
            lower_index = self.stochastic_process_indices[process_name][0]
            upper_index = self.stochastic_process_indices[process_name][1]
            self.model_dynamics.state[:, lower_index:upper_index] = process.current_state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.model_dynamics.update_state(arrivals, fills, action)
        self._clip_inventory_and_cash()
        self.model_dynamics.state[:, TIME_INDEX] += self.step_size

    def _get_dones(self):
        done = self.model_dynamics.state[0, TIME_INDEX] >= self.terminal_time - self.step_size / 2
        return np.full((self.num_trajectories,), done, dtype=bool)

    def _calculate_infos(self, current_state, action, rewards):
        return (
            self.info_calculator.calculate(current_state, action, rewards)
            if self.info_calculator is not None
            else self._empty_infos
        )

    def _get_max_cash(self) -> float:
        return self.n_steps * self.max_stock_price  # TODO: make this a tighter bound

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
            if self.model_dynamics.round_initial_inventory:
                initial_inventory = int(np.round(initial_inventory))
            return initial_inventory
        else:
            raise Exception("Initial inventory must be a tuple of length 2 or an int.")

    def _clip_inventory_and_cash(self):
        self.model_dynamics.state[:, INVENTORY_INDEX] = self._clip(
            self.model_dynamics.state[:, INVENTORY_INDEX], -self.max_inventory, self.max_inventory, cash_flag=False
        )
        self.model_dynamics.state[:, CASH_INDEX] = self._clip(
            self.model_dynamics.state[:, CASH_INDEX], -self.max_cash, self.max_cash, cash_flag=True
        )

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

    def _get_stochastic_processes(self):
        stochastic_processes = dict()
        for process_name in ["midprice_model", "arrival_model", "fill_probability_model", "price_impact_model"]:
            process: StochasticProcessModel = getattr(self.model_dynamics, process_name)
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

    def _remove_max_inventory_fills(self, fills: np.ndarray) -> np.ndarray:
        fill_multiplier = np.concatenate(
            ((1 - self.is_at_max_inventory).reshape(-1, 1), (1 - self.is_at_min_inventory).reshape(-1, 1)), axis=1
        )
        return fill_multiplier * fills

    def _get_inventory_neutral_rewards(self, num_total_trajectories=100_000):
        fixed_action = 1 / self.model_dynamics.fill_probability_model.fill_exponent
        full_trajectory_env = deepcopy(self)
        full_trajectory_env.start_time = 0.0
        full_trajectory_env.num_trajectories = num_total_trajectories
        full_trajectory_env.normalise_rewards_ = False

        class FixedAgent(Agent):
            def get_action(self, obs: np.ndarray) -> np.ndarray:
                return np.ones((num_total_trajectories, 2)) * fixed_action

        fixed_agent = FixedAgent()
        _, _, rewards = generate_trajectory(full_trajectory_env, fixed_agent)
        mean_rewards = np.mean(rewards) * self.n_steps
        return mean_rewards

    def seed(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        for i, process in enumerate(self.stochastic_processes.values()):
            process.seed(seed + i + 1)
