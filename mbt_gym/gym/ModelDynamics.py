import abc
import gym
from copy import copy
from typing import Optional
        
import numpy as np
from numpy.random import default_rng


from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, BID_INDEX, ASK_INDEX
from mbt_gym.stochastic_processes.arrival_models import ArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import FillProbabilityModel
from mbt_gym.stochastic_processes.midprice_models import MidpriceModel
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel


class ModelDynamics(metaclass=abc.ABCMeta):
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        price_impact_model : PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed_ = seed
        self.fill_multiplier = self._get_fill_multiplier()
        self.round_initial_inventory = False
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.stochastic_processes = self._get_stochastic_processes()
        self.stochastic_process_indices = self._get_stochastic_process_indices()
        self.state = None

    @abc.abstractmethod
    def update_state(self, action: np.ndarray):
        pass

    @abc.abstractmethod
    def get_action_space(self) -> gym.spaces.Space:
        pass

    @abc.abstractmethod
    def get_required_stochastic_processes(self):
        pass

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)  # TODO: remove?

    def _update_market_state(self, **kwargs):
        for process_name, process in self.stochastic_processes.items():
            process.update(**kwargs)
            lower_index = self.stochastic_process_indices[process_name][0]
            upper_index = self.stochastic_process_indices[process_name][1]
            self.state[:, lower_index:upper_index] = process.current_state

    def _check_processes_are_not_none(self, processes):
        for process in processes:
            self._check_process_is_not_none(process)

    def _check_process_is_not_none(self, process: str):
        assert getattr(self, process) is not None, f"This model dynamics cannot have env.{process} to be None."

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

    def _clip_inventory_and_cash(self):
        self.state[:, INVENTORY_INDEX] = self._clip(
            self.state[:, INVENTORY_INDEX], -self.max_inventory, self.max_inventory, cash_flag=False
        )
        self.state[:, CASH_INDEX] = self._clip(
            self.state[:, CASH_INDEX], -self.max_cash, self.max_cash, cash_flag=True
        )

    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped).any() and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped).any() and not cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped


class LimitOrderModelDynamics(ModelDynamics):
    """ModelDynamics for 'limit'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth : float = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_depth = max_depth or self._get_max_depth()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True
        
    def update_state(self, action: np.ndarray):
        arrivals, fills = self.get_arrivals_and_fills(action)
        if fills is not None:
            fills = self._remove_max_inventory_fills(fills)
        self._update_agent_state(arrivals, fills, action)
        self._update_market_state(arrivals, fills, action)
        return self.model_dynamics.state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
            self.fill_multiplier
            * arrivals
            * fills
            * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
            axis=1,
        )
        self._clip_inventory_and_cash()
        self.state[:, TIME_INDEX] += self.step_size

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(low=np.float32(0.0), high=np.float32(self.max_depth), shape=(2,))
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths)
        return arrivals, fills

    def _remove_max_inventory_fills(self, fills: np.ndarray) -> np.ndarray:
        fill_multiplier = np.concatenate(
            ((1 - self.is_at_max_inventory).reshape(-1, 1), (1 - self.is_at_min_inventory).reshape(-1, 1)), axis=1
        )
        return fill_multiplier * fills

    def _get_max_depth(self) -> Optional[float]:
        if self.fill_probability_model is not None:
            return self.fill_probability_model.max_depth
        else:
            return None

    def _get_fill_multiplier(self):
        ones = np.ones((self.num_trajectories, 1))
        return np.append(-ones, ones, axis=1)


class AtTheTouchModelDynamics(ModelDynamics):
    """ModelDynamics for 'touch'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        fixed_market_half_spread: float = 0.5,
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.round_initial_inventory = True
        self.fixed_market_half_spread = fixed_market_half_spread
        
    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self.fixed_market_half_spread * self.fill_multiplier),
                axis=1,
            )
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)

    def _post_at_touch(self, action: np.ndarray):
        return action[:, 0:2]

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiBinary(2) 
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        fills = self._post_at_touch(action)
        return arrivals, fills


class LimitAndMarketOrderModelDynamics(ModelDynamics):
    """ModelDynamics for 'limit_and_market'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth : float = None,
        fixed_market_half_spread : float = 0.5,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_depth = max_depth or self._get_max_depth()
        self.fixed_market_half_spread = fixed_market_half_spread
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True

    def _market_order_buy(self, action: np.ndarray):
        return action[:, 2 + BID_INDEX]
        
    def _market_order_sell(self, action: np.ndarray):
        return action[:, 2 + ASK_INDEX]

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        mo_buy = np.single(self._market_order_buy(action) > 0.5)
        mo_sell = np.single(self._market_order_sell(action) > 0.5)
        best_bid = (self.midprice - self.fixed_market_half_spread).reshape(-1,)
        best_ask = (self.midprice + self.fixed_market_half_spread).reshape(-1,)
        self.state[:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
        self.state[:, INVENTORY_INDEX] += mo_buy - mo_sell
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
                axis=1,
            )

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(
                low=np.zeros(4),
                high=np.array([self.max_depth, self.max_depth, 1, 1], dtype=np.float32),
            )
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals()
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths)
        return arrivals, fills

    def _limit_depths(self, action: np.ndarray):
        return action[:, 0:2]


class TradinghWithSpeedModelDynamics(ModelDynamics):
    """ModelDynamics for 'speed'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        price_impact_model : PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_speed : float = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        price_impact_model = price_impact_model,
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_speed = max_speed or self._get_max_speed()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = False

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        price_impact = self.price_impact_model.get_impact(action)
        execution_price = self.midprice + price_impact
        volume = action * self.midprice_model.step_size
        self.state[:, CASH_INDEX] -= np.squeeze(volume * execution_price)
        self.state[:, INVENTORY_INDEX] += np.squeeze(volume)

    def get_action_space(self) -> gym.spaces.Space:
        # agent chooses speed of trading: positive buys, negative sells
        return gym.spaces.Box(low=np.float32([-self.max_speed]), high=np.float32([self.max_speed]))
    
    def get_required_stochastic_processes(self):
        processes = ["price_impact_model"]
        return processes

    def _get_max_speed(self) -> float:
        if self.price_impact_model is not None:
            return self.price_impact_model.max_speed
        else:
            return None
