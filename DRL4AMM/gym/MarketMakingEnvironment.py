from xmlrpc.client import boolean
import gym
import numpy as np

from gym.spaces import Box
from math import isclose

from DRL4AMM.gym.probability_models import (
    MidpriceModel,
    FillProbabilityModel,
    ArrivalModel,
    BrownianMotionMidpriceModel,
    PoissonArrivalModel,
    ExponentialFillFunction,
)
from DRL4AMM.rewards.RewardFunctions import RewardFunction, CJ_criterion

ACTION_SPACES = ["touch", "limit", "limit_and_market"]


class MarketMakingEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        terminal_time: float = 30.0,
        n_steps: int = 30 * 10,
        reward_function: RewardFunction = None,
        arrival_model: ArrivalModel = None,
        midprice_model: MidpriceModel = None,
        fill_probability_model: FillProbabilityModel = None,
        action_type: str = "limit",
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        max_inventory: int = 10_000,
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        market_order_penalty: float = None,
        half_spread: float = None,
        seed: int = None,
    ):
        super(MarketMakingEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or CJ_criterion(phi=2 * 10 ** (-4), alpha=0.0001)
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel()
        self.midprice_model: MidpriceModel = midprice_model or BrownianMotionMidpriceModel()
        self.fill_probability_model: FillProbabilityModel = fill_probability_model or ExponentialFillFunction()
        assert action_type in ["limit", "limit_and_market", "touch"]
        self.action_type = action_type
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.cash = initial_cash
        self.inventory = initial_inventory
        self.max_stock_price = max_stock_price or self.midprice_model.max_value
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self._get_max_depth()
        self.rng = np.random.default_rng(seed)
        self.dt = self.terminal_time / self.n_steps
        self.initial_state = self._get_initial_state()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.time = 0.0
        self.state = self._get_initial_state()
        self.market_order_penalty = market_order_penalty
        self.half_spread = half_spread

    def reset(self):
        self.reset_internal_state()
        self.arrival_model.reset()
        self.midprice_model.reset()
        self.fill_probability_model.reset()
        return self.state

    def step(self, actions: np.ndarray):
        current_state = self.state
        next_state = self._update_state(actions)
        done = isclose(next_state[2], self.terminal_time)  # due to floating point arithmetic
        reward = self.reward_function.calculate(current_state, actions, next_state, done)
        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    # actions = [bid_depth, ask_depth, MO_buy, MO_sell]
    # state[0]=cash, state[1]=inventory, state[2]=time, then remaining states depend on dimensionality of the arrival
    # process, the midprice process and the fill probability process.
    def _update_state(self, actions: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()  # [arrivebid, arriveask]
        depths = actions[0:2]  # [depthbid, depthask]
        fills = self.fill_probability_model.get_hypothetical_fills(depths)  # [fillbid, fillask]
        self.arrival_model.update(arrivals, fills, actions)  # TODO
        self.midprice_model.update(arrivals, fills, actions)  # TODO
        self.fill_probability_model.update(arrivals, fills, actions)  # TODO
        self._update_cash_and_inventory(arrivals, fills, actions)  # TODO
        self.state = np.array([self.cash, self.inventory, self.time], dtype=np.float32)
        self.state = np.append(self.state, self.arrival_model.current_state)
        self.state = np.append(self.state, self.midprice_model.current_state)
        self.state = np.append(self.state, self.fill_probability_model.current_state)
        return self.state

    def _update_cash_and_inventory(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray):
        if self.action_type == "limit_and_market":
            MO_buy = float(actions[2] > 0.5)
            MO_sell = float(actions[3] > 0.5)
            best_bid = self.midprice_model.current_state - self.market_order_penalty
            best_ask = self.midprice_model.current_state + self.market_order_penalty
            self.cash += (MO_sell) * (best_bid) - (MO_buy) * best_ask
            self.inventory += (MO_buy) - (MO_sell)
        self.inventory += np.sum(arrivals * fills * [1, -1])
        self.inventory = self._clip_and_warning(
            self.inventory, -self.max_inventory, self.max_inventory, cash_flag=False
        )
        if self.action_type == "touch":
            bidask = actions[0:2]  # [postedbid, postedask] for 'touch' action
            self.cash += np.sum(
                arrivals * fills * (self.midprice_model.current_state + bidask * self.half_spread) * [-1, 1]
            )
        else:
            depths = actions[0:2]  # [depthbid, depthask] for 'limit'-type actions
            self.cash += np.sum(arrivals * fills * (self.midprice_model.current_state + depths) * [-1, 1])
        self.cash = self._clip_and_warning(self.cash, -self.max_cash, self.max_cash, cash_flag=True)
        self.time += self.dt
        self.time = np.minimum(self.time, self.terminal_time)

    def _clip_and_warning(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped) and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped) and ~cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    def reset_internal_state(self):
        self.state = self.initial_state  # TODO: do we need self.state? There is repetition.
        self.cash = self.initial_cash
        self.inventory = self.initial_inventory
        self.time = 0.0

    def _get_initial_state(self) -> np.ndarray:
        state = np.array([self.initial_cash, self.initial_inventory, 0])
        state = np.append(state, self.arrival_model.initial_state)
        state = np.append(state, self.midprice_model.initial_state)
        state = np.append(state, self.fill_probability_model.initial_state)
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
            max_depth = self.fill_probability_model.max_depth
            return gym.spaces.Box(low=0.0, high=max_depth, shape=(2,))  # agent chooses spread on bid and ask
        if self.action_type == "limit_and_market":
            max_depth = self.fill_probability_model.max_depth
            return gym.spaces.Box(
                low=np.zeros(
                    4,
                ),
                high=np.array(max_depth, max_depth, 1, 1),
                shape=(2,),
            )

    @staticmethod
    def _get_max_depth():
        return 4.0  # TODO: improve

    @staticmethod
    def _clamp(probability):
        return max(min(probability, 1), 0)
