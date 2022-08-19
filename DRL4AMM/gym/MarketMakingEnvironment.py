from copy import copy
import gym
import numpy as np

from gym.envs.registration import EnvSpec
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
from DRL4AMM.gym.tracking.InfoCalculator import InfoCalculator, ActionInfoCalculator
from DRL4AMM.rewards.RewardFunctions import RewardFunction, CjCriterion, PnL

ACTION_SPACES = ["touch", "limit", "limit_and_market"]


class MarketMakingEnvironment(gym.Env):
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
        initial_inventory: int = 0,
        max_inventory: int = 10_000,
        max_cash: float = None,
        max_stock_price: float = None,
        max_depth: float = None,
        market_order_penalty: float = None,
        info_calculator: InfoCalculator = None,
        seed: int = None,
    ):
        super(MarketMakingEnvironment, self).__init__()
        self.terminal_time = terminal_time
        self.n_steps = n_steps
        self.reward_function = reward_function or PnL()  # CjCriterion(phi=2 * 10 ** (-4), alpha=0.0001)
        self.midprice_model: MidpriceModel = midprice_model or BrownianMotionMidpriceModel(
            step_size=self.terminal_time / self.n_steps
        )
        self.arrival_model: ArrivalModel = arrival_model or PoissonArrivalModel(
            step_size=self.terminal_time / self.n_steps
        )
        self.fill_probability_model: FillProbabilityModel = fill_probability_model or ExponentialFillFunction(
            step_size=self.terminal_time / self.n_steps
        )
        self.action_type = action_type
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.cash = initial_cash
        self.inventory = initial_inventory
        self.max_stock_price = max_stock_price or self.midprice_model.max_value[0, 0]
        self.max_cash = max_cash or self._get_max_cash()
        self.max_depth = max_depth or self.fill_probability_model.max_depth
        self.rng = np.random.default_rng(seed)
        self.dt = self.terminal_time / self.n_steps
        self.initial_state = self._get_initial_state()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.time = 0.0
        self.book_half_spread = market_order_penalty
        self.info_calculator = info_calculator or ActionInfoCalculator()
        self._check_params()

    def reset(self):
        self._reset_agent_state()
        self.midprice_model.reset()
        self.arrival_model.reset()
        self.fill_probability_model.reset()
        return self.state

    def step(self, action: np.ndarray):
        current_state = copy(self.state)
        next_state = self._update_state(action)
        done = isclose(self.time, self.terminal_time)  # due to floating point arithmetic
        reward = self.reward_function.calculate(current_state, action, next_state, done)
        info = {} if self.info_calculator is None else self.info_calculator.calculate(next_state, action, reward)
        return self.state, reward, done, info

    def render(self, mode="human"):
        pass

    # TODO: add the spec attribute externally by registering the env with a max_episode_steps
    # @property
    # def spec(self):
    #     return EnvSpec(id="MarketMakingEnv-v0", max_episode_steps=self.n_steps)

    def _get_max_cash(self) -> float:
        return self.max_inventory * self.max_stock_price

    # action = [bid_depth, ask_depth, MO_buy, MO_sell]
    # state[0]=cash, state[1]=inventory, state[2]=time, state[3] = asset_price, and then remaining states depend on
    # the dimensionality of the arrival process, the midprice process and the fill probability process.
    def _update_state(self, action: np.ndarray) -> np.ndarray:
        arrivals = self.arrival_model.get_arrivals()
        if self.action_type in ["limit", "limit_and_market"]:
            depths = np.array([[self.limit_buy_depth(action), self.limit_sell_depth(action)]])
            fills = self.fill_probability_model.get_hypothetical_fills(depths)
        else:
            fills = np.array([self.post_buy_at_touch(action), self.post_sell_at_touch(action)])
        self.arrival_model.update(arrivals, fills, action)  # TODO
        self.midprice_model.update(arrivals, fills, action)  # TODO
        self.fill_probability_model.update(arrivals, fills, action)  # TODO
        self._update_agent_state(arrivals, fills, action)  # TODO
        return self.state

    def _update_agent_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        fill_multiplier = np.array([-1, 1])
        if self.action_type == "limit_and_market":
            mo_buy = float(self.market_order_buy(action) > 0.5)
            mo_sell = float(self.market_order_sell(action) > 0.5)
            best_bid = self.midprice_model.current_state - self.book_half_spread
            best_ask = self.midprice_model.current_state + self.book_half_spread
            self.cash += mo_sell * best_bid - mo_buy * best_ask
            self.inventory += mo_buy - mo_sell
        self.inventory += np.sum(arrivals * fills * -fill_multiplier)
        if self.action_type == "touch":
            self.cash += np.sum(
                fill_multiplier * arrivals * fills * (self.midprice + self.book_half_spread * fill_multiplier)
            )
        else:
            depths = np.array([self.limit_buy_depth(action), self.limit_sell_depth(action)])
            self.cash += np.sum(fill_multiplier * arrivals * fills * (self.midprice + depths * fill_multiplier))
        self._clip_inventory_and_cash()
        self.time += self.dt
        self.time = np.minimum(self.time, self.terminal_time)

    @property
    def state(self):
        state = np.array([self.cash, self.inventory, self.time])
        state = np.append(state, self.midprice_model.current_state)
        state = np.append(state, self.arrival_model.current_state)
        state = np.append(state, self.fill_probability_model.current_state)
        return state.reshape(1, -1)

    @property
    def midprice(self):
        return self.midprice_model.current_state[0]

    def _clip_inventory_and_cash(self):
        self.inventory = self._clip(self.inventory, -self.max_inventory, self.max_inventory, cash_flag=False)
        self.cash = self._clip(self.cash, -self.max_cash, self.max_cash, cash_flag=True)

    def _clip(self, not_clipped: float, min: float, max: float, cash_flag: bool) -> float:
        clipped = np.clip(not_clipped, min, max)
        if (not_clipped != clipped) and cash_flag:
            print(f"Clipping agent's cash from {not_clipped} to {clipped}.")
        if (not_clipped != clipped) and ~cash_flag:
            print(f"Clipping agent's inventory from {not_clipped} to {clipped}.")
        return clipped

    def limit_buy_depth(self, action: np.ndarray):
        if self.action_type in ["limit", "limit_and_market"]:
            return action[0, 0]
        else:
            raise Exception('Bid depth only exists for action_type in ["limit", "limit_and_market"].')

    def limit_sell_depth(self, action: np.ndarray):
        if self.action_type in ["limit", "limit_and_market"]:
            return action[0, 1]
        else:
            raise Exception('Ask depth only exists for action_type in ["limit", "limit_and_market"].')

    def market_order_buy(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[2]
        else:
            raise Exception('Market order buy action only exists for action_type == "limit_and_market".')

    def market_order_sell(self, action: np.ndarray):
        if self.action_type == "limit_and_market":
            return action[3]
        else:
            raise Exception('Market order sell action only exists for action_type == "limit_and_market".')

    def post_buy_at_touch(self, action: np.ndarray):
        if self.action_type == "touch":
            return action[0]
        else:
            raise Exception('Post buy at touch action only exists for action_type == "touch".')

    def post_sell_at_touch(self, action: np.ndarray):
        if self.action_type == "touch":
            return action[1]
        else:
            raise Exception('Post buy at touch action only exists for action_type == "touch".')

    def _reset_agent_state(self):
        self.cash = self.initial_cash
        self.inventory = self.initial_inventory
        self.time = 0.0

    def _get_initial_state(self) -> np.ndarray:
        state = np.array([self.initial_cash, self.initial_inventory, 0])
        state = np.append(state, self.midprice_model.initial_vector_state)
        state = np.append(state, self.arrival_model.initial_vector_state)
        state = np.append(state, self.fill_probability_model.initial_vector_state)
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

    def _check_params(self):
        assert self.action_type in ["limit", "limit_and_market", "touch"]
        for stochastic_process in [self.midprice_model, self.arrival_model, self.fill_probability_model]:
            assert np.isclose(stochastic_process.step_size, self.terminal_time / self.n_steps, 2), (
                f"{type(self.midprice_model).__name__}.step_size = {stochastic_process.step_size}, "
                + f" but env.step_size = {self.terminal_time/self.n_steps}"
            )
