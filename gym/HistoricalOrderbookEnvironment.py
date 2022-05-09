from datetime import datetime, timedelta
from typing import List

import gym
import numpy as np

from gym.spaces import Discrete, Tuple
from gym.utils import seeding

from RL4MM.features.Feature import Feature
from RL4MM.simulation.StaleOrderbookSimulator import (
    StaleHistoricalOrderbookSimulator,
    OrderbookSimulator,
    ResultsDict,
    StaleOrderbookMessage,
)

NASDAQ_START_DELTA = timedelta(hours=9, minutes=30)
NASDAQ_END_DELTA = timedelta(hours=16, minutes=0)
ORDER_SIZE = 1


class HistoricalOrderbookEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        simulator: OrderbookSimulator,
        min_date: datetime,
        max_date: datetime,
        step_size: timedelta,
        features: List[Feature],
        num_steps: int = 10,
        initial_portfolio: dict = None,
    ):
        super(HistoricalOrderbookEnvironment, self).__init__()
        self.simulator = simulator
        self.min_date = min_date
        self.max_date = max_date
        self.step_size = step_size
        self.features = features
        self.num_steps = num_steps
        self.initial_portfolio = initial_portfolio or {"cash": 1000, "stock": 0}
        self.portfolio = self.initial_portfolio
        self.now = min_date
        self.underlying_state: ResultsDict = None
        self.current_step = 0
        # Actions can be (0,0), (0,1), (1,0), (1,1). Here, we are only posting orders of size one at the touch.
        self.action_space = Tuple((Discrete(2), Discrete(2)))
        # Observation spaces are determined by the features used
        self.observation_space = Tuple(Discrete(len(feature.feature_space)) for feature in self.features)
        self.reset()

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _take_action(self, action):
        pass

    def step(self, action):
        done, info = 0, 0
        self.current_step += 1
        our_messages = self._convert_action_to_message(action=action)
        results_dict = self.simulator.simulate_step(
            start_date=self.now,
            end_date=self.now + self.step_size,
            messages_to_fill=our_messages,
            start_book=self.underlying_state,
        )
        filled_messages = results_dict["filled_messages"]
        self._update_portfolio(filled_messages)
        observation = self._get_features_from_underlying(results_dict)
        reward = 0
        if self.current_step == self.num_steps:
            reward = 100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
            done, info = 1, 1
        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        self.portfolio = self.initial_portfolio
        random_offset_days = np.random.randint(
            int((self.max_date.date() - self.min_date.date()) / timedelta(days=1)) + 1
        )
        max_step = int((NASDAQ_END_DELTA - NASDAQ_START_DELTA) / self.step_size) - self.num_steps
        random_offset_steps = np.random.randint(low=0, high=max_step)
        self.now = self.min_date + timedelta(days=random_offset_days) + random_offset_steps * self.step_size
        self.underlying_state = self.simulator.simulate_step(start_date=self.now, end_date=self.now)
        return

    def _get_features_from_underlying(self, results: ResultsDict):
        return (feature.calculate(results) for feature in self.features)

    def _convert_action_to_message(self, action):
        if not isinstance(self.simulator, StaleHistoricalOrderbookSimulator):
            raise NotImplementedError
        ticker = self.simulator.ticker
        our_messages = list()
        for i, side in enumerate(["bid", "ask"]):
            if action[i] == 1:
                our_messages.append(
                    StaleOrderbookMessage(
                        _id="-1",
                        timestamp=self.now,
                        message_type="submission",
                        ticker=ticker,
                        size=ORDER_SIZE,
                        price=self.underlying_state[side + "_price_0"],
                        side=side,
                    )
                )
        return our_messages

    def _update_portfolio(self, filled_messages):
        for message in filled_messages:
            if message.side == "ask":
                self.portfolio["stock"] -= message.volume
                self.portfolio["cash"] += message.volume * message.price
            if message.side == "bid":
                self.portfolio["stock"] += message.volume
                self.portfolio["cash"] -= message.volume * message.price
