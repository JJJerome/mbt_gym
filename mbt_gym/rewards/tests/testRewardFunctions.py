from unittest import TestCase, main

import numpy as np

from mbt_gym.rewards.RewardFunctions import RunningInventoryPenalty, PnL
from mbt_gym.gym.TradingEnvironment import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX

STEP_SIZE = 0.1
TEST_CURRENT_STATE = np.array([[120, 2, 0.5, 100]])
TEST_ACTION = np.array([[1, 1]])
TEST_NEXT_STATE = np.array([[20, 3, 0.5 + STEP_SIZE, 100.05]])  # Buy order gets filled
TERMINAL_TIME = 1.0


class testPnL(TestCase):
    def test_calculate_per_step_reward(self):
        current_value = (
            TEST_CURRENT_STATE[:, CASH_INDEX]
            + TEST_CURRENT_STATE[:, INVENTORY_INDEX] * TEST_CURRENT_STATE[:, ASSET_PRICE_INDEX]
        )
        next_value = (
            TEST_NEXT_STATE[:, CASH_INDEX] + TEST_NEXT_STATE[:, INVENTORY_INDEX] * TEST_NEXT_STATE[:, ASSET_PRICE_INDEX]
        )
        expected = next_value - current_value
        actual = PnL().calculate(current_state=TEST_CURRENT_STATE, action=TEST_ACTION, next_state=TEST_NEXT_STATE)
        self.assertEqual(expected, actual, f"PnL calculation should give {expected}. Instead got {actual}!")


PER_STEP_INVENTORY_AVERSION = 0.01
TERMINAL_INVENTORY_AVERSION = 10


class testInventoryReward(TestCase):
    def test_calculate_per_step_reward(self):
        reward_function = RunningInventoryPenalty(PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION)
        pnl = PnL().calculate(current_state=TEST_CURRENT_STATE, action=TEST_ACTION, next_state=TEST_NEXT_STATE)
        inventory_penalty = PER_STEP_INVENTORY_AVERSION * STEP_SIZE * abs(TEST_NEXT_STATE[:, INVENTORY_INDEX]) ** 2
        expected = pnl - inventory_penalty
        actual = reward_function.calculate(TEST_CURRENT_STATE, TEST_ACTION, TEST_NEXT_STATE)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    main()
