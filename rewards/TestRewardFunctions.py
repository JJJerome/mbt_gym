from unittest import TestCase, main

import numpy as np

from RL4MM.rewards.RewardFunctions import InventoryAdjustedPnL, PnL

STEP_SIZE = 0.1
TEST_CURRENT_STATE = np.array([100, 120, 2, 0.5])
TEST_ACTION = np.array([1, 1])
TEST_NEXT_STATE = np.array([100.05, 20, 3, 0.5 + STEP_SIZE])  # Buy order gets filled
TERMINAL_TIME = 1.0


class TestPnL(TestCase):
    def test_calculate_per_step_reward(self):
        current_value = 120 + 2 * 100
        next_value = 20 + 3 * 100.05
        expected = next_value - current_value
        actual = PnL().calculate(current_state=TEST_CURRENT_STATE, action=TEST_ACTION, next_state=TEST_NEXT_STATE)
        self.assertEqual(expected, actual, f"PnL calculation should give {expected}. Instead got {actual}!")


PER_STEP_INVENTORY_AVERSION = 0.01
TERMINAL_INVENTORY_AVERSION = 10


class TestInventoryReward(TestCase):
    def test_calculate_per_step_reward(self):
        reward_function = InventoryAdjustedPnL(
            PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION, step_size=STEP_SIZE
        )
        pnl = PnL().calculate(current_state=TEST_CURRENT_STATE, action=TEST_ACTION, next_state=TEST_NEXT_STATE)
        inventory_penalty = PER_STEP_INVENTORY_AVERSION * STEP_SIZE * abs(TEST_NEXT_STATE[2]) ** 2
        expected = pnl - inventory_penalty
        actual = reward_function.calculate(TEST_CURRENT_STATE, TEST_ACTION, TEST_NEXT_STATE)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    main()
