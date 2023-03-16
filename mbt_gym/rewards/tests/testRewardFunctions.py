from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from mbt_gym.rewards.RewardFunctions import RunningInventoryPenalty, PnL, CjMmCriterion
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, TIME_INDEX, ASSET_PRICE_INDEX

STEP_SIZE = 0.2
TEST_CURRENT_STATE = np.array([[120, 2, 0.5, 100]])
TEST_ACTION = np.array([[1, 1]])
TEST_NEXT_STATE = np.array([[20, 3, 0.5 + STEP_SIZE, 100.05]])  # Buy order gets filled
TERMINAL_TIME = 1.0

# CASH, INVENTORY, TIME, ASSET_PRICE
MOCK_OBSERVATIONS = [
    np.array([[100.0, 0, 0.0, 100]]),
    np.array([[0.5, 1, STEP_SIZE, 101]]),
    np.array([[102.0, 0, 2 * STEP_SIZE, 102]]),
    np.array([[103.0, 0, 3 * STEP_SIZE, 103]]),
    np.array([[206.5, -1, 4 * STEP_SIZE, 104]]),
    np.array([[103.0, 0, 5 * STEP_SIZE, 103]]),
]
MOCK_ACTIONS = [
    np.array([[0.5, 0.5]]),
    np.array([[0.5, 1]]),
    np.array([[0.5, 0.5]]),
    np.array([[1, 0.5]]),
    np.array([[0.5, 0.5]]),
]


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
TERMINAL_INVENTORY_AVERSION = 1


class testInventoryReward(TestCase):
    def test_calculate_per_step_reward(self):
        reward_function = RunningInventoryPenalty(PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION)
        pnl = PnL().calculate(current_state=TEST_CURRENT_STATE, action=TEST_ACTION, next_state=TEST_NEXT_STATE)
        inventory_penalty = PER_STEP_INVENTORY_AVERSION * STEP_SIZE * abs(TEST_NEXT_STATE[:, INVENTORY_INDEX]) ** 2
        expected = pnl - inventory_penalty
        actual = reward_function.calculate(TEST_CURRENT_STATE, TEST_ACTION, TEST_NEXT_STATE)
        self.assertAlmostEqual(expected.item(), actual.item(), places=5)


class testCjMmCriterion(TestCase):
    cj_mm_criterion = CjMmCriterion(
        per_step_inventory_aversion=PER_STEP_INVENTORY_AVERSION,
        terminal_inventory_aversion=TERMINAL_INVENTORY_AVERSION,
        terminal_time=TERMINAL_TIME,
    )

    def test_agreement_with_non_decontructed_version(self):
        target_reward_function = RunningInventoryPenalty(PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION)
        cj_mm_rewards = []
        target_rewards = []
        self.cj_mm_criterion.reset(MOCK_OBSERVATIONS[0])
        for i in range(len(MOCK_ACTIONS)):
            is_terminal_step = MOCK_OBSERVATIONS[i + 1][:, TIME_INDEX] == 1
            cj_mm_rewards.append(
                self.cj_mm_criterion.calculate(
                    MOCK_OBSERVATIONS[i], MOCK_ACTIONS[i], MOCK_OBSERVATIONS[i + 1], is_terminal_step
                )
            )
            target_rewards.append(
                target_reward_function.calculate(
                    MOCK_OBSERVATIONS[i], MOCK_ACTIONS[i], MOCK_OBSERVATIONS[i + 1], is_terminal_step
                )
            )
        self.assertAlmostEqual(float(sum(cj_mm_rewards)), float(sum(target_rewards)), places=5)

    def test_agreement_with_non_decontructed_version_nonzero_initial_inventory(self):
        target_reward_function = RunningInventoryPenalty(PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION)
        cj_mm_rewards = []
        target_rewards = []
        mock_observations = deepcopy(MOCK_OBSERVATIONS)
        mock_observations[0][:, INVENTORY_INDEX] = 2
        mock_observations[0][:, CASH_INDEX] = -100
        mock_observations[-1] = deepcopy(mock_observations[-2])
        mock_observations[-1][:, TIME_INDEX] = 1.0
        self.cj_mm_criterion.reset(mock_observations[0])
        for i in range(len(MOCK_ACTIONS)):
            is_terminal_step = mock_observations[i + 1][:, TIME_INDEX] == 1
            cj_mm_rewards.append(
                self.cj_mm_criterion.calculate(
                    mock_observations[i], MOCK_ACTIONS[i], mock_observations[i + 1], is_terminal_step
                )
            )
            target_rewards.append(
                target_reward_function.calculate(
                    mock_observations[i], MOCK_ACTIONS[i], mock_observations[i + 1], is_terminal_step
                )
            )
        self.assertAlmostEqual(float(sum(cj_mm_rewards)), float(sum(target_rewards)), places=5)

    def test_agreement_with_non_decontructed_version_partial_trajectory(self):
        target_reward_function = RunningInventoryPenalty(PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION)
        cj_mm_rewards = []
        target_rewards = []
        START_STEP = 2
        self.cj_mm_criterion.reset(MOCK_OBSERVATIONS[START_STEP])
        for i in range(len(MOCK_ACTIONS[START_STEP:])):
            is_terminal_step = MOCK_OBSERVATIONS[START_STEP + i + 1][:, TIME_INDEX] == 1
            cj_mm_rewards.append(
                self.cj_mm_criterion.calculate(
                    MOCK_OBSERVATIONS[START_STEP + i],
                    MOCK_ACTIONS[START_STEP + i],
                    MOCK_OBSERVATIONS[START_STEP + i + 1],
                    is_terminal_step,
                )
            )
            target_rewards.append(
                target_reward_function.calculate(
                    MOCK_OBSERVATIONS[START_STEP + i],
                    MOCK_ACTIONS[START_STEP + i],
                    MOCK_OBSERVATIONS[START_STEP + i + 1],
                    is_terminal_step,
                )
            )
        self.assertAlmostEqual(float(sum(cj_mm_rewards)), float(sum(target_rewards)), places=5)


if __name__ == "__main__":
    main()
