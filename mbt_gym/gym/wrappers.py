import gym

import numpy as np

from mbt_gym.gym.index_names import INVENTORY_INDEX, TIME_INDEX

from math import sqrt


class ReduceStateSizeWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, list_of_state_indices: list = [INVENTORY_INDEX, TIME_INDEX]):
        # Call the parent constructor, so we can access self.env later
        super(ReduceStateSizeWrapper, self).__init__(env)
        assert type(env.observation_space) == gym.spaces.box.Box
        self.observation_space = gym.spaces.box.Box(
            low=env.observation_space.low[list_of_state_indices],
            high=env.observation_space.high[list_of_state_indices],
            dtype=np.float64,
        )
        self.list_of_state_indices = list_of_state_indices

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs[:, self.list_of_state_indices]

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        return obs[:, self.list_of_state_indices], reward, done, info

    @property
    def spec(self):
        return self.env.spec


class NormaliseASObservation(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(NormaliseASObservation, self).__init__(env)
        self.normalisation_factor = 2 / (env.observation_space.high - env.observation_space.low)
        self.normalisation_offset = (env.observation_space.high + env.observation_space.low) / 2
        assert type(env.observation_space) == gym.spaces.box.Box
        self.observation_space = gym.spaces.box.Box(
            low=-np.ones(env.observation_space.shape),
            high=np.ones(env.observation_space.shape),
            dtype=np.float64,
        )

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return (obs - self.normalisation_offset) * self.normalisation_factor

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        return obs / self.normalisation_factor, reward, done, info


class RemoveTerminalRewards(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, num_final_steps: int = 5):
        # Call the parent constructor, so we can access self.env later
        super(RemoveTerminalRewards, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        state, reward, done, _ = self.env.step(action)
        if done:
            reward *= (
                self.env.reward_function.per_step_inventory_aversion
                / self.env.reward_function.terminal_inventory_aversion
            )
        return state, reward, done, {}
