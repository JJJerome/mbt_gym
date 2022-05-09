import gym

import numpy as np

from math import sqrt


class ReduceStateSizeWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(ReduceStateSizeWrapper, self).__init__(env)
        assert type(env.observation_space) == gym.spaces.box.Box
        self.observation_space = gym.spaces.box.Box(
            low=env.observation_space.low[2:],
            high=env.observation_space.high[2:],
            dtype=np.float64,
        )

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs[2:]

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        return obs[2:], reward, done, info


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


class LearnTerminalStrategy(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, num_final_steps: int = 5):
        # Call the parent constructor, so we can access self.env later
        super(LearnTerminalStrategy, self).__init__(env)
        self.num_final_steps = num_final_steps
        env.observation_space.low[-1] = env.terminal_time - num_final_steps * env.dt

    def reset(self):
        """
        Reset the environment
        """
        self.env.reset()
        start_time = self.env.terminal_time - self.num_final_steps * self.env.dt
        start_asset_price = (
            self.env.drift * self.env.terminal_time + self.env.volatility * sqrt(start_time) * self.env.rng.normal()
        )
        start_cash = self.env.rng.normal(self.env.initial_cash, self.env.initial_cash / 2)
        start_inventory = self.env.rng.integers(self.env.observation_space.low[2], self.env.observation_space.high[2])
        self.env.state = np.array([start_asset_price, start_cash, start_inventory, start_time])
        return self.env.state

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        return self.env.step(action)


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
