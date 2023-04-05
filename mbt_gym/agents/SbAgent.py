import numpy as np

from mbt_gym.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm


class SbAgent(Agent):
    def __init__(self, model: BaseAlgorithm, reduced_training_indices: list = None, num_trajectories: int = None):
        self.model = model
        self.num_trajectories = num_trajectories or self.model.env.num_trajectories
        self.num_actions = self.model.action_space.shape[0]
        if reduced_training_indices is not None:
            self.reduced_training = True
            self.reduced_training_indices = reduced_training_indices
        else:
            self.reduced_training = False

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if self.reduced_training:
            state = state[:, self.reduced_training_indices]
        # return self.model.predict(state, deterministic=True)[0].reshape(self.num_trajectories, self.num_actions)
        return self.model.predict(state, deterministic=True)[0].reshape(state.shape[0], self.num_actions)

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)
