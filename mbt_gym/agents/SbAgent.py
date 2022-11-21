import numpy as np

from mbt_gym.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm


class SbAgent(Agent):
    def __init__(self, model: BaseAlgorithm, reduced_training_indices: list = None):
        self.model = model
        if reduced_training_indices is not None:
            self.reduced_training = True
            self.reduced_training_indices = reduced_training_indices
        else:
            self.reduced_training = False

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if self.reduced_training:
            state = state[:, self.reduced_training_indices]
        return np.reshape(self.model.predict(state, deterministic=True)[0], -1)

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)
