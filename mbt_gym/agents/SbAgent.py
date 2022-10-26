import numpy as np

from mbt_gym.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm


class SbAgent(Agent):
    def __init__(self, model: BaseAlgorithm, reduced_training: bool = False):
        self.model = model
        self.reduced_training = reduced_training

    def get_action(self, state: np.ndarray) -> np.ndarray:
        if self.reduced_training:
            state = state[-2:]
        return np.reshape(self.model.predict(state, deterministic=True)[0], -1)

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)
