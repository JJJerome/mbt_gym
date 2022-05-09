import numpy as np

from RL4MM.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm

from RL4MM.gym.models import Action


class SBAgent(Agent):
    def __init__(self, model: BaseAlgorithm):
        self.model = model

    def get_action(self, state: np.ndarray) -> Action:
        return Action(*np.reshape(self.model.predict(state)[0], 2))

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)
