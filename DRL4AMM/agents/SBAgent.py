import numpy as np

from DRL4AMM.agents.Agent import Agent

from stable_baselines3.common.base_class import BaseAlgorithm

from DRL4AMM.gym.models import Action


class SBAgent(Agent):
    def __init__(self, model: BaseAlgorithm, reduced_training: bool = False):
        self.model = model
        self.reduced_training = reduced_training

    def get_action(self, state: np.ndarray) -> Action:
        if self.reduced_training:
            state = state[-2:]
        return Action(*np.reshape(self.model.predict(state)[0], 2))

    def train(self, total_timesteps: int = 100000):
        self.model.learn(total_timesteps=total_timesteps)
