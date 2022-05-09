import gym
import numpy as np

from RL4MM.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    observations = []
    actions = []
    rewards = []
    obs = env.reset()
    observations.append(obs)
    while True:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        if done:
            break
    return observations, actions, rewards
