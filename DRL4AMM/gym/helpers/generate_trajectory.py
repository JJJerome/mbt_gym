import gym
import numpy as np

from DRL4AMM.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None, vectorised: bool = False):
    if seed is not None:
        np.random.seed(seed)
    observations = np.zeros((env.num_trajectories, env.observation_space.shape[0], env.n_steps))
    actions = np.zeros((env.num_trajectories, env.action_space.shape[0], env.n_steps))
    rewards = np.zeros((env.num_trajectories, 1, env.n_steps))
    obs = env.reset()
    count = 0
    while True:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        if (vectorised and done[0]) or (not vectorised and done):
            break
    return np.array(observations), np.array(actions), np.array(rewards)
