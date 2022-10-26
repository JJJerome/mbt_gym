import gym
import numpy as np
import torch

from mbt_gym.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None, include_log_probs: bool = False):
    if seed is not None:
        env.seed(seed)
    obs_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    observations = np.zeros((env.num_trajectories, obs_space_dim, env.n_steps + 1))
    actions = np.zeros((env.num_trajectories, action_space_dim, env.n_steps))
    rewards = np.zeros((env.num_trajectories, 1, env.n_steps))
    if include_log_probs:
        log_probs = torch.zeros((env.num_trajectories, env.action_space.shape[0], env.n_steps))
    obs = env.reset()
    observations[:, :, 0] = obs
    count = 0
    while True:
        if include_log_probs:
            action, log_prob = agent.get_action(obs, include_log_probs=True)
        else:
            action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        actions[:, :, count] = action
        observations[:, :, count + 1] = obs
        rewards[:, :, count] = reward.reshape(-1, 1)
        if include_log_probs:
            log_probs[:, :, count] = log_prob
        if (env.num_trajectories > 1 and done[0]) or (env.num_trajectories == 1 and done):
            break
        count += 1
    if include_log_probs:
        return observations, actions, rewards, log_probs
    else:
        return observations, actions, rewards
