import gym
import numpy as np

from DRL4AMM.agents.Agent import Agent


def generate_trajectory(env: gym.Env, agent: Agent, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    observations = np.zeros((env.num_trajectories, env.observation_space.shape[0], env.n_steps + 1))
    actions = np.zeros((env.num_trajectories, env.action_space.shape[0], env.n_steps))
    rewards = np.zeros((env.num_trajectories, 1, env.n_steps))
    obs = env.reset()
    observations[:, : , 0] = obs
    count = 0

    while True:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        actions[:,:,count] = action
        observations[:,:,count+1] = obs
        rewards[:,:, count] = reward.reshape(-1,1)
        if (env.num_trajectories > 1 and done[0]) or (env.num_trajectories == 1 and done):
            break
        count +=1
    return np.array(observations), np.array(actions), np.array(rewards)
