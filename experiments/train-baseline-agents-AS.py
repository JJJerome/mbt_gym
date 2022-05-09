import gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC, DDPG, PPO

import sys

sys.path.append("../../")

from RL4MM.gym.AvellanedaStoikovEnvironment import AvellanedaStoikovEnvironment  # noqa: E402
from RL4MM.gym.wrappers import ReduceStateSizeWrapper  # noqa: E402
from RL4MM.rewards.RewardFunctions import InventoryAdjustedPnL  # noqa: E402

save_dir = "../../../../experiment-results/"


alpha = 0.0
phi = 2 * 10 ** (-5)
n_steps = 1000  # have changed
terminal_time = 30
reward_function = InventoryAdjustedPnL(per_step_inventory_aversion=phi, terminal_inventory_aversion=alpha)
cartea_params = {
    "terminal_time": 30.0,
    "n_steps": n_steps,
    "reward_function": reward_function,
    "volatility": 0.01,
    "arrival_rate": 1,
    "fill_exponent": 100.0,
    "max_inventory": 100.0,
    "max_action": 0.2,  # Have changed
}
timestamps = np.linspace(0, 30, n_steps + 1)

env = ReduceStateSizeWrapper(AvellanedaStoikovEnvironment(**cartea_params))  # type: ignore  ##  for mypy

# It is necessary to register gym environment to parallelise it with SB
gym.envs.register(
    id="cartea-env-v0", entry_point="__main__:AvellanedaStoikovEnvironment", max_episode_steps=320, kwargs=cartea_params
)

algorithms = [PPO, SAC, DDPG]
tensorboard_logdir = save_dir + "tensorboard_logs/testing-stable-baselines-algos-deterministic"

normalised_vec_env = make_vec_env("cartea-env-v0", n_envs=1, wrapper_class=ReduceStateSizeWrapper)

# models = {}
# for algorithm in algorithms:
#     models[algorithm] = algorithm("MlpPolicy", normalised_vec_env, verbose=1, tensorboard_log=tensorboard_logdir)
#
# for algorithm in algorithms:
#     models[algorithm].learn(total_timesteps=1000000)
#     models[algorithm].save(save_dir + "saved_models/" + algorithm.__name__ + "reduced_env_deterministic")

model = SAC.load(save_dir + "saved_models/" + "SAC" + "reduced_env_deterministic")
model.env = normalised_vec_env
model.learn(total_timesteps=500000)
model.save(save_dir + "saved_models/" + "SAC" + "reduced_env_no_terminal_penalty")
