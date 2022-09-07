import os

import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


import sys

sys.path.append("../")

from mbt_gym.gym.MMAtTouchEnvironment import MMAtTouchEnvironment
from mbt_gym.gym.wrappers import *
from mbt_gym.gym.helpers.plotting import *
from mbt_gym.rewards.RewardFunctions import CjCriterion

reward_function = CjCriterion(phi=0.01, alpha=10 * 0.01)
env_params = {"reward_function": reward_function, "max_inventory_exceeded_penalty": 0.1}
env = MMAtTouchEnvironment(**env_params)

terminal_time = env.terminal_time
n_steps = env.n_steps
seed = 42
timestamps = np.linspace(0, terminal_time, n_steps + 1)

save_dir = "../../../../experiment-results/"

tensorboard_logdir = save_dir + "tensorboard/SAC-learning-touch/"
best_model_path = save_dir + "SB_models/PPO-best-touch"
trained_model_path = save_dir + "SB_models/PPO-last-touch"
reduced_env = Monitor(ReduceStateSizeWrapper(env))
n_envs = 10
gym.envs.register(id="touch-env-v0", entry_point="__main__:MMAtTouchEnvironment", kwargs=env_params)
vec_env = make_vec_env(env_id="touch-env-v0", n_envs=n_envs, wrapper_class=ReduceStateSizeWrapper)

ppo_policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])])


# Add a linearly decreasing learning rate function
def linear_schedule(initial_value):
    def func(progress):
        return progress * initial_value

    return func


schedule = linear_schedule(0.00003)  # Here, we use the default SB value

ppo_params = {
    "policy": "MlpPolicy",
    "env": vec_env,
    "verbose": 1,
    "policy_kwargs": ppo_policy_kwargs,
    "tensorboard_log": tensorboard_logdir,
    "batch_size": 256,
    "learning_rate": schedule,
    "device": "cpu",
}  # 256 before (batch size)
callback_params = dict(
    eval_env=reduced_env,
    n_eval_episodes=500,  # 200 before  (n_eval_episodes)
    best_model_save_path=best_model_path,
    deterministic=True,
    eval_freq=5000,
)
callback = EvalCallback(**callback_params)
model = PPO(**ppo_params)

print("training model")

model.learn(total_timesteps=5_000_000, callback=callback)

best_model = PPO.load(best_model_path + "/best_model")
max_inventory = 5
inventories = np.arange(-max_inventory, max_inventory + 1, 1)
timesteps = np.linspace(0, terminal_time, 11)
timestep_mesh, inventory_mesh = np.meshgrid(timesteps, inventories)
bid_actions = np.zeros_like(timestep_mesh)
ask_actions = np.zeros_like(timestep_mesh)
for i, row in enumerate(timestep_mesh):
    for j, value in enumerate(row):
        bid_actions[i, j] = best_model.predict(np.array([inventory_mesh[i, j], value]), deterministic=True)[0][0]
        ask_actions[i, j] = best_model.predict(np.array([inventory_mesh[i, j], value]), deterministic=True)[0][1]
extent = 0, 10, np.min(inventories), np.max(inventories)
plt.imshow(bid_actions, extent=extent)

print("saving model figures")
fig_dir = save_dir + "figs/"
os.mkdirs(fig_dir, exists_ok=False)

plt.savefig(fig_dir + "bid_actions")
plt.imshow(ask_actions, extent=extent)
plt.savefig(fig_dir + "ask_actions")
model.save(trained_model_path)
