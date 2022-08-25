# import external modules
from copy import copy
import torch

from typing import Dict
import ray
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Add DRL4AMM to path so that we can import it
import sys
sys.path.append("/LOCAL2/jjerome/GitHub/DRL4AMM/")

# Set the PYTHONPATH so that the workers are all aware of DRL4AMM
import os
os.environ["PYTHONPATH"] = "/LOCAL2/jjerome/GitHub/DRL4AMM/"

from DRL4AMM.gym.MarketMakingEnvironment import MarketMakingEnvironment
from DRL4AMM.gym.wrappers import ReduceStateSizeWrapper
from DRL4AMM.rewards.RewardFunctions import PnL
from DRL4AMM.gym.probability_models import *

import multiprocessing
num_cpus = multiprocessing.cpu_count()
num_gpus = torch.cuda.device_count()

num_workers = 10
print("Initialising Ray")
info = ray.init(ignore_reinit_error=True,num_cpus=num_cpus,num_gpus = num_gpus, include_dashboard=True)
print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))

as_env = MarketMakingEnvironment()
terminal_time = as_env.terminal_time
n_steps = as_env.n_steps
timestamps = np.linspace(0, terminal_time, n_steps + 1)

lambda_ = 40
as_terminal_time = 1.0
n_steps = int(lambda_ * as_terminal_time/0.1)
as_arrival = PoissonArrivalModel(intensity=[lambda_,lambda_], step_size=as_terminal_time/n_steps)
as_fill = ExponentialFillFunction(fill_exponent=1.5, step_size=as_terminal_time/n_steps)
as_midprice = BrownianMotionMidpriceModel(volatility=0.01, step_size=as_terminal_time/n_steps)
as_max_inv = 100
as_reward_func = PnL()

as_config = dict(terminal_time=as_terminal_time,
                 midprice_model=as_midprice,
                 arrival_model=as_arrival,
                 fill_probability_model=as_fill,
                 max_inventory=as_max_inv,
                 n_steps = n_steps,
                 reward_function=as_reward_func)

as_env = MarketMakingEnvironment(**as_config)


def wrapped_env_creator(env_config:dict):
    env = ReduceStateSizeWrapper(MarketMakingEnvironment(**env_config))
    return env


register_env("AsEnv-v0", wrapped_env_creator)

from ray.rllib.agents.ppo import DEFAULT_CONFIG

config = copy(DEFAULT_CONFIG)
config["use_gae"] = False  # Don't use generalised advantage estimation
config["framework"] = "tf"
config["env"] = "AsEnv-v0"
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config["rollout_fragment_length"] = 200
config["env_config"] = as_config
config["num_workers"] = num_cpus - 1
config["num_envs_per_worker"] = 5
config["model"] = {"fcnet_activation": "tanh", "fcnet_hiddens": [16, 16]}
config["batch_mode"] = "complete_episodes"
config["train_batch_size"] = config["num_workers"] * config["rollout_fragment_length"] * 10
config["num_gpus"] = num_gpus # 1/num_cpus
config["horizon"] = as_config["n_steps"]
config["lr"] = 5e-5 # tune.loguniform(1e-8, 1e-4)
# config["lr_schedule"] = schedule
config["entropy_coeff"] = 0.0
config["optimizer"] = torch.optim.SGD
tensorboard_logdir = "tensorboard"

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        episode.custom_metrics["actions"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        episode.custom_metrics["actions"].append(episode.last_action_for())

config["callbacks"] = MyCallbacks

print("Traing PPO agent.\n")

# Resume from checkpoint
analysis = tune.run(
    "PPO",
    config=config,
    checkpoint_at_end=True,
    local_dir=tensorboard_logdir,
    stop={"training_iteration": 100},
    name = "ppo-for-pnl",
    checkpoint_freq=3,
    checkpoint_score_attr="episode_reward_mean",
    mode = "max",
    reuse_actors=True,
    keep_checkpoints_num=3,
    restore="/home/staffi/ecco/jjerome/Documents/notebooks/DRL4AMM/rllib-approach/tensorboard/ppo-for-pnl/PPO_AsEnv-v0_a9544_00000_0_2022-08-18_16-16-00/checkpoint_000099/checkpoint-99"
)


def save_best_checkpoint_path(path_to_save_dir: str, best_checkpoint_path: str):
    text_file = open(path_to_save_dir + "/best_checkpoint_path.txt", "wt")
    text_file.write(best_checkpoint_path)
    text_file.close()


best_checkpoint = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"), metric="episode_reward_mean"
    )
print(best_checkpoint)
path_to_save_dir = "/home/ray"
save_best_checkpoint_path(path_to_save_dir, best_checkpoint[0][0])