import sys

sys.path.append("/LOCAL2/jjerome/GitHub/DRL4AMM/")  # Location on rahul-n
import os

os.environ["PYTHONPATH"] = "/LOCAL2/jjerome/GitHub/DRL4AMM/"  # Location on rahul-n
# Note that we need to set the PYTHONPATH so that the workers are all aware of DRL4AMM
import ray
from ray import tune
from ray.tune.registry import register_env
from copy import copy
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.tune.schedulers import ASHAScheduler


from DRL4AMM.gym.AvellanedaStoikovEnvironment import AvellanedaStoikovEnvironment
from DRL4AMM.gym.wrappers import ReduceStateSizeWrapper
from DRL4AMM.rewards.RewardFunctions import CJ_criterion, PnL

num_workers = 10
info = ray.init(
    ignore_reinit_error=True,
    num_cpus=num_workers + 1,
    include_dashboard=True,
    dashboard_host="0.0.0.0",
    dashboard_port=8266,
)
print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))

terminal_time = 3
arrival_rate = 1.0
env_config = dict(
    terminal_time=terminal_time,
    arrival_rate=arrival_rate,
    n_steps=int(terminal_time * arrival_rate * 10),
    reward_function=PnL(),  # CJ_criterion(phi=2 * 10 ** (-4), alpha=0.0001),
    drift=0.0,
    volatility=0.01,
    fill_exponent=100.0,
    max_inventory=100,
    max_half_spread=10.0,
)


def wrapped_env_creator(env_config: dict):
    return ReduceStateSizeWrapper(AvellanedaStoikovEnvironment(**env_config))


register_env("AvellanedaStoikovEnvironment", wrapped_env_creator)

config = copy(DEFAULT_CONFIG)
config["use_gae"] = True  # Don't use generalised advantage estimation
config["framework"] = "tf2"
config["num_envs_per_worker"] = 50
config["sample_async"] = False
config["entropy_coeff"] = 0.01
config["lr"] = 0.001
config["use_critic"] = True  # False # For reinforce,
config["optimizer"] = "SGD"
config["model"]["fcnet_hiddens"] = [64, 64]
config["eager_tracing"] = True
config["train_batch_size"] = tune.choice([2**7, 2**9, 2**11, 2**13, 2**15])
config["env"] = "AvellanedaStoikovEnvironment"
config["env_config"] = env_config

config["num_workers"] = num_workers
config["num_gpus"] = 1

config["rollout_fragment_length"] = tune.choice([30, 100, 300])
config["model"] = {"fcnet_activation": "tanh", "fcnet_hiddens": [16, 16]}
config["sgd_minibatch_size"] = tune.choice([2**3, 2**5, 2**7])
config["num_sgd_iter"] = tune.choice([10, 20, 30])

tensorboard_logdir = "../data/tensorboard"

print("Starting training")
analysis = tune.run(
    "PPO",
    num_samples=10,
    config=config,
    checkpoint_at_end=True,
    local_dir=tensorboard_logdir,
    stop={"training_iteration": 3000},
    scheduler=ASHAScheduler(metric="episode_reward_mean", mode="max"),
)

best_checkpoint = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean", mode="max"
)
print(best_checkpoint)
path_to_save_dir = tensorboard_logdir
save_best_checkpoint_path(path_to_save_dir, best_checkpoint[0][0])
