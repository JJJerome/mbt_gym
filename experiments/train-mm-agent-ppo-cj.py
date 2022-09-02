from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.append("../")

from mbt_gym.gym.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.wrappers import *
from mbt_gym.rewards.RewardFunctions import PnL, CjCriterion
from mbt_gym.stochastic_processes.midprice_models import *
from mbt_gym.stochastic_processes.fill_probability_models import *
from mbt_gym.stochastic_processes.arrival_models import *

tensorboard_logdir = "./tensorboard/SAC-learning-AS-CJ/"
best_model_path = "./SB_models/SAC-best-CJ"

num_trajectories = 10000
terminal_time = 1.0
arrival_rate = 10.0
fill_exponent = 1
sigma = 0.1
alpha = 0.0001
initial_inventory = (-4,5)
initial_price = 100
n_steps = int(10 * terminal_time * arrival_rate)
step_size = 1/n_steps
timestamps = np.linspace(0, terminal_time, n_steps + 1)
env_params = dict(terminal_time=terminal_time,
                  n_steps=n_steps,
                  initial_inventory = initial_inventory,
                  midprice_model = BrownianMotionMidpriceModel(volatility=sigma,
                                                               terminal_time=terminal_time,
                                                               step_size=step_size,
                                                               initial_price=initial_price,
                                                               num_trajectories=num_trajectories),
                  arrival_model = PoissonArrivalModel(intensity=np.array([arrival_rate,arrival_rate]),
                                                      step_size=step_size,
                                                      num_trajectories=num_trajectories),
                  fill_probability_model = ExponentialFillFunction(fill_exponent=fill_exponent,
                                                                   step_size=step_size,
                                                                   num_trajectories=num_trajectories),
                  max_inventory=n_steps,
                  reward_function = CjCriterion(),
                  num_trajectories=num_trajectories)

env = ReduceStateSizeWrapper(TradingEnvironment(**env_params))
sb_env = StableBaselinesTradingEnvironment(trading_env=env)

policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[64, 64])])
PPO_params = {"policy":'MlpPolicy', "env": sb_env, "verbose":1,
              "policy_kwargs":policy_kwargs,
              "tensorboard_log":tensorboard_logdir,
              #"batch_size": int(n_steps * num_trajectories / 20),
              "n_steps": int(n_steps)} #256 before (batch size)
callback_params = dict(eval_env=sb_env, n_eval_episodes = 2048, #200 before  (n_eval_episodes)
                       best_model_save_path = best_model_path,
                       deterministic=True)

callback = EvalCallback(**callback_params)
model = PPO(**PPO_params, device="cpu")

model.learn(total_timesteps = 50_000_000, callback=callback)

