from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from DRL4AMM.gym.wrappers import *
from DRL4AMM.gym.MarketMakingEnvironment import MarketMakingEnvironment
from DRL4AMM.rewards.RewardFunctions import CJ_criterion
from DRL4AMM.gym.probability_models import *

# Add a linearly decreasing learning rate function
def linear_schedule(initial_value):
    def func(progress):
        return progress * initial_value
    return func
schedule = linear_schedule(0.00003) # Here, we use the default SB value

tensorboard_logdir = "./tensorboard/SAC-learning-AS-CJ/"
best_model_path = "./SB_models/SAC-best-CJ"

terminal_time = 30.0
arrival_rate = 1.0
fill_exponent = 100
alpha = 0.0001
phi = 2E-4
sigma = 0.01
initial_price = 100
n_steps = int(10 * terminal_time/arrival_rate)
step_size = 1/n_steps
timestamps = np.linspace(0, terminal_time, n_steps + 1)
env_params = dict(terminal_time=terminal_time,
                  n_steps=n_steps,
                  reward_function = CJ_criterion(phi=phi,alpha=alpha),
                  midprice_model = BrownianMotionMidpriceModel(volatility=sigma,
                                                               terminal_time=terminal_time,
                                                               step_size=step_size,
                                                               initial_price=initial_price),
                  arrival_model = PoissonArrivalModel(intensity=arrival_rate,
                                                      step_size=step_size),
                  fill_probability_model = ExponentialFillFunction(fill_exponent=fill_exponent,
                                                                   step_size=step_size),
                  max_inventory=n_steps)

mm_env = MarketMakingEnvironment(**env_params)
reduced_env = ReduceStateSizeWrapper(mm_env)
n_envs = 14
gym.envs.register(id="mm-env-v0", entry_point="__main__:MarketMakingEnvironment", kwargs=env_params)
vec_env = make_vec_env(env_id="mm-env-v0", n_envs=n_envs, wrapper_class=ReduceStateSizeWrapper)

policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[64, 64])])
PPO_params = {"policy":'MlpPolicy', "env": vec_env, "verbose":1,
              "policy_kwargs":policy_kwargs,
              "tensorboard_log":tensorboard_logdir,
              "batch_size": 10000, "learning_rate": schedule} #256 before (batch size)
callback_params = dict(eval_env=reduced_env, n_eval_episodes = 10000, #200 before  (n_eval_episodes)
                       best_model_save_path = best_model_path,
                       deterministic=True)

callback = EvalCallback(**callback_params)
model = PPO(**PPO_params, device="cpu")

model.learning_rate = linear_schedule(0.000001)

model.learn(total_timesteps = 3_000_000, callback=callback)