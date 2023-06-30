from copy import deepcopy

import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

from mbt_gym.agents.BaselineAgents import CarteaJaimungalMmAgent
from mbt_gym.agents.SbAgent import SbAgent
from mbt_gym.gym.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.wrappers import ReduceStateSizeWrapper
from mbt_gym.rewards.RewardFunctions import CjMmCriterion, PnL
from mbt_gym.stochastic_processes.arrival_models import PoissonArrivalModel
from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
from mbt_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel
from mbt_gym.gym.ModelDynamics import LimitAndMarketOrderModelDynamics

def get_cj_env(
    num_trajectories: int = 1,
    terminal_time: float = 1.0,
    arrival_rate: float = 10.0,
    fill_exponent: float = 0.1,
    phi: float = 0.5,
    alpha: float = 0.001,
    sigma: float = 0.1,
    initial_inventory=(-5, 6),
    random_start: tuple = None,
):
    initial_price = 100
    n_steps = int(10 * terminal_time * arrival_rate)
    step_size = terminal_time / n_steps
    reward_function = CjMmCriterion(phi, alpha) if phi > 0 or alpha > 0 else PnL()
    midprice_model=BrownianMotionMidpriceModel(
            volatility=sigma,
            terminal_time=terminal_time,
            step_size=step_size,
            initial_price=initial_price,
            num_trajectories=num_trajectories,
        )
    arrival_model=PoissonArrivalModel(
        intensity=np.array([arrival_rate, arrival_rate]), step_size=step_size, num_trajectories=num_trajectories
    )
    fill_probability_model=ExponentialFillFunction(
        fill_exponent=fill_exponent, step_size=step_size, num_trajectories=num_trajectories
    )
    env_params = dict(
        terminal_time=terminal_time,
        n_steps=n_steps,
        model_dynamics = LimitAndMarketOrderModelDynamics(midprice_model = midprice_model, arrival_model= arrival_model, fill_probability_model = fill_probability_model, 
                                                          num_trajectories = num_trajectories),
        initial_inventory=initial_inventory,
        reward_function=reward_function,
        max_inventory=n_steps,
        num_trajectories=num_trajectories,
        random_start=random_start,
    )
    return TradingEnvironment(**env_params)


def wrap_env(env: TradingEnvironment):
    env = StableBaselinesTradingEnvironment(trading_env=ReduceStateSizeWrapper(env))
    return VecMonitor(env)


def get_ppo_learner_and_callback(
    env: TradingEnvironment, tensorboard_base_logdir: str = "./tensorboard/", best_model_path: str = "./best_models"
):
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    experiment_string = get_experiment_string(env)
    tensorboard_logdir = tensorboard_base_logdir + "/" + experiment_string
    PPO_params = {
        "policy": "MlpPolicy",
        "env": wrap_env(env),
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
        "tensorboard_log": tensorboard_logdir,
        "n_epochs": 10,
        "batch_size": int(env.n_steps * env.num_trajectories / 4),
        "normalize_advantage": True,
        "n_steps": int(env.n_steps),
        "gae_lambda": 0.95,
        "gamma": 1,
    }
    callback_params = dict(
        eval_env=wrap_env(env),
        n_eval_episodes=10,
        best_model_save_path=best_model_path + "/" + experiment_string,
        deterministic=True,
        eval_freq=env.n_steps * env.num_trajectories * 10,
    )
    callback = EvalCallback(**callback_params)
    model = PPO(**PPO_params)
    return model, callback


def get_experiment_string(env):
    phi = env.reward_function.phi if isinstance(env.reward_function, CjMmCriterion) else 0
    alpha = env.reward_function.alpha if isinstance(env.reward_function, CjMmCriterion) else 0
    return (
        f"n_traj_{env.num_trajectories}__"
        + f"arrival_rate_{env.arrival_model.intensity}__"
        + f"fill_exponent_{env.fill_probability_model.fill_exponent}__"
        + f"phi_{phi}__"
        + f"alpha_{alpha}__"
        + f"initial_inventory_{env.initial_inventory}__"
        + f"random_start_{env.start_time}"
    )


def create_inventory_plot(
    model: PPO,
    env: TradingEnvironment,
    min_inventory: int = -3,
    max_inventory: int = 3,
    reduced_training_indices: list = None,
    model_uses_normalisation: bool = True,
    time_of_action: float = 0.5,
    save_figure: bool = False,
    path_to_figures: str = "./figures",
):
    if model_uses_normalisation:
        normalised_env = StableBaselinesTradingEnvironment(ReduceStateSizeWrapper(env, reduced_training_indices))
    assert env.num_trajectories == 1, "Plotting actions must be done with a single trajectory env"
    ppo_agent = SbAgent(model)
    cj_agent = CarteaJaimungalMmAgent(env=env)
    inventories = np.arange(min_inventory, max_inventory + 1, 1)
    bid_actions, ask_actions, cj_bid_actions, cj_ask_actions = [], [], [], []
    for inventory in inventories:
        state = np.array([[0, inventory, time_of_action, 100]])
        reduced_state = state[:, reduced_training_indices] if reduced_training_indices is not None else state
        if model_uses_normalisation:
            reduced_state = normalised_env.normalise_observation(reduced_state)
        action = ppo_agent.get_action(reduced_state)
        if model_uses_normalisation:
            action = normalised_env.normalise_action(action, inverse=True)
        bid_action, ask_action = action
        cj_bid_action, cj_ask_action = cj_agent.get_action(state).reshape(-1)

        if inventory == min_inventory:
            ask_action = np.NaN
            cj_ask_action = np.NaN
        if inventory == max_inventory:
            bid_action = np.NaN
            cj_bid_action = np.NaN

        bid_actions.append(bid_action)
        ask_actions.append(ask_action)
        cj_bid_actions.append(cj_bid_action)
        cj_ask_actions.append(cj_ask_action)

    plt.plot(inventories, bid_actions, label="bid", color="k")
    plt.plot(inventories, ask_actions, label="ask", color="r")
    plt.plot(inventories, cj_bid_actions, label="bid cj", color="k", linestyle="--")
    plt.plot(inventories, cj_ask_actions, label="ask cj", color="r", linestyle="--")
    plt.legend()
    if save_figure:
        plt.title(get_experiment_string(env))
        plt.savefig(path_to_figures + "/inventory_plots/" + get_experiment_string(env) + ".pdf")
    else:
        plt.show()


def create_time_plot(
    model: PPO,
    env: TradingEnvironment,
    min_inventory: int = -3,
    max_inventory: int = 3,
    reduced_training_indices: list = None,
    model_uses_normalisation: bool = True,
    save_figure: bool = False,
    path_to_figures: str = "./figures",
):
    if model_uses_normalisation:
        normalised_env = StableBaselinesTradingEnvironment(ReduceStateSizeWrapper(env, reduced_training_indices))
    assert env.num_trajectories == 1, "Plotting actions must be done with a single trajectory env"
    ppo_agent = SbAgent(model)
    cj_agent = CarteaJaimungalMmAgent(env=env)
    inventories = np.arange(min_inventory, max_inventory + 1, 1)
    times = np.arange(0, env.terminal_time + 0.01, 0.01)
    inventory_dict = {inventory: [] for inventory in inventories}
    action_dict = {
        "rl bid actions": deepcopy(inventory_dict),
        "cj bid actions": deepcopy(inventory_dict),
        "rl ask actions": deepcopy(inventory_dict),
        "cj ask actions": deepcopy(inventory_dict),
    }
    for inventory in inventories:
        for time in times:
            state = np.array([[0, inventory, time, 100]])
            reduced_state = state[:, reduced_training_indices] if reduced_training_indices is not None else state
            if model_uses_normalisation:
                reduced_state = normalised_env.normalise_observation(reduced_state)
            action = ppo_agent.get_action(reduced_state)
            if model_uses_normalisation:
                action = normalised_env.normalise_action(action, inverse=True)
            bid_action, ask_action = action

            cj_actions = cj_agent.get_action(state)
            cj_bid_action = cj_actions[0, 0]
            cj_ask_action = cj_actions[0, 1]

            if inventory == min_inventory:
                ask_action = np.NaN
                cj_ask_action = np.NaN
            if inventory == max_inventory:
                bid_action = np.NaN
                cj_bid_action = np.NaN

            action_dict["rl bid actions"][inventory].append(bid_action)
            action_dict["rl ask actions"][inventory].append(ask_action)
            action_dict["cj bid actions"][inventory].append(cj_bid_action)
            action_dict["cj ask actions"][inventory].append(cj_ask_action)
    fig, axs = plt.subplots(2, 2, sharey=True, figsize=(15, 10))
    for count, (name, actions) in enumerate(action_dict.items()):
        axs[count // 2, count % 2].set_title(name, fontsize=20)
        for inventory in inventories:
            axs[count // 2, count % 2].plot(times, actions[inventory], label=f"inventory = {inventory}")
            axs[count // 2, count % 2].legend()
    fig.tight_layout()
    if save_figure:
        plt.savefig(path_to_figures + "/time_plots/" + get_experiment_string(env) + ".pdf")
    else:
        plt.show()
