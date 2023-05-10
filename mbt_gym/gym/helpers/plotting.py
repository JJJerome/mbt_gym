import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment 
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory


def plot_trajectory(env: gym.Env, agent: Agent, seed: int = None):
    # assert env.num_trajectories == 1, "Plotting a trajectory can only be done when env.num_trajectories == 1."
    timestamps = get_timestamps(env)
    observations, actions, rewards = generate_trajectory(env, agent, seed)
    action_dim = actions.shape[1]
    colors = ["r", "k", "b", "g"]
    rewards = np.squeeze(rewards, axis=1)
    cum_rewards = np.cumsum(rewards, axis=-1)
    cash_holdings = observations[:, CASH_INDEX, :]
    inventory = observations[:, INVENTORY_INDEX, :]
    asset_prices = observations[:, ASSET_PRICE_INDEX, :]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    ax3a = ax3.twinx()
    ax1.title.set_text("cum_rewards")
    ax2.title.set_text("asset_prices")
    ax3.title.set_text("inventory and cash holdings")
    ax4.title.set_text("Actions")
    for i in range(env.num_trajectories):
        traj_label = f" trajectory {i}" if env.num_trajectories > 1 else ""
        ax1.plot(timestamps[1:], cum_rewards[i, :])
        ax2.plot(timestamps, asset_prices[i, :])
        ax3.plot(
            timestamps,
            inventory[i, :],
            label=f"inventory" + traj_label,
            color="r",
            alpha=(i + 1) / (env.num_trajectories + 1),
        )
        ax3a.plot(
            timestamps,
            cash_holdings[i, :],
            label=f"cash holdings" + traj_label,
            color="b",
            alpha=(i + 1) / (env.num_trajectories + 1),
        )
        for j in range(action_dim):
            ax4.plot(
                timestamps[0:-1],
                actions[i, j, :],
                label=f"Action {j}" + traj_label,
                color=colors[j],
                alpha=(i + 1) / (env.num_trajectories + 1),
            )
    ax3.legend()
    ax4.legend()
    plt.show()


def plot_stable_baselines_actions(model, env):
    timestamps = get_timestamps(env)
    inventory_action_dict = {}
    price = 100
    cash = 100
    for inventory in [-3, -2, -1, 0, 1, 2, 3]:
        actions = model.predict([price, cash, inventory, 0], deterministic=True)[0].reshape((1, 2))
        for ts in timestamps[1:]:
            actions = np.append(
                actions, model.predict([price, cash, inventory, ts], deterministic=True)[0].reshape((1, 2)), axis=0
            )
        inventory_action_dict[inventory] = actions
    for inventory in [-3, -2, -1, 0, 1, 2, 3]:
        plt.plot(np.array(inventory_action_dict[inventory]).T[0], label=inventory)
    plt.legend()
    plt.show()
    for inventory in [-3, -2, -1, 0, 1, 2, 3]:
        plt.plot(np.array(inventory_action_dict[inventory]).T[1], label=inventory)
    plt.legend()
    plt.show()


def plot_pnl(rewards, symmetric_rewards=None):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    if symmetric_rewards is not None:
        sns.histplot(symmetric_rewards, label="Rewards of symmetric strategy", stat="density", bins=50, ax=ax)
    sns.histplot(rewards, label="Rewards", color="red", stat="density", bins=50, ax=ax)
    ax.legend()
    plt.close()
    return fig


def generate_results_table_and_hist(vec_env: TradingEnvironment, agent: Agent, n_episodes: int = 1000):
    assert vec_env.num_trajectories > 1, "To generate a results table and hist, vec_env must roll out > 1 trajectory."
    observations, actions, rewards = generate_trajectory(vec_env, agent)
    total_rewards = rewards.sum(axis=-1).reshape(-1)
    terminal_inventories = observations[:, INVENTORY_INDEX, -1]
    half_spreads = actions.mean(axis=(-1, -2))

    rows = ["Inventory"]
    columns = ["Mean spread", "Mean PnL", "Std PnL", "Mean terminal inventory", "Std terminal inventory"]
    results = pd.DataFrame(index=rows, columns=columns)
    results.loc[:, "Mean spread"] = 2 * np.mean(half_spreads)
    results.loc["Inventory", "Mean PnL"] = np.mean(total_rewards)
    results.loc["Inventory", "Std PnL"] = np.std(total_rewards)
    results.loc["Inventory", "Mean terminal inventory"] = np.mean(terminal_inventories)
    results.loc["Inventory", "Std terminal inventory"] = np.std(terminal_inventories)
    fig = plot_pnl(total_rewards)
    return results, fig, total_rewards


def get_timestamps(env):
    return np.linspace(0, env.terminal_time, env.n_steps + 1)
