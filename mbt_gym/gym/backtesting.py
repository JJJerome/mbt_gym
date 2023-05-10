import gym
import numpy as np
import pandas as pd
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX
from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
import warnings


def get_sharpe_ratio(env: gym.Env, agent: Agent, risk_free_rate: float = 0.099):
    """
    The Annualized Sharpe Ratio is calculated as:
                Sharpe_Ratio = sqrt(num_steps)*(Returns  - Risk Free Rate)/(Std of Return)
    It measures the reward in relation to risk.
    """
    assert env.num_trajectories == 1, "Backtesting is applied on a single trajectory"
    obs, _, _ = generate_trajectory(env, agent)
    portfolio_values = (obs[:, CASH_INDEX, :] + obs[:, INVENTORY_INDEX, :] * obs[:, ASSET_PRICE_INDEX, :]).squeeze()
    if min(np.abs(portfolio_values)) < 1e-6:
        warnings.warn("Runtime Warning: Division by Zero")
    return_pcts = np.diff(portfolio_values, 1) / portfolio_values[1:]
    annualized_std_returns = return_pcts.std() * np.sqrt(env.n_steps)
    return_pcts_mean = return_pcts.mean()
    if return_pcts_mean < 0:
        warnings.warn("Warning: Mean Return % is negative. Sharpe Ratio may not be appropriate.")
    return (return_pcts_mean * env.n_steps - risk_free_rate) / annualized_std_returns


def get_sortino_ratio(env: gym.Env, agent: Agent, risk_free_rate: float = 0.099):
    """
    The Sortino Ratio is the Sharpe Ratio but restricted to only negative returns.
                Sortino_Ratio = sqrt(num_steps)*(Returns - Risk Free Rate)/(Std of negative returns)
    """
    assert env.num_trajectories == 1, "Backtesting is applied on a single trajectory"
    obs, _, _ = generate_trajectory(env, agent)
    portfolio_values = (obs[:, CASH_INDEX, :] + obs[:, INVENTORY_INDEX, :] * obs[:, ASSET_PRICE_INDEX, :]).squeeze()
    if min(np.abs(portfolio_values)) < 1e-6:
        warnings.warn("Runtime Warning: Division by Zero")
    return_pcts = np.diff(portfolio_values, 1) / portfolio_values[1:]
    loss_pcts = return_pcts[return_pcts < 0]
    annualized_std_returns = loss_pcts.std() * np.sqrt(env.n_steps)
    return_pcts_mean = return_pcts.mean()
    if return_pcts_mean < 0:
        warnings.warn("Warning: Mean Return % is negative. Sortino Ratio may not be appropriate.")
    return (return_pcts_mean * env.n_steps - risk_free_rate) / annualized_std_returns


def get_maximum_drawdown(env: TradingEnvironment, agent: Agent):
    """
    The maximum drawdown is the biggest difference between a peak and a trough in portfolio value.
    """
    assert env.num_trajectories == 1, "Backtesting is applied on a single trajectory"
    obs, _, _ = generate_trajectory(env, agent)
    portfolio_values = (obs[:, CASH_INDEX, :] + obs[:, INVENTORY_INDEX, :] * obs[:, ASSET_PRICE_INDEX, :]).squeeze()
    return_pcts = pd.Series(np.diff(portfolio_values, 1) / portfolio_values[1:])
    cum_prods = (return_pcts + 1).cumprod()
    peak = cum_prods.expanding(min_periods=1).max()
    drawdown = (cum_prods / peak) - 1
    return drawdown.min()
