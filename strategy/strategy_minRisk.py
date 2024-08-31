import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict

from simulator.strategy_interface import StrategyInterface

from data_market.get_retornos_sp import get_retornos_sp
from strategy.initial_weights import get_uniform_noneg


def _sel_stocks(returns, size):
    """
    Select top 'size' stocks based on standard deviation of returns.

    Args:
        returns (DataFrame): DataFrame containing returns data.
        size (int): Number of stocks to select.

    Returns:
        DataFrame: Selected returns data.
    """
    if (size is None) or (size > len(returns.columns)):
        size = len(returns.columns)
    # Select the top 'size' stocks based on standard deviation
    stocks_sel = returns.std().sort_values().head(size).index

    # Filter the returns to the top symbols
    returns_sel = returns[stocks_sel]
    return returns_sel


def _calculate_risk_stat(weights, returns):
    """
    Calculate the risk statistic based on portfolio weights and returns' covariance.

    Args:
        weights (array-like): Portfolio weights.
        returns (DataFrame): DataFrame containing returns data.

    Returns:
        float: Calculated risk statistic.
    """
    return (weights @ returns.cov() @ weights * 252) ** 0.5


class MinRiskStrategy(StrategyInterface):

    def __init__(self):
        """
        Initialize the minimum risk strategy.
        """
        pass

    def calculate_next_weights(self, data: Dict[str, pd.DataFrame], t: int, size=30, window_size=500) -> pd.DataFrame:
        """
        Implement a minimum risk strategy.

        Args:
            data (dict): Data dictionary containing 'sp' and 'prices' DataFrames.
            t (int): The desired time.
            size (int, optional): Number of stocks to consider. Defaults to 30.
            window_size (int, optional): Size of the window for calculations. Defaults to 500.

        Returns:
            DataFrame: Optimal weights for the selected stocks.
        """
        returns = get_retornos_sp(data, t, window_size)
        returns_sel = _sel_stocks(returns, size)
        initial_weights = get_uniform_noneg(size)

        # Define optimization constraints
        constraints = [
            # Sum of weights must equal 1
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        # Define bounds for weights (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(size))

        # Perform optimization
        result = minimize(
            _calculate_risk_stat,
            initial_weights,
            args=(returns_sel,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        opt_weights = pd.DataFrame({
            'date': [data['stocks'].index[t]] * len(result.x),
            'ticker': returns_sel.columns,
            'weights': result.x,
        })

        return opt_weights
