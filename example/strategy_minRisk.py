import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random

def strategy_minRisk(data, t, size = 30, window_size = 500):

    def _get_retornos_sp(data, t, window_size):
        """
        Calculates the returns of sp stocks at time 't'.

        Args:
            t (int): The desired time.

        Returns:
            DataFrame: Returns calculated based on the input data.

        """
        sp500 = data['sp']
        prices = data['prices']
        dates_prices = prices.index

        local_sp500 = dates_prices[t] > sp500['Date']
        data_sp500 = sp500['Date'][local_sp500].tail(1).values[0]

        sp500_t = sp500['Ticker'].loc[sp500['Date'] == data_sp500]
        prices_t = prices[sp500_t].loc[dates_prices[t-window_size:t]].dropna(axis=1)
        returns_t = np.log(prices_t).diff().fillna(0)

        return returns_t

    def _get_ini_w_unif(size):
        aux = [random.uniform(0, 2/size) for _ in range(size)]
        initial_weights = np.array(aux)  # Start with all weights set to 0
        return initial_weights

    def _calculate_risk_stat(weights, returns):
        return (weights @ returns.cov() @ weights * 252) ** 0.5


    returns = _get_retornos_sp(data, t, window_size)
    if (size == None) | (size > len(returns.columns)):
        size = len(returns.columns)

    stocks_sel = returns.std().sort_values().head(size).index
    returns_sel = returns[stocks_sel]

    initial_weights = _get_ini_w_unif(size)
    
    # Define optimization constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights must equal 0
    ]

    # Define bounds for weights (-2 <= weight <= 2)
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
                                    'date': [data['prices'].index[t]] * len(result.x),
                                    'ticker': returns_sel.columns,
                                    'weights': result.x,
                                    })

    return opt_weights