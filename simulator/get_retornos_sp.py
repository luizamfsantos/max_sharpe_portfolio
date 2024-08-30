import numpy as np


def get_retornos_sp(data, t, window_size):
    """
    Calculate the returns of S&P 500 stocks at a specific time.

    Args:
        data (dict): Data dictionary containing 'sp' and 'prices' DataFrames.
        t (int): The desired time.
        window_size (int): Size of the window for calculations.

    Returns:
        DataFrame: Calculated returns based on the input data.
    """
    sp500 = data['sp']
    prices = data['prices']
    dates_prices = prices.index

    # From days before dates_prices[t], take the greatest
    dates_sp500 = sp500.Date.unique()
    date_sp500 = dates_sp500[dates_prices[t] > dates_sp500].max()

    # Stocks in S&P 500 at time data_sp500
    sp500_t = sp500.loc[sp500['Date'] == date_sp500, 'Ticker']
    # Prices of S&P 500 stocks at time from t-window_size to t
    prices_t = prices.loc[
        dates_prices[t - window_size:t], sp500_t
    ].dropna(axis=1)
    returns_t = np.log(prices_t).diff().fillna(0)

    return returns_t
