import pandas as pd
import numpy as np

from simulator.strategy_interface import StrategyInterface
from data_market.get_retornos_sp import get_retornos_sp
from strategy.initial_weights import get_uniform_noneg


def calculate_conversion_line(
    data: pd.DataFrame,
    rolling_periods: int | None = 9
) -> pd.Series:
    """
    Calculate the conversion line (tenkan_sen) 
    of the Ichimoku strategy. Using the formula:
    Tenkan-sen = (k-period high + k-period low) / 2
    It represents the midpoint of the price action
    over the last k periods. 
    It aims to smooth out short-term price movements
    and provide a trend-following signal.
    When the price is above the Tenkan-sen, it suggests
    a bullish (increasing) trend, and when the price is
    below the Tenkan-sen, it suggests a bearish (decreasing)
    trend.
    """
    if rolling_periods is None:
        rolling_periods = 9
    if ['high', 'low'] not in data.columns:
        raise ValueError("Data must contain 'high' and 'low' columns")
    return (data['high'].rolling(rolling_periods).max() + data['low'].rolling(rolling_periods).min()) / 2


def calculate_baseline(
    data: pd.DataFrame,
    rolling_periods: int | None = 26
) -> pd.Series:
    """
    Calculate the base line (kijun_sen)
    of the Ichimoku strategy. Using the formula:
    Kijun-sen = (k-period high + k-period low) / 2
    It represents the baseline or the equilibrium level
    in the Ichimoku cloud. It is calculated over a longer
    period than the Tenkan-sen. It aims to provide a
    medium-term trend-following signal, smoothing out
    short-term price movements.
    Traders often use it to assess the overall direction
    of the market and to identify potential support or 
    resistance levels.
    """
    if rolling_periods is None:
        rolling_periods = 26
    return calculate_conversion_line(data, rolling_periods)


def calculate_leading_span_A(
    data: pd.DataFrame,
    baseline_rolling_periods: int | None = None,
    conversion_rolling_periods: int | None = None,
    future_periods: int = 52
) -> pd.Series:
    """
    Calculate the leading span A (senkou_span_A)
    of the Ichimoku strategy. 
    It provides insights into potential future support
    and resistance levels by averaging key price points
    and projecting them forward. 
    It is often plotted 52 periods ahead of the current
    """
    if future_periods is None:
        future_periods = 52
    if 'baseline' not in data.columns:
        data['baseline'] = calculate_baseline(data, baseline_rolling_periods)
    if 'conversion_line' not in data.columns:
        data['conversion_line'] = calculate_conversion_line(
            data, conversion_rolling_periods)
    return ((data['baseline'] + data['conversion_line']) / 2).shift(future_periods)


def calculate_leading_span_B(
    data: pd.DataFrame,
    rolling_periods: int = 26,
    future_periods: int = 30
) -> pd.Series:
    """
    Calculate the leading span B (senkou_span_B)
    of the Ichimoku strategy. 
    It provides insights into potential future support
    and resistance levels by averaging key price points
    and projecting them forward. 
    It is often plotted 52 periods ahead of the current
    """
    if rolling_periods is None:
        rolling_periods = 26
    if future_periods is None:
        future_periods = 30
    if ['high', 'low'] not in data.columns:
        raise ValueError("Data must contain 'high' and 'low' columns")
    return ((data['high'].rolling(rolling_periods).max() + data['low'].rolling(rolling_periods).min()) / 2).shift(future_periods)


def calculate_lagging_span(
    data: pd.DataFrame,
    lagging_periods: int = 30
) -> pd.Series:
    """ 
    Calculate the lagging span (chikou_span)
    of the Ichimoku strategy.
    It is the close price plotted k periods in the past.
    """
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column")
    if lagging_periods is None:
        lagging_periods = 30
    lagging_periods = -1 * abs(lagging_periods)
    return data['close'].shift(lagging_periods)

def get_entry_signal(
    data: pd.DataFrame
) -> None:
    if signal not in data.columns:
        data['signal'] = np.nan

    # Prices are above the cloud
    condition_1 = (data.close > data.leading_span_A) & (data.close > data.leading_span_B)

    # Leading Span A (senkou_span_A) is rising above the leading span B (senkou_span_B)
    condition_2 = (data.leading_span_A > data.leading_span_B)

    # Conversion Line (tenkan_sen) moves above Base Line (kijun_sen)
    condition_3 = (data.conversion_line > data.base_line)

    # Combine the conditions and store in the signal column 1
    data.loc[condition_1 & condition_2 & condition_3, 'signal'] = 1

def get_exit_signal(
    data: pd.DataFrame
) -> None:
    if signal not in data.columns:
        data['signal'] = np.nan

    # Prices are below the cloud
    condition_1 = (df.close < df.leading_span_A) & (df.close < df.leading_span_B)

    # Leading Span A (senkou_span_A) is falling below
    # the leading span B (senkou_span_B)
    condition_2 = (df.leading_span_A < df.leading_span_B)

    # Conversion Line (tenkan_sen) moves below Base Line (kijun_sen)
    condition_3 = (df.conversion_line < df.base_line)

    # Combine the conditions and store in the signal
    # column 0 when all the conditions are true
    df.loc[condition_1 & condition_2 & condition_3, 'signal'] = 0

    # Fill missing values in the 'signal' column using
    # forward fill (ffill) method
    df.signal.fillna(method='ffill', inplace=True)

def calculate_cloud(
    data: pd.DataFrame,
    baseline_rolling_periods: int | None = None,
    conversion_rolling_periods: int | None = None,
    future_periods: int | None = None
) -> pd.Series[int]:
    """
    Calculate whether the prices are above, below, 
    or within the cloud.
    It returns: 
    +1 if the close price is above the cloud, # trend up
    -1 if the close price is below the cloud, # trend down
    0 if the close price is within the cloud. # flat
    """
    if 'leading_span_A' not in data.columns:
        data['leading_span_A'] = calculate_leading_span_A(
            data, baseline_rolling_periods, conversion_rolling_periods, future_periods)
    if 'leading_span_B' not in data.columns:
        data['leading_span_B'] = calculate_leading_span_B(
            data, baseline_rolling_periods, future_periods)

    upper_span = data[['leading_span_A', 'leading_span_B']].max(axis=1)
    lower_span = data[['leading_span_A', 'leading_span_B']].min(axis=1)

    # Prices are above/below or within the cloud
    condition_price_above = (data.close > data.leading_span_A) & (
        data.close > data.leading_span_B)
    condition_price_below = data.close < data.leading_span_A & (
        data.close < data.leading_span_B)
    data['cloud_condition_1'] = np.where(condition_price_above, 1, np.where(
        condition_price_below, -1, 0))

    # Leading Span A is above/below the leading span B
    condition_leading_span_A_above = data.leading_span_A > data.leading_span_B
    condition_leading_span_A_below = data.leading_span_A < data.leading_span_B
    data['cloud_condition_2'] = np.where(condition_leading_span_A_above & condition_price_above, 1, np.where(
        condition_leading_span_A_below & condition_price_below, -1, 0))

    # Conversion Line moves above/below the Base Line
    condition_conversion_above = data.conversion_line > data.base_line
    condition_conversion_below = data.conversion_line < data.base_line
    data['cloud_condition_3'] = np.where(condition_conversion_above & condition_price_above, 1, np.where(
        condition_conversion_below & condition_price_below, -1, 0))

    # Combine the conditions
    data['cloud_combined'] = data['cloud_condition_1'] + \
        data['cloud_condition_2'] + data['cloud_condition_3']
    data['cloud'] = np.where(data['cloud_combined'] == 3, 1, np.where(
        data['cloud_combined'] == -3, -1, 0))
    
    return data['cloud']

class IchimokuStrategy(StrategyInterface):

    def __init__(self):
        self.market = None

    def _set_market_condition(self, data: pd.DataFrame, index: int):
        if self.market is not None:
            if self.market.index == index:
                return  # already set
        self.market = MarketCondition(data, index)

    def _buy_stocks(data: pd.DataFrame, index: int) -> list[str]:
        """ 
        When the close price is above the cloud,
        leading Span A (senkou_span_A) is above 
        the leading span B (senkou_span_B), and
        conversion line (tenkan_sen) moves above
        the base line (kijun_sen), we buy stocks.
        Returns:
            list[str]: List of stocks to buy.
        """
        _set_market_condition(data, index)
        return self.market.cloud[self.market.cloud == 1].index.tolist()

    def _sell_stocks(data: pd.DataFrame, index: int) -> list[str]:
        """
        When the close price is below the cloud,
        leading Span A (senkou_span_A) is below 
        the leading span B (senkou_span_B), and
        conversion line (tenkan_sen) moves below
        the base line (kijun_sen), we sell stocks.
        Returns:
            list[str]: List of stocks to sell.
        """
        _set_market_condition(data, index)
        return self.market.cloud[self.market.cloud == -1].index.tolist()

    def calculate_next_weights(self, data: pd.DataFrame, t: int, size=30, window_size=500) -> pd.DataFrame:
        """
        Implement the Ichimoku strategy.
        """
        stocks_buy = _buy_stocks(
            data, t)  # create a portfolio of stocks to buy

        # TODO: balance the portfolio
        return stocks


class MarketCondition():

    def __init__(self, data: pd.DataFrame, index: int):
        # only look at the data up to the current index
        self.data = data.iloc[:index, :]
        self.index = index
        self.stock_list = self._get_stocks()
        self.cloud = {stock: self._calculate_stock_cloud_condition(
            stock) for stock in self.stock_list}

    def _get_stocks(self) -> list[str]:
        return [col for col in self.data.columns if col not in ['Date']]

    def _get_stock_data(self, stock: str) -> pd.DataFrame:
        return self.data[stock]

    def _calculate_stock_cloud_condition(self, stock: str):
        stock_data = self._get_stock_data(stock)
        baseline = calculate_baseline(stock_data)
        conversion_line = calculate_conversion_line(stock_data)
        leading_span_A = calculate_leading_span_A(stock_data)
        leading_span_B = calculate_leading_span_B(stock_data)
        #return calculate_cloud(stock_data)
