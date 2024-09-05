import pandas as pd
import numpy as np

from simulator.strategy_interface import StrategyInterface
from data_market.get_retornos_sp import get_retornos_sp
from strategy.initial_weights import get_uniform_noneg


def calculate_conversion_line(
    data: pd.DataFrame,
    rolling_periods: int = 9
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
    if ['high', 'low'] not in data.columns:
        raise ValueError("Data must contain 'high' and 'low' columns")
    return (data['high'].rolling(rolling_periods).max() + data['low'].rolling(rolling_periods).min()) / 2


def calculate_baseline(
    data: pd.DataFrame, 
    rolling_periods: int = 26
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
    lagging_periods = -1 * abs(lagging_periods)
    return data['close'].shift(lagging_periods)


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

    conditions = [
        (data['close'] > upper_span),
        (data['close'] < lower_span),
        (data['close'].between(lower_span, upper_span))
    ]
    choices = [1, -1, 0]

    return pd.Series(np.select(conditions, choices, default=0), index=data.index)


class IchimokuStrategy(StrategyInterface):

    def __init__(self):
        """
        Initialize the Ichimoku strategy.
        """
        ...

    def _buy_stocks():
        """ 
        When the close price is above the cloud,
        leading Span A (senkou_span_A) is above 
        the leading span B (senkou_span_B), and
        conversion line (tenkan_sen) moves above
        the base line (kijun_sen), we buy stocks.
        """
        ...

    def _sell_stocks():
        """
        When the close price is below the cloud,
        leading Span A (senkou_span_A) is below 
        the leading span B (senkou_span_B), and
        conversion line (tenkan_sen) moves below
        the base line (kijun_sen), we sell stocks.
        """
        ...
