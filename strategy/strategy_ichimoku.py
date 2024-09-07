import pandas as pd
import numpy as np

from simulator.strategy_interface import StrategyInterface


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
    if ['High', 'Low'] not in data.columns:
        raise ValueError("Data must contain 'High' and 'Low' columns")
    return (data['High'].rolling(rolling_periods).max() + data['Low'].rolling(rolling_periods).min()) / 2


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
    if ['High', 'Low'] not in data.columns:
        raise ValueError("Data must contain 'High' and 'Low' columns")
    return ((data['High'].rolling(rolling_periods).max() + data['Low'].rolling(rolling_periods).min()) / 2).shift(future_periods)


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


def calculate_cloud(
        data: pd.DataFrame,
        baseline_rolling_periods: int | None = None,
        conversion_rolling_periods: int | None = None,
        future_periods: int | None = None) -> pd.Series:
    """
    Calculate whether the prices are above, below, or within the cloud.
    Returns:
    +1 if the close price is above the cloud (trend up)
    -1 if the close price is below the cloud (trend down)
    0 if the close price is within the cloud (flat)
    """
    # Calculate leading spans if not present
    for span, func in [('A', calculate_leading_span_A), ('B', calculate_leading_span_B)]:
        col = f'leading_span_{span}'
        if col not in data.columns:
            data[col] = func(data, baseline_rolling_periods,
                             conversion_rolling_periods, future_periods)

    # Define conditions
    price_above = (data.close > data.leading_span_A) & (
        data.close > data.leading_span_B)
    price_below = (data.close < data.leading_span_A) & (
        data.close < data.leading_span_B)
    span_A_above = data.leading_span_A > data.leading_span_B
    conversion_above = data.conversion_line > data.baseline

    # Calculate cloud conditions
    cloud_conditions = [
        np.where(price_above, 1, np.where(price_below, -1, 0)),
        np.where(span_A_above & price_above, 1, np.where(
            ~span_A_above & price_below, -1, 0)),
        np.where(conversion_above & price_above, 1, np.where(
            ~conversion_above & price_below, -1, 0))
    ]

    # Combine conditions
    data['cloud_combined'] = sum(cloud_conditions)
    return pd.Series(np.where(data['cloud_combined'] == 3, 1,
                              np.where(data['cloud_combined'] == -3, -1, 0)),
                     name='cloud')


class IchimokuStrategy(StrategyInterface):
    """
    Ichimoku Cloud strategy finds the best time to buy and
    sell a stock. It does not optimize the portfolio weights.
    So we'll implement a naive strategy that allocates equal
    weight to all stocks that are in the buy list and 0 weight
    to all stocks that are in the sell list.
     """

    def __init__(self):
        self.market = None
        self.current_weights = None

    def _set_market_condition(
        self,
        data: pd.DataFrame,
        index: pd.Index
    ) -> None:
        if self.market is not None:
            if self.market.index == index.max():
                return  # already set
        self.market = MarketCondition(data, index)

    def _buy_stocks(self):
        """ 
        When the close price is above the cloud,
        leading Span A (senkou_span_A) is above 
        the leading span B (senkou_span_B), and
        conversion line (tenkan_sen) moves above
        the base line (kijun_sen), we buy stocks.
        """
        stocks_to_buy = [
            ticker for ticker, cloud_signal in self.market.cloud.items() if cloud_signal == 1]
        if self.current_weights is None:
            self.current_weights = pd.DataFrame({
                'ticker': stocks_to_buy,
                'weights': 1/len(stocks_to_buy),
            })
        else:
            available_portfolio = 1 - self.current_weights['weights'].sum()
            if available_portfolio and stocks_to_buy:
                added_weights_per_stock = available_portfolio / \
                    len(stocks_to_buy)
                # Update weights for existing stocks and add new ones
                for ticker in stocks_to_buy:
                    if ticker in self.current_weights['ticker'].values:
                        self.current_weights.loc[self.current_weights['ticker']
                                                 == ticker, 'weights'] += added_weights_per_stock
                    else:
                        new_stock = pd.DataFrame(
                            {'ticker': [ticker], 'weights': [added_weights_per_stock]})
                        self.current_weights = pd.concat(
                            [self.current_weights, new_stock], ignore_index=True)

    def _sell_stocks(self):
        """
        When the close price is below the cloud,
        leading Span A (senkou_span_A) is below 
        the leading span B (senkou_span_B), and
        conversion line (tenkan_sen) moves below
        the base line (kijun_sen), we sell stocks.
        """
        if self.current_weights is None:
            return
        stocks_to_sell = [
            ticker for ticker, cloud_signal in self.market.cloud.items() if cloud_signal == -1]
        if sum(self.current_weights['ticker'].isin(stocks_to_sell)):
            self.current_weights.loc[self.current_weights['ticker'].isin(
                stocks_to_sell), 'weights'] = 0

    def calculate_next_weights(
        self,
        data: dict[str, pd.DataFrame],
        t: int
    ) -> pd.DataFrame:
        """
        Implement the Ichimoku strategy.
        data (dict): Data dictionary containing 'sp', 
            'prices' and 'fed_rate' DataFrames.
        t (int): Current time index.

        """
        stocks_df = data['stocks']
        index_considered = stocks_df.index[:t]  # returns list of dates up to t
        complete_prices_df = data.get('prices_complete')
        if not complete_prices_df:
            raise ValueError("Data must contain open, high, low, "
                             "close columns in order to calculate Ichimoku Cloud")
        self._set_market_condition(complete_prices_df, index_considered)
        self._sell_stocks()
        self._buy_stocks()
        opt_weights = self.current_weights.copy()
        opt_weights['date'] = self.market.index
        return opt_weights


class MarketCondition():

    def __init__(self, data: pd.DataFrame, index: pd.Index):
        # only look at the data up to the current index
        self.data = data[data['Date'].isin(pd.to_datetime(index))].sort_values(
            'Date').reset_index(drop=True)
        self.index = index.max()
        self.stock_list = self._get_stocks()
        self.cloud = {stock: self._calculate_stock_cloud_condition(
            stock) for stock in self.stock_list}

    def _get_stocks(self) -> list[str]:
        return data['Symbol'].unique().tolist()

    def _get_stock_data(self, stock: str) -> pd.DataFrame:
        return self.data[self.data['Symbol'] == stock].copy().reset_index(drop=True)

    def _calculate_stock_cloud_condition(self, stock: str):
        stock_data = self._get_stock_data(stock)
        baseline = calculate_baseline(stock_data)
        conversion_line = calculate_conversion_line(stock_data)
        leading_span_A = calculate_leading_span_A(stock_data)
        leading_span_B = calculate_leading_span_B(stock_data)
        return calculate_cloud(stock_data)
