import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from simulator.strategy_interface import StrategyInterface


class MaxSharpeStrategy(StrategyInterface):
    def calculate_next_weights(
        self,
        data: dict[str, pd.DataFrame],
        t: int
    ) -> pd.DataFrame:
        # Filter only data until time t
        prices_df = data['stocks'].sort_values('Data')
        prices_df = prices_df.iloc[:t]
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(prices_df)
        S = risk_models.sample_cov(prices_df)
        # Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        opt_weights = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=[
                                             'weights']).reset_index(names='ticker')
        opt_weights['date'] = prices_df.index[-1]
        return opt_weights
