# Load Data
import pandas as pd
from pathlib import Path


def load_data():
    """
    Load the available data from the dataset directory.
    """

    # path
    basepath = Path(__file__).parent / "datasets/"

    # # load fed rate
    # fed_rate_df = pd.read_parquet(basepath / "US/fed_rate.parquet")

    # # load sp500_components
    # sp_df = pd.read_parquet(basepath / "US/sp_comp.parquet")

    # load prices_sp
    # melt_prices_df = pd.read_parquet(basepath / "US/prices_sp.parquet")
    # prices_df = melt_prices_df.pivot_table(
    #     index="Date", columns="Ticker", values="value")
    prices_df = pd.read_parquet(basepath / "BR/prices_ibov.parquet")
    prices_complete_df = pd.read_parquet(basepath / "BR/bovespa_ochl.parquet")

    # create a dictionary with data
    # dict_data = {'fed_rate': fed_rate_df, 'sp': sp_df, 'stocks': prices_df}
    dict_data = {'stocks': prices_df, 'prices_complete': prices_complete_df}
    return dict_data

if __name__ == "__main__":
    data = load_data()
    print(data)