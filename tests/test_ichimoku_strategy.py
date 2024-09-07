from strategy.strategy_ichimoku import calculate_conversion_line
import pandas as pd
from pathlib import Path
from unittest import TestCase

class IchimokuTest(TestCase):

    @classmethod
    def setUpClass(cls):
        test_data_path = Path(__file__).parent / 'data-test/stocks_subset.csv'
        cls.data = {'stocks': pd.read_csv(test_data_path)}

    def test_calculate_conversion_line(self):
        data = self.data['stocks'].copy()
        stock_data = data['AAPL']
        stock_data['conversion_line'] = calculate_conversion_line(stock_data)
        print(stock_data)