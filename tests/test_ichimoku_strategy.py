from strategy.strategy_ichimoku import calculate_conversion_line
import pandas as pd
from pathlib import Path
from unittest import TestCase
from data_market.datalake import load_data

class IchimokuTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_calculate_conversion_line(self):
        data = self.data['stocks'].copy()
        stock_data = data['AAPL']
        stock_data['conversion_line'] = calculate_conversion_line(stock_data)
        print(stock_data)