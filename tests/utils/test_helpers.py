# tests/utils/test_helpers.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_market_data(symbol='BTC/USDT', periods=100):
    """Create synthetic market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='H')
    data = pd.DataFrame({
        'open': np.random.randn(periods).cumsum() + 100,
        'high': np.random.randn(periods).cumsum() + 102,
        'low': np.random.randn(periods).cumsum() + 98,
        'close': np.random.randn(periods).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, periods)
    }, index=dates)
    return data
