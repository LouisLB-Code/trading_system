# tests/utils/test_market_conditions.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_flash_crash_data(base_price=50000, crash_percent=0.9, periods=100):
    """Create flash crash market data"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
    data = pd.DataFrame(index=dates)
    
    # Normal market conditions for first 70% of periods
    normal_periods = int(periods * 0.7)
    crash_start = int(periods * 0.7)
    recovery_start = int(periods * 0.8)
    
    # Generate price series
    prices = np.ones(periods) * base_price
    
    # Add flash crash
    crash_bottom = base_price * (1 - crash_percent)
    prices[crash_start:recovery_start] = np.linspace(base_price, crash_bottom, recovery_start - crash_start)
    prices[recovery_start:] = np.linspace(crash_bottom, base_price * 0.7, periods - recovery_start)
    
    data['open'] = prices
    data['high'] = prices * 1.02
    data['low'] = prices * 0.98
    data['close'] = prices
    data['volume'] = np.where(
        (np.arange(periods) >= crash_start) & (np.arange(periods) < recovery_start),
        np.random.randint(50000, 100000, periods),  # Higher volume during crash
        np.random.randint(1000, 10000, periods)     # Normal volume
    )
    
    return data

def create_pump_dump_data(base_price=50000, pump_percent=2.0, periods=100):
    """Create pump and dump market data"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
    data = pd.DataFrame(index=dates)
    
    pump_start = int(periods * 0.4)
    pump_peak = int(periods * 0.5)
    dump_end = int(periods * 0.6)
    
    prices = np.ones(periods) * base_price
    peak_price = base_price * (1 + pump_percent)
    
    # Generate pump phase
    prices[pump_start:pump_peak] = np.linspace(base_price, peak_price, pump_peak - pump_start)
    # Generate dump phase
    prices[pump_peak:dump_end] = np.linspace(peak_price, base_price * 0.8, dump_end - pump_peak)
    # Post dump phase
    prices[dump_end:] = base_price * 0.8
    
    data['open'] = prices
    data['high'] = prices * 1.01
    data['low'] = prices * 0.99
    data['close'] = prices
    data['volume'] = np.where(
        (np.arange(periods) >= pump_start) & (np.arange(periods) < dump_end),
        np.random.randint(30000, 80000, periods),   # Higher volume during pump/dump
        np.random.randint(1000, 5000, periods)      # Normal volume
    )
    
    return data

def create_low_liquidity_data(base_price=50000, periods=100):
    """Create low liquidity market data"""
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
    data = pd.DataFrame(index=dates)
    
    prices = base_price + np.random.randn(periods) * 100  # High spread
    
    data['open'] = prices
    data['high'] = prices + np.random.randint(50, 200, periods)
    data['low'] = prices - np.random.randint(50, 200, periods)
    data['close'] = prices
    data['volume'] = np.random.randint(10, 100, periods)  # Very low volume
    
    return data
