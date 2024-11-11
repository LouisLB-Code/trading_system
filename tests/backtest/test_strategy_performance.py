# tests/backtest/test_strategy_performance.py
import pytest
from datetime import datetime, timedelta

def test_strategy_backtest_performance(trading_system):
    """Test strategy performance in backtest"""
    # Setup backtest parameters
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Run backtest
    results = await trading_system.run_backtest(
        start_date=start_date,
        end_date=end_date,
        pairs=['BTC/USDT']
    )
    
    # Verify performance metrics
    assert results['total_return'] > 0
    assert results['sharpe_ratio'] > 1.0
    assert results['max_drawdown'] < 0.3
