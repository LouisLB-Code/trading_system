# tests/performance/test_system_performance.py
import time
import pytest
from src.utils.performance_metrics import calculate_metrics

class TestSystemPerformance:
    def test_execution_speed(self, trading_system):
        """Test strategy execution speed"""
        market_data = create_test_market_data()
        
        start_time = time.time()
        signals = trading_system.generate_signals(market_data)
        execution_time = time.time() - start_time
        
        assert execution_time < 0.1  # Should execute in under 100ms
    
    def test_strategy_metrics(self, trading_system):
        """Test strategy performance metrics"""
        results = trading_system.run_backtest(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        metrics = calculate_metrics(results)
        assert metrics['sharpe_ratio'] > 1.5
        assert metrics['sortino_ratio'] > 1.0
        assert metrics['max_drawdown'] < 0.2
        assert metrics['win_rate'] > 0.6
