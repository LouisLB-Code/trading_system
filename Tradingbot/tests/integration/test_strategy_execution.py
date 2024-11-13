# tests/integration/test_strategy_execution.py
import pytest
import pandas as pd

def test_strategy_signal_to_execution(trading_system, state_manager):
    """Test complete flow from signal generation to execution"""
    # Setup test market data
    market_data = create_test_market_data()
    
    # Generate signals
    signals = await trading_system.strategy_manager.analyze(market_data)
    assert signals is not None
    
    # Process signals and generate orders
    orders = await trading_system.process_signals(signals)
    assert len(orders) > 0
    
    # Verify orders match risk parameters
    for order in orders:
        assert order['size'] <= trading_system.config.MAX_POSITION_SIZE
        assert order['stop_loss'] is not None
