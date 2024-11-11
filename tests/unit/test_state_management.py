# tests/unit/test_state_management.py
import pytest
from datetime import datetime

def test_system_state_transitions(state_manager):
    """Test system state transitions"""
    # Test state transition
    await state_manager.update_system_state(SystemState.INITIALIZING)
    assert state_manager.system_state == SystemState.INITIALIZING
    
    # Test trading state updates
    updates = {
        'active_positions': {'BTC/USDT': {'size': 1.0, 'entry_price': 50000}},
        'pending_orders': []
    }
    await state_manager.update_trading_state(updates)
    assert state_manager.trading_state.active_positions == updates['active_positions']

def test_market_state_updates(state_manager):
    """Test market state updates"""
    market_updates = {
        'current_regime': 'TRENDING',
        'regime_confidence': 0.85
    }
    await state_manager.update_market_state(market_updates)
    assert state_manager.market_state.current_regime == 'TRENDING'
