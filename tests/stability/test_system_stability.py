# tests/stability/test_system_stability.py
import pytest
from unittest.mock import patch
import numpy as np
from src.core.trading_system import TradingSystem
from src.exceptions import MarketDataError, ExecutionError

class TestSystemStability:
    @pytest.fixture
    def trading_system(self):
        config = TestConfig()
        return TradingSystem(config)

    def test_error_recovery(self, trading_system):
        """Test system recovery after errors"""
        with patch('src.core.market_data.fetch_data') as mock_fetch:
            mock_fetch.side_effect = [
                ConnectionError("Network error"),
                {'price': 100, 'volume': 1000}
            ]
            
            result = trading_system.process_market_update()
            assert result['recovered']
            assert result['retry_count'] == 1
            assert result['final_state'] == 'operational'
    
    def test_data_consistency(self, trading_system):
        """Test data consistency during updates"""
        initial_state = trading_system.get_state()
        
        # Simulate market updates with random data
        updates = [
            {
                'price': 100 + np.random.normal(0, 1),
                'volume': 1000 + np.random.normal(0, 100)
            }
            for _ in range(100)
        ]
        
        for update in updates:
            trading_system.process_update(update)
            
        final_state = trading_system.get_state()
        
        # Verify state consistency
        assert final_state['last_price'] > 0
        assert final_state['total_volume'] >= initial_state['total_volume']
        assert final_state['timestamp'] > initial_state['timestamp']
        
    @pytest.mark.asyncio
    async def test_recovery_from_execution_error(self, trading_system):
        """Test recovery from order execution errors"""
        with patch('src.core.exchange.execute_order') as mock_execute:
            mock_execute.side_effect = [
                ExecutionError("Rate limit exceeded"),
                {'order_id': '123', 'status': 'executed'}
            ]
            
            order = {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 0.1}
            result = await trading_system.execute_order(order)
            
            assert result['status'] == 'executed'
            assert result['recovery_attempts'] == 1
            
    def test_state_persistence(self, trading_system):
        """Test system state persistence during errors"""
        # Setup initial state
        trading_system.update_state({
            'positions': {'BTC/USDT': 0.1},
            'balance': 10000
        })
        
        # Simulate system crash
        with pytest.raises(SystemError):
            trading_system.simulate_crash()
            
        # Verify state after recovery
        recovered_state = trading_system.get_state()
        assert recovered_state['positions']['BTC/USDT'] == 0.1
        assert recovered_state['balance'] == 10000
