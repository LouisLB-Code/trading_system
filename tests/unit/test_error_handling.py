# tests/unit/test_error_handling.py
import pytest
from src.core.trading_system import TradingSystem
from src.core.market_analysis import MarketConditionAnalyzer
from src.exceptions import (
    MarketDataError,
    OrderExecutionError,
    StrategyError
)

class TestErrorHandling:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        self.config = TestConfig()
        self.trading_system = TradingSystem(self.config)
        self.market_analyzer = MarketConditionAnalyzer(self.config)
    
    def test_invalid_market_data_handling(self, setup):
        """Test handling of invalid market data"""
        # Test with None data
        with pytest.raises(MarketDataError) as exc_info:
            self.market_analyzer.analyze_market_conditions(None)
        assert "Invalid market data" in str(exc_info.value)
        
        # Test with empty DataFrame
        with pytest.raises(MarketDataError) as exc_info:
            self.market_analyzer.analyze_market_conditions(pd.DataFrame())
        assert "Empty market data" in str(exc_info.value)
        
        # Test with missing columns
        invalid_data = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(MarketDataError) as exc_info:
            self.market_analyzer.analyze_market_conditions(invalid_data)
        assert "Missing required columns" in str(exc_info.value)
    
    def test_order_execution_errors(self, setup):
        """Test handling of order execution errors"""
        # Test insufficient balance
        with pytest.raises(OrderExecutionError) as exc_info:
            self.trading_system.execute_order({
                'type': 'market',
                'side': 'buy',
                'amount': 999999999,  # Unrealistic amount
                'symbol': 'BTC/USDT'
            })
        assert "Insufficient balance" in str(exc_info.value)
        
        # Test invalid order parameters
        with pytest.raises(OrderExecutionError) as exc_info:
            self.trading_system.execute_order({
                'type': 'invalid_type',
                'side': 'buy',
                'amount': 1,
                'symbol': 'BTC/USDT'
            })
        assert "Invalid order type" in str(exc_info.value)
    
    def test_strategy_error_handling(self, setup):
        """Test handling of strategy execution errors"""
        # Test strategy timeout
        with pytest.raises(StrategyError) as exc_info:
            self.trading_system.execute_strategy(
                timeout=0.001  # Unrealistic timeout
            )
        assert "Strategy execution timeout" in str(exc_info.value)
        
        # Test strategy state errors
        self.trading_system.force_state('ERROR')
        with pytest.raises(StrategyError) as exc_info:
            self.trading_system.execute_strategy()
        assert "Invalid system state" in str(exc_info.value)
    
    def test_recovery_mechanisms(self, setup):
        """Test system recovery mechanisms"""
        # Test recovery after network error
        self.trading_system.handle_error("NETWORK_ERROR")
        assert self.trading_system.state == "RECOVERY"
        assert self.trading_system.is_trading_halted
        
        # Test auto-recovery
        self.trading_system.attempt_recovery()
        assert self.trading_system.state == "OPERATIONAL"
        assert not self.trading_system.is_trading_halted
