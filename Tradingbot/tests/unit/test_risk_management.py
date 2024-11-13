# tests/unit/test_risk_management.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.core.risk_management import RiskManager
from config import TestConfig
from tests.utils.test_helpers import create_test_market_data

class TestRiskManagement:
    @pytest.fixture
    def setup(self):
        """Setup risk management tests"""
        self.config = TestConfig()
        self.risk_manager = RiskManager(self.config)
        
    def test_position_sizing(self, setup):
        """Test position sizing calculations"""
        # Test parameters
        capital = 10000
        risk_per_trade = 0.01  # 1% risk per trade
        entry_price = 50000    # BTC price
        stop_loss_price = 49000
        
        position_size = self.risk_manager.calculate_position_size(
            capital=capital,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        # Assertions
        assert isinstance(position_size, float)
        assert position_size > 0
        assert position_size <= capital / entry_price  # Position can't be larger than available capital
        
        # Calculate max loss
        max_loss = (entry_price - stop_loss_price) * position_size
        assert max_loss <= capital * risk_per_trade  # Verify risk limit
    
    def test_risk_limits(self, setup):
        """Test risk limit enforcement"""
        test_orders = [
            {
                'pair': 'BTC/USDT',
                'direction': 'long',
                'size': 0.1,
                'entry_price': 50000,
                'stop_loss': 49000,
                'take_profit': 52000
            },
            {
                'pair': 'ETH/USDT',
                'direction': 'short',
                'size': 1.0,
                'entry_price': 3000,
                'stop_loss': 3150,
                'take_profit': 2700
            }
        ]
        
        portfolio_value = 100000
        
        # Test portfolio exposure limits
        valid_orders = self.risk_manager.validate_orders(
            orders=test_orders,
            portfolio_value=portfolio_value
        )
        
        assert isinstance(valid_orders, list)
        for order in valid_orders:
            exposure = order['size'] * order['entry_price']
            assert exposure <= portfolio_value * self.config.MAX_POSITION_SIZE
    
    def test_stop_loss_validation(self, setup):
        """Test stop loss placement validation"""
        market_data = create_test_market_data()
        
        # Test long position
        valid_stop_long = self.risk_manager.validate_stop_loss(
            direction='long',
            entry_price=50000,
            stop_loss=49000,
            market_data=market_data
        )
        
        # Test short position
        valid_stop_short = self.risk_manager.validate_stop_loss(
            direction='short',
            entry_price=50000,
            stop_loss=51000,
            market_data=market_data
        )
        
        assert isinstance(valid_stop_long, bool)
        assert isinstance(valid_stop_short, bool)
    
    def test_portfolio_risk_metrics(self, setup):
        """Test portfolio risk metrics calculation"""
        positions = [
            {
                'pair': 'BTC/USDT',
                'direction': 'long',
                'size': 0.1,
                'entry_price': 50000,
                'current_price': 51000,
                'stop_loss': 49000
            },
            {
                'pair': 'ETH/USDT',
                'direction': 'short',
                'size': 1.0,
                'entry_price': 3000,
                'current_price': 2900,
                'stop_loss': 3150
            }
        ]
        
        portfolio_value = 100000
        
        risk_metrics = self.risk_manager.calculate_portfolio_risk_metrics(
            positions=positions,
            portfolio_value=portfolio_value
        )
        
        # Verify risk metrics
        assert isinstance(risk_metrics, dict)
        assert 'total_exposure' in risk_metrics
        assert 'total_risk' in risk_metrics
        assert 'position_correlation' in risk_metrics
        assert risk_metrics['total_exposure'] <= portfolio_value
        assert risk_metrics['total_risk'] <= self.config.MAX_PORTFOLIO_RISK
    
    def test_drawdown_limits(self, setup):
        """Test drawdown limit enforcement"""
        initial_capital = 100000
        current_capital = 95000  # 5% drawdown
        
        # Test if trading should continue
        should_continue = self.risk_manager.check_drawdown_limits(
            initial_capital=initial_capital,
            current_capital=current_capital
        )
        
        assert isinstance(should_continue, bool)
        
        # Test with max drawdown exceeded
        current_capital = 80000  # 20% drawdown
        should_stop = self.risk_manager.check_drawdown_limits(
            initial_capital=initial_capital,
            current_capital=current_capital
        )
        
        assert isinstance(should_stop, bool)
        assert not should_stop  # Should return False when max drawdown is exceeded

    def test_risk_adjusted_order_sizing(self, setup):
        """Test risk-adjusted order sizing based on market volatility"""
        market_data = create_test_market_data()
        base_position_size = 0.1
        
        adjusted_size = self.risk_manager.adjust_position_size_for_volatility(
            base_size=base_position_size,
            market_data=market_data,
            lookback_period=20
        )
        
        assert isinstance(adjusted_size, float)
        assert adjusted_size > 0
        assert adjusted_size <= base_position_size * self.config.MAX_POSITION_MULTIPLIER

    def test_portfolio_correlation(self, setup):
        """Test portfolio correlation risk"""
        positions = [
            {'pair': 'BTC/USDT', 'size': 0.1, 'entry_price': 50000},
            {'pair': 'ETH/USDT', 'size': 1.0, 'entry_price': 3000}
        ]
        
        correlation = self.risk_manager.calculate_position_correlation(positions)
        assert -1 <= correlation <= 1
        assert correlation != 0  # Crypto pairs typically have correlation

    def test_risk_limits_extreme_conditions(self, setup):
        """Test risk limits under extreme market conditions"""
        # Test with high volatility
        volatile_market = create_test_market_data(volatility_factor=3.0)
        size = self.risk_manager.adjust_position_size_for_volatility(
            base_size=0.1,
            market_data=volatile_market
        )
        assert size < 0.1  # Should reduce position size in high volatility
