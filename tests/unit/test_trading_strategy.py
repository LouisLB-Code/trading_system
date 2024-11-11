import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.strategy.trading_strategy import TradingStrategy
from src.models.model_manager import ModelManager
from src.data.data_collector import DataCollector
from config.config import TestConfig

class TestTradingSystem:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        self.config = TestConfig()
        self.strategy = TradingStrategy(self.config)
        self.model_manager = ModelManager(self.config)
        self.data_collector = DataCollector(self.config)
        
    def test_data_collection(self, setup):
        """Test data collection functionality"""
        # Test historical data collection
        data = self.data_collector.fetch_historical_data(
            'BTC/USDT',
            '1h',
            limit=100
        )
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
    def test_model_predictions(self, setup):
        """Test model predictions"""
        # Prepare test data
        test_data = self._prepare_test_data()
        
        # Get predictions
        predictions = self.model_manager.predict('BTC/USDT', test_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) > 0
        
    def test_strategy_signals(self, setup):
        """Test strategy signal generation"""
        # Prepare test market data
        market_data = self._prepare_market_data()
        
        # Generate signals
        signals = self.strategy.analyze_market(market_data)
        
        assert isinstance(signals, dict)
        assert 'BTC/USDT' in signals
        assert 'signal' in signals['BTC/USDT']
        
    def test_order_generation(self, setup):
        """Test order generation"""
        # Prepare test signals
        signals = self._prepare_test_signals()
        market_data = self._prepare_market_data()
        
        # Generate orders
        orders = self.strategy.generate_orders(signals, market_data)
        
        assert isinstance(orders, list)
        if orders:
            assert all(key in orders[0] for key in [
                'pair', 'direction', 'size', 'entry_price', 
                'stop_loss', 'take_profit'
            ])
    
    def _prepare_test_data(self):
        """Prepare test dataset"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data
    
    def _prepare_market_data(self):
        """Prepare test market data"""
        return {
            'BTC/USDT': {
                'ohlcv': self._prepare_test_data(),
                'indicators': self._prepare_test_indicators(),
                'orderbook': self._prepare_test_orderbook()
            }
        }
    
    def _prepare_test_signals(self):
        """Prepare test signals"""
        return {
            'BTC/USDT': {
                'signal': 0.75,
                'technical': {'trend': 0.8, 'momentum': 0.7},
                'ml': 0.65,
                'timestamp': datetime.now()
            }
        }
