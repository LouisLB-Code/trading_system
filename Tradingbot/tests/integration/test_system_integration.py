# tests/integration/test_system_integration.py
import pytest
import pandas as pd
from datetime import datetime

class TestSystemIntegration:
    @pytest.mark.integration
    async def test_full_trading_cycle(self, trading_strategy, model_manager, data_collector):
        """Test a complete trading cycle from data collection to order generation"""
        # 1. Collect market data
        market_data = await data_collector.fetch_historical_data('BTC/USDT', '1h', limit=100)
        assert isinstance(market_data, pd.DataFrame)
        
        # 2. Generate predictions
        predictions = await model_manager.predict('BTC/USDT', market_data)
        assert len(predictions) > 0
        
        # 3. Generate trading signals
        signals = await trading_strategy.analyze_market({'BTC/USDT': market_data})
        assert 'BTC/USDT' in signals
        
        # 4. Generate orders
        orders = await trading_strategy.generate_orders(signals, {'BTC/USDT': market_data})
        assert isinstance(orders, list)
