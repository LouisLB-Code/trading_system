# File: tests/core/data/test_data_integration.py

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from src.core.data.data_manager import DataManager
from src.core.data.data_storage import DataStorage
from src.core.data.data_processor import DataProcessor
from config.data_config import DataSystemConfig

class TestDataIntegration:
    @pytest.fixture
    async def setup(self):
        """Setup test environment"""
        config = DataSystemConfig()
        config.data.DATA_PATH = "tests/data"  # Use test data directory
        
        data_manager = DataManager(config)
        await data_manager.initialize()
        
        yield data_manager
        
        # Cleanup
        await data_manager.storage.clean_old_data(0)  # Remove all test data
    
    @pytest.mark.asyncio
    async def test_data_fetch_and_storage(self, setup):
        """Test data fetching and storage integration"""
        data_manager = setup
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        # Fetch data
        data = await data_manager.get_market_data(symbol, timeframe)
        
        # Verify data structure
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert data.index.is_monotonic_increasing  # Check timestamps are ordered
        
        # Verify data is cached
        cache_key = f"{symbol}_{timeframe}"
        assert cache_key in data_manager.cache
        
        # Verify data is stored
        stored_data = await data_manager.storage.get_market_data(
            symbol,
            timeframe,
            data.index[0],
            data.index[-1]
        )
        assert not stored_data.empty
        pd.testing.assert_frame_equal(data, stored_data)
    
    @pytest.mark.asyncio
    async def test_multiple_timeframes(self, setup):
        """Test fetching multiple timeframes"""
        data_manager = setup
        symbol = "BTC/USDT"
        timeframes = ["1h", "4h", "1d"]
        
        # Fetch multiple timeframes
        results = await data_manager.get_multiple_timeframes(symbol, timeframes)
        
        # Verify results
        assert isinstance(results, dict)
        assert all(tf in results for tf in timeframes)
        assert all(isinstance(df, pd.DataFrame) for df in results.values())
        
        # Verify data alignment
        for tf, df in results.items():
            assert not df.empty
            if tf == "1h":
                assert len(df) >= len(results["4h"])
                assert len(df) >= len(results["1d"])
    
    @pytest.mark.asyncio
    async def test_data_processing(self, setup):
        """Test data processing pipeline"""
        data_manager = setup
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        # Get processed data
        data = await data_manager.get_market_data(symbol, timeframe)
        
        # Verify technical indicators
        expected_columns = [
            'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd',
            'bollinger_upper', 'bollinger_lower', 'volume_sma'
        ]
        assert all(col in data.columns for col in expected_columns)
        
        # Verify no missing values in key columns
        assert not data[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()
    
    @pytest.mark.asyncio
    async def test_cache_management(self, setup):
        """Test cache management and updates"""
        data_manager = setup
        symbol = "BTC/USDT"
        timeframe = "1h"
        
        # Initial fetch
        data1 = await data_manager.get_market_data(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}"
        
        # Immediate fetch should use cache
        data2 = await data_manager.get_market_data(symbol, timeframe)
        assert id(data1) == id(data2)  # Same object in cache
        
        # Modify last_update to force refresh
        data_manager.last_update[cache_key] = datetime.now() - timedelta(hours=1)
        data3 = await data_manager.get_market_data(symbol, timeframe)
        assert id(data2) != id(data3)  # Different object after refresh
    
    @pytest.mark.asyncio
    async def test_error_handling(self, setup):
        """Test error handling in data pipeline"""
        data_manager = setup
        
        # Test with invalid symbol
        with pytest.raises(Exception):
            await data_manager.get_market_data("INVALID/PAIR", "1h")
        
        # Test with invalid timeframe
        with pytest.raises(Exception):
            await data_manager.get_market_data("BTC/USDT", "invalid")
        
        # Test with invalid date range
        future_start = datetime.now() + timedelta(days=365)
        future_end = future_start + timedelta(days=1)
        
        with pytest.raises(Exception):
            await data_manager.storage.get_market_data(
                "BTC/USDT",
                "1h",
                future_start,
                future_end
            )
