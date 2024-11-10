# File: tests/core/data/test_data_mock.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.core.data.data_manager import DataManager, BinanceDataSource
from config.data_config import DataSystemConfig

class TestDataMock:
    @pytest.fixture
    def mock_response(self):
        """Create mock response data"""
        def create_kline_data(start_time: datetime, periods: int = 100):
            data = []
            for i in range(periods):
                timestamp = start_time + timedelta(hours=i)
                kline = [
                    int(timestamp.timestamp() * 1000),  # Open time
                    "50000.0",  # Open
                    "51000.0",  # High
                    "49000.0",  # Low
                    "50500.0",  # Close
                    "100.0",    # Volume
                    int((timestamp + timedelta(hours=1)).timestamp() * 1000),  # Close time
                    "5000000.0",  # Quote volume
                    100,        # Number of trades
                    "50.0",     # Taker buy base volume
                    "2500000.0",  # Taker buy quote volume
                    "0"         # Ignore
                ]
                data.append(kline)
            return data
            
        return create_kline_data

    @pytest.fixture
    def mock_binance_source(self, mock_response):
        """Create mock Binance data source"""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock the response
            mock_resp = Mock()
            mock_resp.status = 200
            start_time = datetime.now() - timedelta(days=7)
            mock_resp.json = Mock(return_value=mock_response(start_time))
            
            # Mock the context manager
            mock_context = Mock()
            mock_context.__aenter__ = Mock(return_value=mock_resp)
            mock_context.__aexit__ = Mock(return_value=None)
            
            # Mock the get method
            mock_session.return_value.get = Mock(return_value=mock_context)
            
            yield BinanceDataSource(DataSystemConfig())

    @pytest.mark.asyncio
    async def test_fetch_historical_data(self, mock_binance_source):
        """Test fetching historical data with mock"""
        symbol = "BTC/USDT"
        timeframe = "1h"
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        data = await mock_binance_source.fetch_data(
            symbol,
            timeframe,
            start_time,
            end_time
        )
        
        # Verify data structure
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert data.index.is_monotonic_increasing

    @pytest.mark.asyncio
    async def test_data_manager_with_mock(self, mock_binance_source):
        """Test DataManager with mock data source"""
        config = DataSystemConfig()
        config.data.DATA_PATH = "tests/data"
        
        data_manager = DataManager(config)
        data_manager.data_sources = {'binance': mock_binance_source}
        
        await data_manager.initialize()
        
        # Test data fetching
        data = await data_manager.get_market_data("BTC/USDT", "1h")
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        
        # Clean up
        await data_manager.storage.clean_old_data(0)

    @pytest.mark.asyncio
    async def test_error_handling_with_mock(self, mock_binance_source):
        """Test error handling with mock failures"""
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock error response
            mock_resp = Mock()
            mock_resp.status = 400
            mock_resp.text = Mock(return_value="Invalid symbol")
            
            mock_context = Mock()
            mock_context.__aenter__ = Mock(return_value=mock_resp)
            mock_context.__aexit__ = Mock(return_value=None)
            
            mock_session.return_value.get = Mock(return_value=mock_context)
            
            with pytest.raises(Exception):
                await mock_binance_source.fetch_data(
                    "INVALID/PAIR",
                    "1h",
                    datetime.now() - timedelta(days=1),
                    datetime.now()
                )

    @pytest.mark.asyncio
    async def test_rate_limiting_with_mock(self, mock_binance_source, mock_response):
        """Test rate limiting behavior"""
        config = DataSystemConfig()
        config.data.BINANCE_RATE_LIMIT = 2  # Set very low for testing
        
        data_manager = DataManager(config)
        data_manager.data_sources = {'binance': mock_binance_source}
        
        # Test multiple rapid requests
        symbol = "BTC/USDT"
        timeframes = ["1m", "5m", "15m", "30m", "1h"]
        
        tasks = [
            data_manager.get_market_data(symbol, tf)
            for tf in timeframes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed
        assert len(results) == len(timeframes)
        assert all(isinstance(r, pd.DataFrame) for r in results)
