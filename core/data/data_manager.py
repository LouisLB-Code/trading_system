import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from .data_storage import DataStorage
from .data_processor import DataProcessor

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def fetch_data(self,
                        symbol: str,
                        timeframe: str,
                        start_time: datetime,
                        end_time: datetime) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def fetch_real_time(self, symbol: str, timeframe: str) -> pd.DataFrame:
        pass

class BinanceDataSource(DataSource):
    """Binance data source implementation"""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        self.base_url = "https://api.binance.com/api/v3"
        
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def fetch_data(self,
                        symbol: str,
                        timeframe: str,
                        start_time: datetime,
                        end_time: datetime) -> pd.DataFrame:
        """Fetch historical data from Binance"""
        try:
            await self.initialize()
            
            # Convert timeframe to milliseconds
            interval = self._convert_timeframe(timeframe)
            
            # Prepare request parameters
            params = {
                'symbol': symbol.replace('/', ''),
                'interval': interval,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'limit': self.config.data.BINANCE_MAX_LIMIT
            }
            
            # Make API request
            async with self.session.get(f"{self.base_url}/klines", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_kline_data(data)
                else:
                    raise Exception(f"API request failed: {await response.text()}")
                    
        except Exception as e:
            logging.error(f"Data fetch error: {str(e)}")
            raise

    async def fetch_real_time(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch real-time market data from Binance"""
        try:
            await self.initialize()
            
            # Prepare request parameters
            params = {
                'symbol': symbol.replace('/', ''),
                'limit': 1  # Get only latest data
            }
            
            # Make API request
            async with self.session.get(f"{self.base_url}/ticker", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_ticker_data(data)
                else:
                    raise Exception(f"API request failed: {await response.text()}")
                    
        except Exception as e:
            logging.error(f"Real-time data fetch error: {str(e)}")
            raise
            
    def _process_ticker_data(self, data: Dict) -> pd.DataFrame:
        """Process real-time ticker data"""
        try:
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(data['closeTime'], unit='ms'),
                'open': float(data['openPrice']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'close': float(data['lastPrice']),
                'volume': float(data['volume'])
            }])
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Ticker data processing error: {str(e)}")
            raise

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe format to Binance format"""
        try:
            timeframe_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            return timeframe_map.get(timeframe, '1h')
            
        except Exception as e:
            logging.error(f"Timeframe conversion error: {str(e)}")
            raise

    def _process_kline_data(self, data: List[List]) -> pd.DataFrame:
        """Process raw kline data from Binance"""
        try:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 
                      'volume', 'close_time', 'quote_volume', 'trades',
                      'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
            
            df = pd.DataFrame(data, columns=columns)
            
            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price and volume columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Keep only necessary columns
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logging.error(f"Kline data processing error: {str(e)}")
            raise

class DataManager:
    """Manages market data collection and storage"""
    
    def __init__(self, config):
        self.config = config
        self.data_sources = self._initialize_data_sources()
        self.storage = DataStorage(config)
        self.processor = DataProcessor(config)
        self.cache = {}
        self.last_update = {}
        self.cleanup_task = None
        
    async def initialize(self):
        """Initialize data manager components"""
        try:
            # Initialize storage
            await self.storage.initialize()
            
            # Initialize data sources
            for source in self.data_sources.values():
                if hasattr(source, 'initialize'):
                    await source.initialize()

            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
                    
        except Exception as e:
            logging.error(f"Data manager initialization error: {str(e)}")
            raise
    
    async def get_market_data(self,
                            symbol: str,
                            timeframe: str,
                            lookback_periods: int = 1000) -> pd.DataFrame:
        """Get market data, with storage and cache integration"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            # Check cache first
            if not self._needs_update(cache_key, timeframe):
                return self.cache[cache_key]
            
            # Calculate time range
            end_time = datetime.now()
            start_time = self._calculate_start_time(
                timeframe,
                lookback_periods
            )
            
            # Try to get from storage first
            stored_data = await self.storage.get_market_data(
                symbol,
                timeframe,
                start_time,
                end_time
            )
            
            # If we have recent enough data in storage, use it
            if not stored_data.empty and self._is_data_recent_enough(stored_data, timeframe):
                self.cache[cache_key] = stored_data
                self.last_update[cache_key] = datetime.now()
                return stored_data
            
            # Fetch new data from exchange
            new_data = await self._fetch_and_store_data(
                symbol,
                timeframe,
                start_time,
                end_time
            )
            
            # Process the data
            processed_data = await self.processor.process_market_data(new_data)
            
            # Update cache
            self.cache[cache_key] = processed_data
            self.last_update[cache_key] = datetime.now()
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Market data retrieval error: {str(e)}")
            raise
    
    async def get_multiple_timeframes(self,
                                    symbol: str,
                                    timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        try:
            results = {}
            tasks = []
            
            for timeframe in timeframes:
                task = self.get_market_data(symbol, timeframe)
                tasks.append(task)
            
            # Fetch all timeframes concurrently
            data_frames = await asyncio.gather(*tasks)
            
            # Combine results
            for timeframe, df in zip(timeframes, data_frames):
                results[timeframe] = df
            
            return results
            
        except Exception as e:
            logging.error(f"Multiple timeframe data retrieval error: {str(e)}")
            raise

    async def _fetch_and_store_data(self,
                                  symbol: str,
                                  timeframe: str,
                                  start_time: datetime,
                                  end_time: datetime) -> pd.DataFrame:
        """Fetch new data and store it"""
        try:
            # Fetch from exchange
            data_source = self.data_sources['binance']
            new_data = await data_source.fetch_data(
                symbol,
                timeframe,
                start_time,
                end_time
            )
            
            # Store in database
            await self.storage.store_market_data(
                symbol,
                timeframe,
                new_data
            )
            
            return new_data
            
        except Exception as e:
            logging.error(f"Data fetch and store error: {str(e)}")
            raise

    async def _periodic_cleanup(self):
        """Periodically clean up old data"""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                await self.storage.clean_old_data(
                    self.config.data.CLEANUP_INTERVAL_DAYS
                )
            except Exception as e:
                logging.error(f"Periodic cleanup error: {str(e)}")
                await asyncio.sleep(3600)  # Wait an hour before retrying
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize different data sources"""
        return {
            'binance': BinanceDataSource(self.config)
        }
    
    def _needs_update(self, cache_key: str, timeframe: str) -> bool:
        """Check if cache needs updating"""
        if cache_key not in self.cache:
            return True
            
        last_update = self.last_update.get(cache_key)
        if not last_update:
            return True
            
        # Get threshold from config
        threshold_minutes = self.config.data.CACHE_UPDATE_THRESHOLDS.get(
            timeframe,
            self.config.data.CACHE_UPDATE_THRESHOLDS['1h']
        )
        
        time_diff = datetime.now() - last_update
        return time_diff.total_seconds() / 60 > threshold_minutes
    
    def _calculate_start_time(self,
                            timeframe: str,
                            lookback_periods: int) -> datetime:
        """Calculate start time based on timeframe and lookback periods"""
        multiplier = self._get_timeframe_multiplier(timeframe)
        delta = timedelta(minutes=multiplier * lookback_periods)
        return datetime.now() - delta
    
    def _get_timeframe_multiplier(self, timeframe: str) -> int:
        """Get minute multiplier for timeframe"""
        multipliers = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return multipliers.get(timeframe, 60)

    def _is_data_recent_enough(self, data: pd.DataFrame, timeframe: str) -> bool:
        """Check if stored data is recent enough to use"""
        try:
            if data.empty:
                return False
                
            last_timestamp = data.index[-1]
            time_diff = datetime.now() - last_timestamp
            
            # Get threshold from config
            threshold_minutes = self.config.data.CACHE_UPDATE_THRESHOLDS.get(
                timeframe,
                self.config.data.CACHE_UPDATE_THRESHOLDS['1h']
            )
            
            return time_diff.total_seconds() / 60 <= threshold_minutes
            
        except Exception as e:
            logging.error(f"Data recency check error: {str(e)}")
            return False