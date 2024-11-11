# File: config/data_config.py

from dataclasses import dataclass
from typing import Dict, List
from datetime import timedelta

@dataclass
class DataConfig:
    """Configuration for data management and processing"""
    
    # Storage parameters
    DATA_PATH: str = "data"
    MAX_CACHE_SIZE: int = 1000
    CLEANUP_INTERVAL_DAYS: int = 30
    
    # Data processing parameters
    MAX_FFILL_PERIODS: int = 5
    MAX_INTERPOLATION_PERIODS: int = 3
    OUTLIER_STD_THRESHOLD: float = 3.0
    
    # Timeframe configurations
    SUPPORTED_TIMEFRAMES: List[str] = [
        '1m', '5m', '15m', '30m',
        '1h', '4h', '1d'
    ]
    DEFAULT_TIMEFRAME: str = '1h'
    
    # Exchange-specific parameters
    BINANCE_RATE_LIMIT: int = 1200  # requests per minute
    BINANCE_WEIGHT_PER_REQUEST: int = 1
    BINANCE_MAX_LIMIT: int = 1000
    
    # Cache parameters
    CACHE_UPDATE_THRESHOLDS: Dict[str, float] = {
        '1m': 0.5,    # Update after 30 seconds
        '5m': 1.0,    # Update after 1 minute
        '15m': 2.0,   # Update after 2 minutes
        '30m': 5.0,   # Update after 5 minutes
        '1h': 10.0,   # Update after 10 minutes
        '4h': 30.0,   # Update after 30 minutes
        '1d': 120.0   # Update after 2 hours
    }
    
    # Data quality thresholds
    MIN_DATA_POINTS: int = 100
    MAX_GAP_THRESHOLD: float = 1.5
    MIN_VOLUME_THRESHOLD: float = 0.0
    
    # Technical indicator parameters
    TECH_INDICATOR_PERIODS: Dict[str, int] = {
        'sma_fast': 20,
        'sma_slow': 50,
        'ema': 20,
        'rsi': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bbands_period': 20,
        'bbands_std': 2.0
    }

class DataSystemConfig:
    """Main configuration for data subsystem"""
    
    def __init__(self):
        self.data = DataConfig()
        
        # Data source configurations
        self.SOURCES = {
            'binance': {
                'base_url': 'https://api.binance.com/api/v3',
                'rate_limit': self.data.BINANCE_RATE_LIMIT,
                'weight_per_request': self.data.BINANCE_WEIGHT_PER_REQUEST,
                'max_limit': self.data.BINANCE_MAX_LIMIT
            }
        }
        
        # Retry configurations
        self.RETRY_CONFIG = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'exponential_backoff': True
        }
