
from dataclasses import dataclass
from typing import Dict

@dataclass
class ExecutionConfig:
    """Execution strategy configuration parameters"""
    
    # General parameters
    MAX_SLIPPAGE: float = 0.001    # Maximum acceptable slippage (0.1%)
    MIN_TRADE_SIZE: float = 0.001  # Minimum trade size
    MAX_RETRY_ATTEMPTS: int = 3    # Maximum retry attempts per execution
    
    # TWAP parameters
    TWAP_DURATION: int = 60        # Total execution duration in minutes
    TWAP_INTERVALS: int = 12       # Number of execution intervals
    TWAP_MAX_DEVIATION: float = 0.01  # Maximum price deviation allowed (1%)
    
    # VWAP parameters
    VWAP_LOOKBACK: int = 100      # Number of periods for VWAP calculation
    VWAP_INTERVAL: int = 5        # Minutes per interval
    VWAP_MIN_VOLUME: float = 1000 # Minimum volume threshold
    
    # POV parameters
    POV_MAX_PARTICIPATION: float = 0.1  # Maximum participation rate (10%)
    POV_INTERVAL: int = 1              # Minutes between volume checks
    POV_VOLUME_WINDOW: int = 15        # Minutes for volume calculation
    
    # Iceberg parameters
    ICEBERG_DISPLAY_SIZE: float = 0.1  # Display size as fraction of total (10%)
    ICEBERG_MIN_SIZE: float = 0.01     # Minimum slice size (1%)
    ICEBERG_RANDOM_SIZE: bool = True   # Randomize slice sizes
    
    # Direct execution parameters
    DIRECT_MAX_SIZE: float = 1.0      # Maximum size for direct execution
    DIRECT_TIMEOUT: int = 30          # Timeout for direct orders (seconds)
    
    # Rate limiting
    RATE_LIMIT_ORDERS: int = 10       # Maximum orders per second
    RATE_LIMIT_WINDOW: int = 1        # Window for rate limiting (seconds)

class ExecutionRateLimit:
    """Rate limiting configuration for exchanges"""
    
    def __init__(self):
        self.EXCHANGE_LIMITS = {
            'binance': {
                'orders_per_second': 10,
                'orders_per_minute': 1200,
                'weight_per_request': 1
            },
            'ftx': {
                'orders_per_second': 8,
                'orders_per_minute': 960,
                'weight_per_request': 1
            },
            'kraken': {
                'orders_per_second': 5,
                'orders_per_minute': 600,
                'weight_per_request': 1
            }
        }

class ExecutionRiskLimits:
    """Risk limits for execution strategies"""
    
    def __init__(self):
        self.STRATEGY_LIMITS = {
            'direct': {
                'max_order_size': 1.0,
                'max_notional': 100000,
                'max_slippage': 0.001
            },
            'twap': {
                'max_order_size': 5.0,
                'max_notional': 500000,
                'min_interval': 60  # seconds
            },
            'vwap': {
                'max_order_size': 5.0,
                'max_notional': 500000,
                'min_volume': 1000
            },
            'pov': {
                'max_participation': 0.1,
                'max_notional': 500000,
                'min_market_vol': 10000
            },
            'iceberg': {
                'max_order_size': 10.0,
                'max_notional': 1000000,
                'min_slice_interval': 30  # seconds
            }
        }

class ExecutionSystemConfig:
    """Main execution system configuration"""
    
    def __init__(self):
        self.execution = ExecutionConfig()
        self.rate_limits = ExecutionRateLimit()
        self.risk_limits = ExecutionRiskLimits()