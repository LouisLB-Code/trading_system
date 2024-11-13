# File: config/config.py

from dataclasses import dataclass
import os
from typing import List, Dict

@dataclass
class MarketAnalysisConfig:
    # Clustering parameters
    CLUSTERING_EPS: float = 0.3
    CLUSTERING_MIN_SAMPLES: int = 2
    MAX_REGIME_HISTORY: int = 1000
    
    # Market condition thresholds
    LIQUIDITY_NORM_FACTOR: float = 1000000  # Normalize liquidity values
    VOLATILITY_THRESHOLD: float = 0.7
    TREND_STRENGTH_THRESHOLD: float = 0.6
    EFFICIENCY_THRESHOLD: float = 0.3
    
    # Regime transition parameters
    SIMILARITY_THRESHOLD: float = 0.8
    MIN_TRANSITION_SAMPLES: int = 5
    
    # Strategy parameters
    MAX_ACTIVE_STRATEGIES: int = 3
    STRATEGY_SELECTION_THRESHOLD: float = 0.6
    
    # Risk management parameters
    BASE_POSITION_SIZE: float = 1.0
    MAX_LEVERAGE: float = 2.0
    BASE_STOP_LOSS: float = 0.02
    MAX_RISK_FACTOR: float = 2.0

class Config:
    """Main configuration class"""
    def __init__(self):
        # Load environment variables if needed
        self._load_env_vars()
        
        # Initialize market analysis config
        self.market_analysis = MarketAnalysisConfig()
        
        # Trading parameters
        self.TRADING_PAIRS: List[str] = ['BTC/USDT', 'ETH/USDT']
        self.TIMEFRAMES: List[str] = ['1h', '4h', '1d']
        self.INITIAL_CAPITAL: float = float(os.getenv('INITIAL_CAPITAL', '10000'))
        
        # API Configuration
        self.BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
        self.BINANCE_API_SECRET: str = os.getenv('BINANCE_API_SECRET', '')
        
        # System paths
        self.BASE_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_PATH: str = os.path.join(self.BASE_PATH, 'data')
        self.MODELS_PATH: str = os.path.join(self.BASE_PATH, 'models')
        self.LOGS_PATH: str = os.path.join(self.BASE_PATH, 'logs')
        
        # Ensure paths exist
        self._ensure_paths_exist()
    
    def _load_env_vars(self):
        """Load environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
    
    def _ensure_paths_exist(self):
        """Ensure required directories exist"""
        paths = [self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH]
        for path in paths:
            os.makedirs(path, exist_ok=True)

# Create test and production configurations
class TestConfig(Config):
    """Test configuration"""
    def __init__(self):
        super().__init__()
        self.IS_TEST = True
        self.USE_TEST_NET = True

class ProductionConfig(Config):
    """Production configuration"""
    def __init__(self):
        super().__init__()
        self.IS_TEST = False
        self.USE_TEST_NET = False
        
        # Validate production requirements
        self._validate_production_config()
    
    def _validate_production_config(self):
        """Validate production configuration"""
        if not all([self.BINANCE_API_KEY, self.BINANCE_API_SECRET]):
            raise ValueError("API keys must be set for production")
