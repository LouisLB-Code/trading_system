# File: config/model_config.py

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Training parameters
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    
    # Risk model parameters
    RISK_INPUT_FEATURES: int = 64
    RISK_OUTPUT_FEATURES: int = 4
    RISK_HIDDEN_SIZE: int = 128
    
    # Pattern model parameters
    PATTERN_INPUT_CHANNELS: int = 5
    PATTERN_OUTPUT_FEATURES: int = 32
    PATTERN_SEQUENCE_LENGTH: int = 100
    
    # Regime model parameters
    REGIME_INPUT_FEATURES: int = 32
    REGIME_OUTPUT_FEATURES: int = 8
    REGIME_HIDDEN_SIZE: int = 128
    
    # Strategy model parameters
    STRATEGY_INPUT_FEATURES: int = 128
    STRATEGY_OUTPUT_FEATURES: int = 64
    STRATEGY_HIDDEN_SIZE: int = 256
    
    # Training thresholds
    MIN_ACCURACY: float = 0.7
    MIN_SHARPE: float = 1.0
    MAX_DRAWDOWN: float = 0.2
    
    # Model paths
    MODEL_SAVE_PATH: str = "models/"
    CHECKPOINT_PATH: str = "models/checkpoints/"
    
    # Update frequency
    UPDATE_INTERVAL: int = 1000  # Updates every 1000 samples
    RETRAIN_INTERVAL: int = 24   # Hours between retraining

class ModelsConfig:
    """Configuration for all models"""
    
    def __init__(self):
        self.model = ModelConfig()
        
        # Feature configurations
        self.PRICE_FEATURES = [
            'open', 'high', 'low', 'close', 'volume',
            'vwap', 'spread', 'depth'
        ]
        
        self.TECHNICAL_FEATURES = [
            'sma', 'ema', 'rsi', 'macd', 'bbands',
            'atr', 'adx', 'obv', 'momentum'
        ]
        
        self.MARKET_FEATURES = [
            'volatility', 'liquidity', 'trend_strength',
            'regime', 'correlation'
        ]
        
        # Model architectures
        self.ARCHITECTURES = {
            'risk': 'lstm',
            'pattern': 'cnn',
            'regime': 'transformer',
            'strategy': 'mlp'
        }
