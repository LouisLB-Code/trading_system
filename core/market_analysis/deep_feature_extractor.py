
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class DeepFeatures:
    temporal_patterns: np.ndarray
    market_structure: np.ndarray
    regime_indicators: np.ndarray
    attention_weights: np.ndarray

class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return attended, attention_weights

class DeepFeatureExtractor(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Feature extraction layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.attention = TemporalAttention(64, 32)
        
        # Market structure analysis
        self.structure_analyzer = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Regime indicators
        self.regime_detector = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
    def forward(self, x: torch.Tensor) -> DeepFeatures:
        # Extract temporal patterns
        temporal_features = self.temporal_conv(x.transpose(1, 2))
        attended, weights = self.attention(temporal_features.transpose(1, 2))
        
        # Analyze market structure
        structure_features, _ = self.structure_analyzer(temporal_features.transpose(1, 2))
        
        # Generate regime indicators
        regime_features = self.regime_detector(attended)
        
        return DeepFeatures(
            temporal_patterns=attended.detach().numpy(),
            market_structure=structure_features[:, -1, :].detach().numpy(),
            regime_indicators=regime_features.detach().numpy(),
            attention_weights=weights.detach().numpy()
        )

class EnhancedFeatureExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepFeatureExtractor(config).to(self.device)
        self.scaler = StandardScaler()
        
    def extract_features(self, market_data: pd.DataFrame) -> DeepFeatures:
        """Extract deep features from market data"""
        try:
            # Prepare input features
            features = self._prepare_features(market_data)
            
            # Convert to tensor
            x = torch.FloatTensor(features).to(self.device)
            
            # Extract deep features
            with torch.no_grad():
                deep_features = self.model(x)
            
            return deep_features
            
        except Exception as e:
            logging.error(f"Deep feature extraction error: {str(e)}")
            raise
            
    def _prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare input features for deep learning model"""
        try:
            # Basic features
            features = np.column_stack([
                self._calculate_returns(market_data),
                self._calculate_volatility_features(market_data),
                self._calculate_volume_features(market_data),
                self._calculate_technical_features(market_data)
            ])
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features.reshape(1, -1, 8)  # Reshape for conv1d
            
        except Exception as e:
            logging.error(f"Feature preparation error: {str(e)}")
            raise

def _calculate_returns(self, market_data: pd.DataFrame) -> np.ndarray:
    """Calculate return-based features"""
    returns = market_data['close'].pct_change().fillna(0)
    return np.column_stack([
        returns,
        returns.rolling(5).mean(),
        returns.rolling(5).std()
    ])

def _calculate_volatility_features(self, market_data: pd.DataFrame) -> np.ndarray:
    """Calculate volatility-based features"""
    log_returns = np.log(market_data['close'] / market_data['close'].shift(1)).fillna(0)
    return np.column_stack([
        log_returns.rolling(10).std(),
        np.abs(market_data['high'] - market_data['low']) / market_data['close']
    ])

def _calculate_volume_features(self, market_data: pd.DataFrame) -> np.ndarray:
    """Calculate volume-based features"""
    volume = market_data['volume']
    return np.column_stack([
        volume / volume.rolling(10).mean(),
        (volume * market_data['close']).rolling(5).mean()
    ])

def _calculate_technical_features(self, market_data: pd.DataFrame) -> np.ndarray:
    """Calculate technical indicators"""
    close = market_data['close']
    return np.column_stack([
        close.rolling(14).mean() / close - 1,  # MA ratio
        (close - close.rolling(14).min()) / (close.rolling(14).max() - close.rolling(14).min())  # Stochastic
    ])
