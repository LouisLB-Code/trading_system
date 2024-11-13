# File: src/core/strategy/pattern_analyzer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

@dataclass
class Pattern:
    type: str
    strength: float
    duration: int
    features: Dict[str, float]
    confidence: float

class DeepPatternAnalyzer:
    """Deep learning based pattern analyzer"""
    
    def __init__(self, config):
        self.config = config
        self.pattern_encoder = self._build_pattern_encoder()
        self.pattern_classifier = self._build_pattern_classifier()
        self.scaler = StandardScaler()
        self.pattern_history = []
        
    def _extract_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract advanced features from market data"""
        try:
            features = []
            
            # Price action features
            features.extend(self._extract_price_features(market_data))
            
            # Volatility features
            features.extend(self._extract_volatility_features(market_data))
            
            # Volume features
            features.extend(self._extract_volume_features(market_data))
            
            # Pattern features
            features.extend(self._extract_pattern_features(market_data))
            
            # Structure features
            features.extend(self._extract_structure_features(market_data))
            
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Feature extraction error: {str(e)}")
            raise
    
    def _extract_price_features(self, data: pd.DataFrame) -> List[float]:
        """Extract price action related features"""
        try:
            price_features = []
            
            # Calculate returns at different timeframes
            for period in [1, 5, 10, 20, 50]:
                returns = data['close'].pct_change(period)
                price_features.extend([
                    returns.mean(),
                    returns.std(),
                    returns.skew(),
                    returns.kurt()
                ])
            
            # Calculate price levels
            for period in [10, 20, 50, 100, 200]:
                ma = data['close'].rolling(period).mean()
                price_features.extend([
                    (data['close'].iloc[-1] / ma.iloc[-1]) - 1,
                    ma.diff().iloc[-1] / ma.iloc[-1]
                ])
            
            # Calculate support/resistance levels
            price_features.extend(
                self._calculate_support_resistance(data)
            )
            
            return price_features
            
        except Exception as e:
            logging.error(f"Price feature extraction error: {str(e)}")
            return [0.0] * 50  # Return zeros if error
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> List[float]:
        """Extract volatility related features"""
        try:
            vol_features = []
            
            # Calculate volatility at different timeframes
            for period in [5, 10, 20, 50]:
                returns = data['close'].pct_change()
                volatility = returns.rolling(period).std()
                vol_features.extend([
                    volatility.iloc[-1],
                    volatility.mean(),
                    volatility.max() / volatility.min(),
                    volatility.diff().iloc[-1]
                ])
            
            # Calculate volatility ratios
            for i, period1 in enumerate([5, 10, 20]):
                for period2 in [10, 20, 50][i:]:
                    vol1 = returns.rolling(period1).std().iloc[-1]
                    vol2 = returns.rolling(period2).std().iloc[-1]
                    vol_features.append(vol1 / vol2)
            
            return vol_features
            
        except Exception as e:
            logging.error(f"Volatility feature extraction error: {str(e)}")
            return [0.0] * 20
    
    def _extract_pattern_features(self, data: pd.DataFrame) -> List[float]:
        """Extract candlestick and chart pattern features"""
        try:
            pattern_features = []
            
            # Candlestick patterns
            pattern_features.extend(
                self._identify_candlestick_patterns(data)
            )
            
            # Chart patterns
            pattern_features.extend(
                self._identify_chart_patterns(data)
            )
            
            # Momentum patterns
            pattern_features.extend(
                self._identify_momentum_patterns(data)
            )
            
            # Harmonic patterns
            pattern_features.extend(
                self._identify_harmonic_patterns(data)
            )
            
            return pattern_features
            
        except Exception as e:
            logging.error(f"Pattern feature extraction error: {str(e)}")
            return [0.0] * 30
    
    def _identify_patterns(self, features: np.ndarray) -> List[Pattern]:
        """Identify patterns using deep learning"""
        try:
            # Prepare features
            scaled_features = self.scaler.fit_transform(
                features.reshape(1, -1)
            )
            feature_tensor = torch.FloatTensor(scaled_features)
            
            # Encode patterns
            with torch.no_grad():
                pattern_encoding = self.pattern_encoder(feature_tensor)
            
            # Classify patterns
            pattern_classes = self.pattern_classifier(pattern_encoding)
            
            # Convert to pattern objects
            patterns = self._convert_to_patterns(
                pattern_classes,
                features
            )
            
            return patterns
            
        except Exception as e:
            logging.error(f"Pattern identification error: {str(e)}")
            return []
    
    def _build_pattern_encoder(self) -> nn.Module:
        """Build pattern encoding network"""
        return nn.Sequential(
            nn.Linear(self.config.FEATURE_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
    
    def _build_pattern_classifier(self) -> nn.Module:
        """Build pattern classification network"""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.config.NUM_PATTERN_CLASSES),
            nn.Softmax(dim=1)
        )
    
    def _convert_to_patterns(self,
                           classifications: torch.Tensor,
                           features: np.ndarray) -> List[Pattern]:
        """Convert network outputs to pattern objects"""
        patterns = []
        probs = classifications.numpy()
        
        for i, prob in enumerate(probs[0]):
            if prob > self.config.PATTERN_THRESHOLD:
                pattern = Pattern(
                    type=self.config.PATTERN_TYPES[i],
                    strength=prob,
                    duration=self._estimate_pattern_duration(features),
                    features=self._extract_pattern_specific_features(
                        features,
                        i
                    ),
                    confidence=prob
                )
                patterns.append(pattern)
                
        return patterns
