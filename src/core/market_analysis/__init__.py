# File: src/core/market_analysis/__init__.py
from .regime_detector import EnhancedRegimeDetector
from .condition_analyzer import MarketConditionAnalyzer
from .regime_transitions import RegimeTransitionManager

# File: src/core/market_analysis/regime_detector.py
# This will be an enhanced version of your existing market_regime.py
# We'll keep the existing MarketRegime class and add the new features

# File: src/core/market_analysis/condition_analyzer.py
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np

@dataclass
class MarketCondition:
    volatility: float
    trend_strength: float
    liquidity: float
    volume_profile: float
    momentum: float
    efficiency_ratio: float
    fractal_dimension: float
    regime_stability: float

class MarketConditionAnalyzer:
    """Analyzes market conditions and characteristics"""
    def __init__(self, config):
        self.config = config
        self.history = []

    async def analyze(self, market_data: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions"""
        try:
            return MarketCondition(
                volatility=self._calculate_volatility(market_data),
                trend_strength=self._calculate_trend_strength(market_data),
                liquidity=self._calculate_liquidity(market_data),
                volume_profile=self._calculate_volume_profile(market_data),
                momentum=self._calculate_momentum(market_data),
                efficiency_ratio=self._calculate_efficiency_ratio(market_data),
                fractal_dimension=self._calculate_fractal_dimension(market_data),
                regime_stability=self._calculate_regime_stability(market_data)
            )
        except Exception as e:
            logging.error(f"Market condition analysis error: {str(e)}")
            raise

# File: src/core/market_analysis/regime_transitions.py
class RegimeTransitionManager:
    """Manages and analyzes regime transitions"""
    def __init__(self, config):
        self.config = config
        self.transition_matrix = pd.DataFrame()
        self.transition_history = []
        
    async def analyze_transition(self,
                               from_regime: str,
                               to_regime: str,
                               market_conditions: MarketCondition) -> Dict:
        """Analyze regime transition"""
        try:
            # Record transition
            self._record_transition(from_regime, to_regime, market_conditions)
            
            # Calculate transition probability
            probability = self._calculate_transition_probability(
                from_regime,
                to_regime
            )
            
            # Analyze transition characteristics
            characteristics = self._analyze_transition_characteristics(
                from_regime,
                to_regime,
                market_conditions
            )
            
            return {
                'probability': probability,
                'characteristics': characteristics,
                'historical_pattern': self._find_similar_transitions(
                    from_regime,
                    to_regime,
                    market_conditions
                )
            }
        except Exception as e:
            logging.error(f"Regime transition analysis error: {str(e)}")
            raise
