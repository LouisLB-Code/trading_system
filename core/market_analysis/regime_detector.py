
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from .condition_analyzer import MarketConditionAnalyzer, MarketCondition
from .regime_transitions import RegimeTransitionManager
from .deep_feature_extractor import EnhancedFeatureExtractor, DeepFeatures

@dataclass
class MarketRegime:
    name: str
    confidence: float
    condition: MarketCondition
    optimal_strategies: List[str]
    risk_profile: Dict[str, float]
    timestamp: datetime

class EnhancedRegimeDetector:
    """Enhanced version of your existing MarketRegimeDetector"""
    
    def __init__(self, config):
        self.config = config
        self.condition_analyzer = MarketConditionAnalyzer(config)
        self.transition_manager = RegimeTransitionManager(config)
        self.history = []
        self.regime_clusters = {}
        self.current_regime = None
        self.scaler = StandardScaler()
        self.feature_extractor = EnhancedFeatureExtractor(config)
        
    async def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime using multiple indicators and ML"""
        try:
             # Extract deep features
            deep_features = self.feature_extractor.extract_features(market_data)
            
            # Analyze market conditions
            condition = await self.condition_analyzer.analyze(market_data)
            
            # Combine traditional and deep features
            features = self._combine_features(condition, deep_features)
            
            # Generate feature vector
            features = self._create_feature_vector(condition)
            
            # Detect regime using clustering
            regime_name, confidence = self._cluster_regime(features)
            
            # Get optimal strategies for this regime
            optimal_strategies = self._get_optimal_strategies(regime_name, condition)
            
            # Calculate regime-specific risk profile
            risk_profile = self._calculate_risk_profile(regime_name, condition)
            
            # Create regime object
            regime = MarketRegime(
                name=regime_name,
                confidence=confidence,
                condition=condition,
                optimal_strategies=optimal_strategies,
                risk_profile=risk_profile,
                timestamp=datetime.now()
            )
            
            # If regime has changed, analyze transition
            if self.current_regime and self.current_regime.name != regime.name:
                await self.transition_manager.analyze_transition(
                    self.current_regime.name,
                    regime.name,
                    condition
                )
            
            # Update current regime and history
            self.current_regime = regime
            self._update_regime_history(regime)
            
            return regime
            
        except Exception as e:
            logging.error(f"Regime detection error: {str(e)}")
            raise

def _create_feature_vector(self, condition: MarketCondition) -> np.ndarray:
    """Create feature vector from market conditions"""
    try:
        features = np.array([
            condition.volatility,
            condition.trend_strength,
            condition.liquidity,
            condition.volume_profile,
            condition.momentum,
            condition.efficiency_ratio,
            condition.fractal_dimension,
            condition.regime_stability
        ]).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features
        
    except Exception as e:
        logging.error(f"Feature vector creation error: {str(e)}")
        return np.zeros((1, 8))

def _cluster_regime(self, features: np.ndarray) -> tuple:
    """Detect market regime using DBSCAN clustering"""
    try:
        # Perform clustering
        clustering = DBSCAN(
            eps=self.config.market_analysis.CLUSTERING_EPS,
            min_samples=self.config.market_analysis.CLUSTERING_MIN_SAMPLES
        )
        labels = clustering.fit_predict(features)
        
        # Calculate confidence based on distance to nearest core point
        if hasattr(clustering, 'components_'):
            distances = np.min(
                np.linalg.norm(
                    features - clustering.components_,
                    axis=1
                )
            )
            confidence = 1.0 / (1.0 + distances)
        else:
            confidence = 0.5
        
        # Map cluster to regime name
        regime_name = self._map_cluster_to_regime(
            labels[0],
            features
        )
        
        return regime_name, confidence
        
    except Exception as e:
        logging.error(f"Regime clustering error: {str(e)}")
        return "unknown", 0.0

def _map_cluster_to_regime(self, cluster_label: int, features: np.ndarray) -> str:
    """Map cluster label to regime name based on features"""
    try:
        if cluster_label == -1:  # Noise cluster
            return self._classify_unknown_regime(features)
            
        # Get regime characteristics
        volatility = features[0][0]
        trend_strength = features[0][1]
        efficiency = features[0][5]
        
        # Classify regime based on characteristics
        if volatility > 0.7:
            if trend_strength > 0.6:
                return "volatile_trend"
            return "high_volatility"
            
        if trend_strength > 0.7:
            return "strong_trend"
            
        if efficiency < 0.3:
            return "mean_reverting"
            
        if volatility < 0.3 and trend_strength < 0.3:
            return "low_volatility"
            
        return "normal"
        
    except Exception as e:
        logging.error(f"Regime mapping error: {str(e)}")
        return "unknown"

def _classify_unknown_regime(self, features: np.ndarray) -> str:
    """Classify regime when clustering fails"""
    try:
        # Use simple rules when clustering fails
        volatility = features[0][0]
        trend_strength = features[0][1]
        
        if volatility > 0.8:
            return "crisis"
        if volatility < 0.2:
            return "stable"
        if trend_strength > 0.8:
            return "trending"
            
        return "transitional"
        
    except Exception as e:
        logging.error(f"Unknown regime classification error: {str(e)}")
        return "unknown"

def _get_optimal_strategies(self, regime_name: str, condition: MarketCondition) -> List[str]:
    """Determine optimal strategies for current regime"""
    try:
        strategies = []
        
        # Strategy selection based on regime and conditions
        if regime_name == "volatile_trend":
            strategies.extend(["trend_following", "breakout"])
            if condition.liquidity > 0.7:
                strategies.append("momentum")
                
        elif regime_name == "high_volatility":
            strategies.extend(["volatility_breakout", "mean_reversion"])
            if condition.liquidity > 0.8:
                strategies.append("market_making")
                
        elif regime_name == "strong_trend":
            strategies.extend(["trend_following", "momentum"])
            
        elif regime_name == "mean_reverting":
            strategies.extend(["mean_reversion", "market_making"])
            if condition.volatility > 0.5:
                strategies.append("volatility_breakout")
                
        elif regime_name == "low_volatility":
            if condition.liquidity > 0.7:
                strategies.extend(["market_making", "statistical_arbitrage"])
                
        else:  # normal or unknown regime
            strategies.extend(["adaptive_momentum", "statistical_arbitrage"])
        
        return strategies
        
    except Exception as e:
        logging.error(f"Strategy selection error: {str(e)}")
        return ["adaptive_momentum"]  # Default safe strategy

def _calculate_risk_profile(self, regime_name: str, condition: MarketCondition) -> Dict[str, float]:
    """Calculate regime-specific risk parameters"""
    try:
        # Base risk profile
        risk_profile = {
            'position_size': 1.0,
            'leverage': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_drawdown': 0.15,
            'risk_factor': 1.0
        }
        
        # Adjust based on regime
        if regime_name == "high_volatility" or regime_name == "crisis":
            risk_profile.update({
                'position_size': 0.5,
                'leverage': 1.0,
                'stop_loss': 0.03,
                'take_profit': 0.06,
                'max_drawdown': 0.10,
                'risk_factor': 0.5
            })
            
        elif regime_name == "low_volatility" or regime_name == "stable":
            risk_profile.update({
                'position_size': 1.5,
                'leverage': 1.5,
                'stop_loss': 0.01,
                'take_profit': 0.03,
                'max_drawdown': 0.20,
                'risk_factor': 1.5
            })
            
        # Further adjust based on conditions
        risk_profile = self._adjust_risk_profile(risk_profile, condition)
        
        return risk_profile
        
    except Exception as e:
        logging.error(f"Risk profile calculation error: {str(e)}")
        return {'position_size': 0.5, 'leverage': 1.0, 'stop_loss': 0.02}

def _adjust_risk_profile(self, profile: Dict[str, float], condition: MarketCondition) -> Dict[str, float]:
    """Fine-tune risk profile based on market conditions"""
    try:
        # Adjust for volatility
        volatility_factor = 1.0 - condition.volatility
        profile['position_size'] *= volatility_factor
        profile['leverage'] *= volatility_factor
        
        # Adjust for liquidity
        if condition.liquidity < 0.5:
            profile['position_size'] *= 0.7
            profile['leverage'] *= 0.7
        
        # Adjust for stability
        stability_factor = condition.regime_stability
        profile['risk_factor'] *= stability_factor
        
        return profile
        
    except Exception as e:
        logging.error(f"Risk profile adjustment error: {str(e)}")
        return profile

def _update_regime_history(self, regime: MarketRegime):
    """Update regime history and maintain history size"""
    try:
        self.history.append({
            'regime': regime,
            'timestamp': datetime.now()
        })
        
        # Maintain history size
        if len(self.history) > self.config.market_analysis.MAX_REGIME_HISTORY:
            self.history = self.history[-self.config.MAX_REGIME_HISTORY:]
            
    except Exception as e:
        logging.error(f"Regime history update error: {str(e)}")

def _combine_features(self, 
                         condition: MarketCondition,
                         deep_features: DeepFeatures) -> np.ndarray:
        """Combine traditional and deep features"""
        try:
            traditional_features = self._create_feature_vector(condition)
            
            combined = np.concatenate([
                traditional_features,
                deep_features.temporal_patterns,
                deep_features.market_structure,
                deep_features.regime_indicators
            ], axis=1)
            
            return self.scaler.fit_transform(combined)
            
        except Exception as e:
            logging.error(f"Feature combination error: {str(e)}")
            raise
