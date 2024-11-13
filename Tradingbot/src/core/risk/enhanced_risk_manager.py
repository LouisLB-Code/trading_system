# File: src/core/risk/enhanced_risk_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    var: float
    cvar: float
    sharpe_ratio: float
    max_drawdown: float
    current_exposure: float
    position_concentration: float
    correlation_risk: float
    regime_risk: float
    volatility_risk: float  # Added
    liquidity_risk: float   # Added
    tail_risk: float        # Added

class EnhancedRiskManager:
    """Advanced risk management system with adaptive thresholds and machine learning"""
    def __init__(self, config):
        self.config = config
        self.risk_metrics = {}
        self.position_manager = PositionRiskManager()
        self.threshold_manager = AdaptiveThresholdManager()
        self.correlation_manager = CorrelationRiskManager()
        self.ml_risk_analyzer = MLRiskAnalyzer(config)  # New ML component
        self.risk_history = []
        
    async def evaluate_risk(self,
                          positions: Dict,
                          market_data: pd.DataFrame,
                          regime: 'MarketRegime') -> RiskMetrics:
        """Evaluate comprehensive risk metrics"""
        try:
            # Calculate traditional risk metrics
            base_metrics = await self._calculate_base_metrics(
                positions,
                market_data
            )
            
            # Calculate ML-based risk predictions
            ml_predictions = await self.ml_risk_analyzer.predict_risk(
                positions,
                market_data,
                regime
            )
            
            # Calculate tail risk using extreme value theory
            tail_risk = self._calculate_tail_risk(
                positions,
                market_data
            )
            
            # Calculate liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(
                positions,
                market_data
            )
            
            # Combine all metrics
            risk_metrics = RiskMetrics(
                **base_metrics,
                tail_risk=tail_risk,
                liquidity_risk=liquidity_risk,
                **ml_predictions
            )
            
            # Store metrics and update adaptive thresholds
            self._update_risk_history(risk_metrics, regime)
            
            return risk_metrics
            
        except Exception as e:
            logging.error(f"Risk evaluation error: {str(e)}")
            raise

    async def _calculate_base_metrics(self,
                                    positions: Dict,
                                    market_data: pd.DataFrame) -> Dict:
        """Calculate base risk metrics"""
        try:
            # Calculate VaR using multiple methods
            var = self._calculate_hybrid_var(positions, market_data)
            
            # Calculate CVaR (Expected Shortfall)
            cvar = self._calculate_cvar(positions, market_data)
            
            # Calculate portfolio metrics
            portfolio_metrics = await self.position_manager.calculate_portfolio_metrics(
                positions,
                market_data
            )
            
            # Calculate correlation risk with regime adjustments
            correlation_risk = await self.correlation_manager.calculate_correlation_risk(
                positions,
                market_data
            )
            
            return {
                'var': var,
                'cvar': cvar,
                **portfolio_metrics,
                'correlation_risk': correlation_risk
            }
            
        except Exception as e:
            logging.error(f"Base metrics calculation error: {str(e)}")
            raise
            
    def _calculate_hybrid_var(self,
                            positions: Dict,
                            market_data: pd.DataFrame,
                            confidence_level: float = 0.99) -> float:
        """Calculate VaR using a hybrid approach (parametric + historical + Monte Carlo)"""
        try:
            # Calculate parametric VaR
            parametric_var = self._calculate_parametric_var(
                positions,
                market_data,
                confidence_level
            )
            
            # Calculate historical VaR
            historical_var = self._calculate_historical_var(
                positions,
                market_data,
                confidence_level
            )
            
            # Calculate Monte Carlo VaR
            monte_carlo_var = self._calculate_monte_carlo_var(
                positions,
                market_data,
                confidence_level
            )
            
            # Weight the different VaR calculations based on market regime
            weights = self._get_var_weights(market_data)
            
            return (
                weights['parametric'] * parametric_var +
                weights['historical'] * historical_var +
                weights['monte_carlo'] * monte_carlo_var
            )
            
        except Exception as e:
            logging.error(f"Hybrid VaR calculation error: {str(e)}")
            return float('inf')
            
    def _calculate_tail_risk(self,
                           positions: Dict,
                           market_data: pd.DataFrame) -> float:
        """Calculate tail risk using Extreme Value Theory"""
        try:
            returns = self._calculate_portfolio_returns(positions, market_data)
            
            # Fit Generalized Pareto Distribution to tail
            tail_threshold = np.percentile(returns, 5)
            tail_returns = returns[returns < tail_threshold]
            
            if len(tail_returns) < 20:  # Need enough tail events
                return float('inf')
            
            # Calculate GPD parameters
            shape, loc, scale = self._fit_gpd(tail_returns)
            
            # Calculate tail risk measure
            tail_risk = -1 * (loc + scale/shape)
            
            return float(tail_risk)
            
        except Exception as e:
            logging.error(f"Tail risk calculation error: {str(e)}")
            return float('inf')
            
    def _calculate_liquidity_risk(self,
                                positions: Dict,
                                market_data: pd.DataFrame) -> float:
        """Calculate portfolio liquidity risk"""
        try:
            liquidity_scores = []
            
            for symbol, position in positions.items():
                # Calculate bid-ask spread
                spread = self._calculate_spread(market_data, symbol)
                
                # Calculate market depth
                depth = self._calculate_market_depth(market_data, symbol)
                
                # Calculate trading volume relative to position size
                volume_ratio = self._calculate_volume_ratio(
                    market_data,
                    symbol,
                    position['size']
                )
                
                # Combine metrics
                liquidity_score = self._combine_liquidity_metrics(
                    spread,
                    depth,
                    volume_ratio
                )
                
                liquidity_scores.append(liquidity_score)
            
            return np.mean(liquidity_scores)
            
        except Exception as e:
            logging.error(f"Liquidity risk calculation error: {str(e)}")
            return float('inf')

class MLRiskAnalyzer:
    """Machine learning based risk analysis"""
    
    def __init__(self, config):
        self.config = config
        self.model = self._build_risk_model()
        self.feature_extractor = self._build_feature_extractor()
        
    async def predict_risk(self,
                         positions: Dict,
                         market_data: pd.DataFrame,
                         regime: 'MarketRegime') -> Dict:
        """Predict various risk metrics using ML"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(
                positions,
                market_data,
                regime
            )
            
            # Make predictions
            predictions = self.model.predict(features)
            
            return {
                'ml_volatility_risk': float(predictions['volatility']),
                'ml_drawdown_risk': float(predictions['drawdown']),
                'ml_tail_risk': float(predictions['tail'])
            }
            
        except Exception as e:
            logging.error(f"ML risk prediction error: {str(e)}")
            return {'ml_volatility_risk': float('inf'),
                    'ml_drawdown_risk': float('inf'),
                    'ml_tail_risk': float('inf')}
