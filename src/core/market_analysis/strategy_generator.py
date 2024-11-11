# core/market_analysis/strategy_generator.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from .regime_detector import MarketRegime
from .condition_analyzer import MarketCondition

@dataclass
class TradingStrategy:
    name: str
    parameters: Dict[str, float]
    risk_settings: Dict[str, float]
    performance_metrics: Dict[str, float]
    regime_compatibility: Dict[str, float]
    timestamp: datetime

class StrategyGenerator:
    """Generates and manages trading strategies based on market regimes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy_library = {}
        self.active_strategies = []
        self.performance_history = []
        
    async def generate_strategies(self, regime: MarketRegime) -> List[TradingStrategy]:
        """Generate optimal strategies for current market regime"""
        try:
            strategies = []
            
            # Get base strategies for regime
            base_strategies = self._get_base_strategies(regime.name)
            
            for strategy_name in base_strategies:
                # Optimize parameters for current conditions
                params = self._optimize_parameters(
                    strategy_name,
                    regime.condition
                )
                
                # Calculate strategy compatibility
                compatibility = self._calculate_compatibility(
                    strategy_name,
                    regime
                )
                
                # Apply risk settings
                risk_settings = self._apply_risk_settings(
                    regime.risk_profile,
                    strategy_name
                )
                
                # Create strategy object
                strategy = TradingStrategy(
                    name=strategy_name,
                    parameters=params,
                    risk_settings=risk_settings,
                    performance_metrics=self._get_performance_metrics(strategy_name),
                    regime_compatibility=compatibility,
                    timestamp=datetime.now()
                )
                
                strategies.append(strategy)
                
            return sorted(
                strategies,
                key=lambda x: x.regime_compatibility[regime.name],
                reverse=True
            )
            
        except Exception as e:
            logging.error(f"Strategy generation error: {str(e)}")
            raise
            
    def _get_base_strategies(self, regime_name: str) -> List[str]:
        """Get base strategies suitable for regime"""
        strategy_map = {
            "volatile_trend": [
                "trend_following",
                "breakout",
                "momentum"
            ],
            "high_volatility": [
                "volatility_breakout",
                "mean_reversion",
                "market_making"
            ],
            "strong_trend": [
                "trend_following",
                "momentum",
                "channel_breakout"
            ],
            "mean_reverting": [
                "mean_reversion",
                "market_making",
                "statistical_arbitrage"
            ],
            "low_volatility": [
                "market_making",
                "statistical_arbitrage",
                "pairs_trading"
            ]
        }
        return strategy_map.get(regime_name, ["adaptive_momentum"])
        
    def _optimize_parameters(self,
                           strategy_name: str,
                           market_condition: MarketCondition) -> Dict[str, float]:
        """Optimize strategy parameters for current market conditions"""
        try:
            base_params = self._get_base_parameters(strategy_name)
            
            # Adjust parameters based on market conditions
            optimized = {}
            for param, value in base_params.items():
                optimized[param] = self._adjust_parameter(
                    param,
                    value,
                    market_condition
                )
                
            return optimized
            
        except Exception as e:
            logging.error(f"Parameter optimization error: {str(e)}")
            return {}
            
    def _calculate_compatibility(self,
                               strategy_name: str,
                               regime: MarketRegime) -> Dict[str, float]:
        """Calculate strategy compatibility with market regime"""
        try:
            compatibility = {}
            
            # Calculate based on historical performance
            historical_perf = self._get_historical_performance(
                strategy_name,
                regime.name
            )
            
            # Calculate based on current conditions
            condition_score = self._get_condition_compatibility(
                strategy_name,
                regime.condition
            )
            
            # Combine scores
            compatibility[regime.name] = 0.7 * historical_perf + 0.3 * condition_score
            
            return compatibility
            
        except Exception as e:
            logging.error(f"Compatibility calculation error: {str(e)}")
            return {regime.name: 0.0}
            
    def _apply_risk_settings(self,
                           risk_profile: Dict[str, float],
                           strategy_name: str) -> Dict[str, float]:
        """Apply regime-specific risk settings to strategy"""
        try:
            base_settings = self._get_base_risk_settings(strategy_name)
            
            # Adjust settings based on risk profile
            settings = {}
            for setting, value in base_settings.items():
                settings[setting] = value * risk_profile['risk_factor']
                
            # Apply position size and leverage limits
            settings['position_size'] = min(
                settings['position_size'],
                risk_profile['position_size']
            )
            settings['leverage'] = min(
                settings['leverage'],
                risk_profile['leverage']
            )
            
            return settings
            
        except Exception as e:
            logging.error(f"Risk settings application error: {str(e)}")
            return {}
            
    def _get_performance_metrics(self, strategy_name: str) -> Dict[str, float]:
        """Get historical performance metrics for strategy"""
        # Implementation depends on your performance tracking system
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
    def _get_base_parameters(self, strategy_name: str) -> Dict[str, float]:
        """Get base parameters for strategy type"""
        # Add your strategy-specific base parameters
        return {
            'lookback_period': 20,
            'entry_threshold': 1.5,
            'exit_threshold': 0.5,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
        
    def _get_base_risk_settings(self, strategy_name: str) -> Dict[str, float]:
        """Get base risk settings for strategy type"""
        # Add your strategy-specific risk settings
        return {
            'position_size': 1.0,
            'leverage': 1.0,
            'max_trades': 5,
            'max_correlation': 0.7
        }
        
    def _adjust_parameter(self,
                         param: str,
                         value: float,
                         market_condition: MarketCondition) -> float:
        """Adjust parameter based on market conditions"""
        try:
            # Add your parameter adjustment logic
            if param == 'lookback_period':
                return value * (1 + market_condition.volatility)
                
            if param == 'entry_threshold':
                return value * (1 + market_condition.trend_strength)
                
            if param == 'exit_threshold':
                return value * (1 + market_condition.efficiency_ratio)
                
            return value
            
        except Exception as e:
            logging.error(f"Parameter adjustment error: {str(e)}")
            return value
