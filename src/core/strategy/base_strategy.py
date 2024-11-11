# File: src/core/strategy/base_strategy.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from ..market_analysis.regime_detector import MarketRegime
from ..market_analysis.condition_analyzer import MarketCondition

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config):
        self.config = config
        self.name: str = "base_strategy"
        self.position = None
        self.parameters: Dict = {}
        self.performance_metrics: Dict = {}
        self.min_data_points: int = 100
        self.timeframes: List[str] = ['1h']  # Default timeframe
        
    @abstractmethod
    async def analyze(self, 
                     market_data: pd.DataFrame,
                     regime: MarketRegime) -> Dict:
        """Analyze market data and generate signals"""
        pass
    
    @abstractmethod
    async def generate_signals(self, 
                             market_data: pd.DataFrame,
                             regime: MarketRegime) -> Dict:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    async def calculate_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        pass
    
    async def validate_signals(self, signals: Dict, regime: MarketRegime) -> Dict:
        """Validate generated signals"""
        try:
            if not signals:
                return None
                
            # Check signal strength against regime-specific threshold
            base_threshold = self.config.market_analysis.STRATEGY_SELECTION_THRESHOLD
            regime_factor = 1.0 if regime.confidence > 0.8 else 1.2
            adjusted_threshold = base_threshold * regime_factor
            
            if abs(signals.get('strength', 0)) < adjusted_threshold:
                return None
            
            # Apply regime-specific filters
            filtered_signals = await self._apply_regime_filters(signals, regime)
            if not filtered_signals:
                return None
            
            # Add metadata
            filtered_signals.update({
                'strategy': self.name,
                'regime': regime.name,
                'confidence': regime.confidence,
                'timestamp': datetime.now()
            })
            
            return filtered_signals
            
        except Exception as e:
            logging.error(f"Signal validation error in {self.name}: {str(e)}")
            return None
    
    async def _apply_regime_filters(self, signals: Dict, regime: MarketRegime) -> Dict:
        """Apply regime-specific filters to signals"""
        try:
            filtered_signals = signals.copy()
            
            # Adjust signal strength based on regime confidence
            filtered_signals['strength'] *= regime.confidence
            
            # Adjust position size based on regime risk profile
            position_size = signals.get('position_size', 1.0)
            risk_factor = regime.risk_profile.get('risk_factor', 1.0)
            filtered_signals['position_size'] = position_size * risk_factor
            
            # Add regime-specific stop loss and take profit
            filtered_signals['stop_loss'] = regime.risk_profile.get('stop_loss')
            filtered_signals['take_profit'] = regime.risk_profile.get('take_profit')
            
            return filtered_signals
            
        except Exception as e:
            logging.error(f"Regime filter error in {self.name}: {str(e)}")
            return None
            
    async def _validate_data(self, market_data: pd.DataFrame) -> bool:
        """Validate input data"""
        try:
            if market_data is None or market_data.empty:
                return False
                
            if len(market_data) < self.min_data_points:
                return False
                
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Data validation error in {self.name}: {str(e)}")
            return False
    
    async def calculate_performance(self, 
                                 trades: List[Dict],
                                 market_data: pd.DataFrame) -> Dict:
        """Calculate strategy performance metrics"""
        try:
            if not trades:
                return {}
                
            returns = [t.get('return', 0) for t in trades]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            
            metrics = {
                'total_trades': len(trades),
                'win_rate': len(wins) / len(trades) if trades else 0,
                'avg_return': np.mean(returns) if returns else 0,
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
                'recovery_factor': self._calculate_recovery_factor(returns)
            }
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            logging.error(f"Performance calculation error in {self.name}: {str(e)}")
            return {}
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) != 0 else 0.0
        
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        cumulative = np.maximum.accumulate(np.array(returns))
        drawdowns = cumulative - np.array(returns)
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
    def _calculate_recovery_factor(self, returns: List[float]) -> float:
        """Calculate recovery factor"""
        if not returns:
            return 0.0
        total_return = sum(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        return abs(total_return / max_drawdown) if max_drawdown != 0 else float('inf')
