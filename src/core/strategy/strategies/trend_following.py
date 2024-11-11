# File: src/core/strategy/strategies/trend_following.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime

from ..base_strategy import BaseStrategy
from ...market_analysis.regime_detector import MarketRegime

class TrendFollowingStrategy(BaseStrategy):
    """Trend Following Strategy Implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "trend_following"
        self.timeframes = ['1h', '4h', '1d']  # Multiple timeframes for trend confirmation
        self.parameters = {
            'slow_ma': 200,
            'medium_ma': 50,
            'fast_ma': 20,
            'atr_period': 14,
            'trend_strength_threshold': 0.6,
            'min_bars': 200
        }
    
    async def analyze(self, 
                     market_data: pd.DataFrame,
                     regime: MarketRegime) -> Dict:
        """Analyze market data for trend following opportunities"""
        try:
            if not await self._validate_data(market_data):
                return {}
            
            # Calculate indicators
            indicators = await self.calculate_indicators(market_data)
            
            # Analyze trend strength and direction
            trend_analysis = self._analyze_trend(indicators)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Combine analyses
            analysis = {
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'regime_alignment': self._check_regime_alignment(
                    trend_analysis,
                    regime
                )
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Trend analysis error: {str(e)}")
            return {}
    
    async def generate_signals(self,
                             market_data: pd.DataFrame,
                             regime: MarketRegime) -> Dict:
        """Generate trend following signals"""
        try:
            # Perform analysis
            analysis = await self.analyze(market_data, regime)
            if not analysis:
                return None
            
            # Check for strong trend
            if not analysis['trend']['is_strong_trend']:
                return None
            
            # Generate signal
            signal = {
                'direction': analysis['trend']['direction'],
                'strength': analysis['trend']['strength'] * analysis['momentum']['strength'],
                'position_size': self._calculate_position_size(
                    analysis,
                    regime
                ),
                'entry_price': market_data['close'].iloc[-1],
                'stop_loss': self._calculate_stop_loss(
                    market_data,
                    analysis
                ),
                'take_profit': self._calculate_take_profit(
                    market_data,
                    analysis
                ),
                'analysis': analysis
            }
            
            return signal
            
        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}")
            return None
    
    async def calculate_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend following indicators"""
        try:
            df = market_data.copy()
            
            # Calculate moving averages
            df['sma_fast'] = df['close'].rolling(
                window=self.parameters['fast_ma']
            ).mean()
            
            df['sma_medium'] = df['close'].rolling(
                window=self.parameters['medium_ma']
            ).mean()
            
            df['sma_slow'] = df['close'].rolling(
                window=self.parameters['slow_ma']
            ).mean()
            
            # Calculate ATR
            df['atr'] = self._calculate_atr(
                df,
                self.parameters['atr_period']
            )
            
            # Calculate trend strength
            df['trend_strength'] = self._calculate_trend_strength(df)
            
            # Calculate momentum
            df['momentum'] = self._calculate_momentum(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Indicator calculation error: {str(e)}")
            raise
    
    def _analyze_trend(self, indicators: pd.DataFrame) -> Dict:
        """Analyze trend direction and strength"""
        try:
            current_price = indicators['close'].iloc[-1]
            
            # Check moving average alignment
            ma_alignment = (
                indicators['sma_fast'].iloc[-1] > indicators['sma_medium'].iloc[-1] and
                indicators['sma_medium'].iloc[-1] > indicators['sma_slow'].iloc[-1]
            )
            
            # Calculate trend direction
            direction = 1 if ma_alignment else -1
            
            # Calculate trend strength
            strength = indicators['trend_strength'].iloc[-1]
            
            return {
                'direction': direction,
                'strength': strength,
                'is_strong_trend': strength > self.parameters['trend_strength_threshold'],
                'ma_alignment': ma_alignment
            }
            
        except Exception as e:
            logging.error(f"Trend analysis error: {str(e)}")
            return {'direction': 0, 'strength': 0, 'is_strong_trend': False}
    
    def _analyze_momentum(self, indicators: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        try:
            momentum = indicators['momentum'].iloc[-1]
            
            return {
                'strength': abs(momentum),
                'direction': np.sign(momentum)
            }
            
        except Exception as e:
            logging.error(f"Momentum analysis error: {str(e)}")
            return {'strength': 0, 'direction': 0}
    
    def _check_regime_alignment(self, 
                              trend_analysis: Dict,
                              regime: MarketRegime) -> float:
        """Check if current trend aligns with market regime"""
        try:
            # Perfect alignment in trending regimes
            if regime.name in ['strong_trend', 'volatile_trend']:
                return 1.0
                
            # Reduced alignment in other regimes
            if regime.name in ['normal', 'low_volatility']:
                return 0.8
                
            # Poor alignment in mean reverting regimes
            if regime.name == 'mean_reverting':
                return 0.4
                
            return 0.6  # Default alignment
            
        except Exception as e:
            logging.error(f"Regime alignment error: {str(e)}")
            return 0.5
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength indicator"""
        try:
            # Calculate price changes
            price_changes = data['close'].diff().abs()
            
            # Calculate directional movement
            directional_movement = (
                data['close'].diff().rolling(
                    window=self.parameters['fast_ma']
                ).sum()
            ).abs()
            
            # Calculate strength as ratio of directional movement to total movement
            strength = directional_movement / price_changes.rolling(
                window=self.parameters['fast_ma']
            ).sum()
            
            return strength
            
        except Exception as e:
            logging.error(f"Trend strength calculation error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            return true_range.rolling(window=period).mean()
            
        except Exception as e:
            logging.error(f"ATR calculation error: {str(e)}")
            return pd.Series(0, index=data.index)
    
    def _calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum indicator"""
        try:
            # Use ROC (Rate of Change) as momentum indicator
            momentum = data['close'].pct_change(
                periods=self.parameters['fast_ma']
            )
            
            return momentum
            
        except Exception as e:
            logging.error(f"Momentum calculation error: {str(e)}")
            return pd.Series(0, index=data.index)
