# File: src/core/market_analysis/condition_analyzer.py

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

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
        
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility using Garman-Klass estimator"""
        try:
            high = np.log(market_data['high'])
            low = np.log(market_data['low'])
            close = np.log(market_data['close'])
            open = np.log(market_data['open'])
            
            volatility = np.sqrt(
                0.5 * (high - low)**2 - 
                (2*np.log(2)-1) * (close - open)**2
            ).mean()
            
            return float(volatility)
        except Exception as e:
            logging.error(f"Volatility calculation error: {str(e)}")
            return 0.0
            
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like measure"""
        try:
            # Calculate directional movement
            high_diff = market_data['high'].diff()
            low_diff = market_data['low'].diff()
            
            plus_dm = np.where(
                (high_diff > 0) & (high_diff > abs(low_diff)),
                high_diff,
                0
            )
            minus_dm = np.where(
                (low_diff < 0) & (abs(low_diff) > high_diff),
                abs(low_diff),
                0
            )
            
            # Calculate true range
            tr = np.maximum(
                market_data['high'] - market_data['low'],
                abs(market_data['high'] - market_data['close'].shift(1))
            )
            tr = np.maximum(tr, abs(market_data['low'] - market_data['close'].shift(1)))
            
            # Calculate trend strength
            dx = abs(
                np.mean(plus_dm) - np.mean(minus_dm)
            ) / (np.mean(plus_dm) + np.mean(minus_dm)) * 100
            
            return float(dx / 100)  # Normalize to [0,1]
        except Exception as e:
            logging.error(f"Trend strength calculation error: {str(e)}")
            return 0.0
            
    def _calculate_liquidity(self, market_data: pd.DataFrame) -> float:
        """Calculate liquidity measure"""
        try:
            # Calculate volume-weighted average price
            vwap = (market_data['close'] * market_data['volume']).sum() / market_data['volume'].sum()
            
            # Calculate average spread
            spread = (market_data['high'] - market_data['low']).mean()
            
            # Calculate volume stability
            volume_stability = 1 - (market_data['volume'].std() / market_data['volume'].mean())
            
            # Combine metrics
            liquidity = (
                (1 / (spread + 1e-8)) * 
                volume_stability * 
                market_data['volume'].mean()
            )
            
            # Normalize to [0,1]
            return float(np.clip(liquidity / self.config.LIQUIDITY_NORM_FACTOR, 0, 1))
        except Exception as e:
            logging.error(f"Liquidity calculation error: {str(e)}")
            return 0.0
            
    def _calculate_volume_profile(self, market_data: pd.DataFrame) -> float:
        """Calculate volume profile analysis"""
        try:
            # Calculate volume-weighted price levels
            price_levels = pd.qcut(market_data['close'], 10)
            volume_profile = market_data.groupby(price_levels)['volume'].sum()
            
            # Calculate concentration of volume
            total_volume = volume_profile.sum()
            concentration = (
                volume_profile.nlargest(3).sum() / total_volume
            )
            
            return float(concentration)
        except Exception as e:
            logging.error(f"Volume profile calculation error: {str(e)}")
            return 0.0
            
    def _calculate_momentum(self, market_data: pd.DataFrame) -> float:
        """Calculate market momentum"""
        try:
            # Calculate multiple momentum indicators
            roc = market_data['close'].pct_change(20)
            macd = (
                market_data['close'].ewm(span=12).mean() -
                market_data['close'].ewm(span=26).mean()
            )
            rsi = self._calculate_rsi(market_data['close'])
            
            # Combine indicators
            momentum = (
                0.4 * roc +
                0.3 * (macd / market_data['close']) +
                0.3 * (rsi / 100)
            ).mean()
            
            return float(np.clip(momentum, -1, 1))
        except Exception as e:
            logging.error(f"Momentum calculation error: {str(e)}")
            return 0.0
            
    def _calculate_efficiency_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate price efficiency ratio"""
        try:
            price_change = abs(
                market_data['close'].iloc[-1] -
                market_data['close'].iloc[0]
            )
            path_length = abs(market_data['close'].diff()).sum()
            
            efficiency = price_change / (path_length + 1e-8)
            return float(efficiency)
        except Exception as e:
            logging.error(f"Efficiency ratio calculation error: {str(e)}")
            return 0.0
            
    def _calculate_fractal_dimension(self, market_data: pd.DataFrame) -> float:
        """Calculate fractal dimension using Hurst exponent"""
        try:
            # Calculate Hurst exponent
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(
                market_data['close'][lag:],
                market_data['close'][:-lag]
            ))) for lag in lags]
            
            # Fit line to log-log plot
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2
            
            # Convert to fractal dimension
            fractal_dim = 2 - hurst
            return float(fractal_dim / 2)  # Normalize to [0,1]
        except Exception as e:
            logging.error(f"Fractal dimension calculation error: {str(e)}")
            return 0.0
            
    def _calculate_regime_stability(self, market_data: pd.DataFrame) -> float:
        """Calculate regime stability measure"""
        try:
            # Calculate various stability metrics
            volatility_stability = 1 - (
                market_data['close'].rolling(20).std().std() /
                market_data['close'].rolling(20).std().mean()
            )
            
            trend_stability = 1 - abs(
                market_data['close'].rolling(20).mean().diff()
            ).mean() / market_data['close'].mean()
            
            volume_stability = 1 - (
                market_data['volume'].rolling(20).std() /
                market_data['volume'].rolling(20).mean()
            )
            
            # Combine metrics
            stability = (
                0.4 * volatility_stability +
                0.4 * trend_stability +
                0.2 * volume_stability
            )
            
            return float(np.clip(stability, 0, 1))
        except Exception as e:
            logging.error(f"Regime stability calculation error: {str(e)}")
            return 0.0
            
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Helper method to calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
