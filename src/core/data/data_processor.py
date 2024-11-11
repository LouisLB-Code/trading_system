# File: src/core/data/data_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class DataQualityMetrics:
    missing_values: int
    outliers: int
    gaps: List[Dict[str, datetime]]
    data_points: int
    start_time: datetime
    end_time: datetime
    symbols: List[str]

class DataProcessor:
    """Processes and prepares market data for analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def process_market_data(self, 
                                data: pd.DataFrame,
                                quality_check: bool = True) -> pd.DataFrame:
        """Process raw market data"""
        try:
            # Make copy to avoid modifying original
            df = data.copy()
            
            # Check data quality if requested
            if quality_check:
                quality_metrics = self._check_data_quality(df)
                self._log_quality_metrics(quality_metrics)
            
            # Basic cleaning
            df = self._clean_data(df)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Add derived features
            df = self._add_derived_features(df)
            
            # Add market microstructure features
            df = self._add_microstructure_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data processing error: {str(e)}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw market data"""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove outliers
            df = self._remove_outliers(df)
            
            # Validate price relationships
            df = self._validate_price_relationships(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data cleaning error: {str(e)}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data"""
        try:
            # Forward fill small gaps
            df = df.ffill(limit=self.config.MAX_FFILL_PERIODS)
            
            # Interpolate remaining gaps for price columns
            price_columns = ['open', 'high', 'low', 'close']
            df[price_columns] = df[price_columns].interpolate(
                method='time',
                limit=self.config.MAX_INTERPOLATION_PERIODS
            )
            
            # Forward fill non-price columns
            non_price_columns = [col for col in df.columns if col not in price_columns]
            df[non_price_columns] = df[non_price_columns].ffill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Missing value handling error: {str(e)}")
            raise

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or adjust outliers"""
        try:
            # Calculate rolling statistics
            rolling_mean = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            
            # Define outlier bounds
            upper_bound = rolling_mean + (rolling_std * self.config.OUTLIER_STD_THRESHOLD)
            lower_bound = rolling_mean - (rolling_std * self.config.OUTLIER_STD_THRESHOLD)
            
            # Identify outliers
            outliers = (df['close'] > upper_bound) | (df['close'] < lower_bound)
            
            # Replace outliers with rolling mean
            df.loc[outliers, 'close'] = rolling_mean[outliers]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Outlier removal error: {str(e)}")
            raise

    def _validate_price_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix price relationships (high >= low, etc.)"""
        try:
            # Ensure high is highest
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            
            # Ensure low is lowest
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Price validation error: {str(e)}")
            raise

    def _check_data_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Check data quality and return metrics"""
        try:
            # Check missing values
            missing_values = df.isnull().sum().sum()
            
            # Check for gaps
            time_gaps = self._find_time_gaps(df)
            
            # Check for outliers
            rolling_std = df['close'].rolling(window=20).std()
            rolling_mean = df['close'].rolling(window=20).mean()
            outliers = ((df['close'] - rolling_mean).abs() > 
                       (rolling_std * self.config.OUTLIER_STD_THRESHOLD)).sum()
            
            return DataQualityMetrics(
                missing_values=missing_values,
                outliers=outliers,
                gaps=time_gaps,
                data_points=len(df),
                start_time=df.index[0],
                end_time=df.index[-1],
                symbols=list(df['symbol'].unique()) if 'symbol' in df else ['unknown']
            )
            
        except Exception as e:
            self.logger.error(f"Data quality check error: {str(e)}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Volatility indicators
            df['atr'] = self._calculate_atr(df)
            df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df)
            
            # Momentum indicators
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['obv'] = self._calculate_obv(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation error: {str(e)}")
            raise

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            atr = true_range.rolling(window=period).mean()
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR calculation error: {str(e)}")
            raise

    def _calculate_bollinger_bands(self, 
                                df: pd.DataFrame,
                                period: int = 20,
                                std: float = 2.0) -> tuple:
        """Calculate Bollinger Bands"""
        try:
            middle_band = df['close'].rolling(window=period).mean()
            std_dev = df['close'].rolling(window=period).std()
            
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return upper_band, lower_band
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {str(e)}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {str(e)}")
            raise

    def _calculate_macd(self, 
                      prices: pd.Series,
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> tuple:
        """Calculate MACD and Signal line"""
        try:
            fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
            
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            
            return macd, signal
            
        except Exception as e:
            self.logger.error(f"MACD calculation error: {str(e)}")
            raise

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = 0
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
            
        except Exception as e:
            self.logger.error(f"OBV calculation error: {str(e)}")
            raise

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            # Calculate price changes
            df['price_change'] = df['close'].diff()
            df['returns'] = df['close'].pct_change()
            
            # Volatility measures
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['parkinson_volatility'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                np.power(np.log(df['high'] / df['low']), 2)
            )
            
            # Volume analysis
            df['volume_intensity'] = (df['volume'] / df['volume'].rolling(window=20).mean())
            df['dollar_volume'] = df['close'] * df['volume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Microstructure feature calculation error: {str(e)}")
            raise

    def _find_time_gaps(self, df: pd.DataFrame) -> List[Dict[str, datetime]]:
        """Find gaps in time series data"""
        try:
            time_diff = df.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=self._get_timeframe_minutes(df))
            
            gaps = []
            for i, diff in enumerate(time_diff[1:], 1):
                if diff > expected_diff * 1.5:  # Gap threshold
                    gaps.append({
                        'start': df.index[i-1],
                        'end': df.index[i],
                        'duration': diff
                    })
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Time gap detection error: {str(e)}")
            raise
            
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived technical features"""
        try:
            # Price action features
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            # Range and spread features
            df['daily_range'] = df['high'] - df['low']
            df['daily_range_pct'] = df['daily_range'] / df['close']
            
            # Trend features
            df['trend_strength'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            df['higher_highs'] = df['high'] > df['high'].shift(1)
            df['lower_lows'] = df['low'] < df['low'].shift(1)
            
            # Price momentum
            df['momentum_1d'] = df['close'] - df['close'].shift(1)
            df['momentum_5d'] = df['close'] - df['close'].shift(5)
            
            # Volume features
            df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Derived features calculation error: {str(e)}")
            raise

    def _get_timeframe_minutes(self, df: pd.DataFrame) -> int:
        """Detect timeframe in minutes from data"""
        try:
            # Get most common time difference between rows
            time_diff = df.index.to_series().diff().mode()[0]
            return int(time_diff.total_seconds() / 60)
            
        except Exception as e:
            self.logger.error(f"Timeframe detection error: {str(e)}")
            return 60  # Default to 1 hour