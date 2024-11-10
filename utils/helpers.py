```python
import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import json

def setup_logging(config):
    """Setup logging configuration"""
    log_file = os.path.join(
        config.LOGS_PATH,
        f'system_{datetime.now().strftime("%Y%m%d")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def calculate_metrics(trades: List[Dict]) -> Dict:
    """Calculate performance metrics"""
    if not trades:
        return {}
        
    profits = [t['profit'] for t in trades]
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    metrics = {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'total_profit': sum(profits),
        'average_profit': np.mean(profits) if profits else 0,
        'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
        'max_drawdown': calculate_max_drawdown(profits)
    }
    
    return metrics

def calculate_max_drawdown(profits: List[float]) -> float:
    """Calculate maximum drawdown"""
    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return np.max(drawdown) if len(drawdown) > 0 else 0

def prepare_data_for_ml(df: pd.DataFrame, window_size: int = 60) -> tuple:
    """Prepare data for machine learning"""
    features = []
    targets = []
    
    for i in range(len(df) - window_size):
        features.append(df.iloc[i:i+window_size].values)
        targets.append(df.iloc[i+window_size]['close'])
        
    return np.array(features), np.array(targets)

def save_to_file(data: Dict, filepath: str):
    """Save data to file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving to file: {str(e)}")
        raise

def load_from_file(filepath: str) -> Dict:
    """Load data from file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading from file: {str(e)}")
        raise

class TradingUtils:
    """Utility functions for trading"""
    
    @staticmethod
    def calculate_position_size(capital: float, risk_per_trade: float, stop_loss_pct: float) -> float:
        """Calculate position size based on risk parameters"""
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        return position_size
    
    @staticmethod
    def calculate_stop_loss(entry_price: float, direction: int, stop_loss_pct: float) -> float:
        """Calculate stop loss price"""
        if direction > 0:
            return entry_price * (1 - stop_loss_pct)
        return entry_price * (1 + stop_loss_pct)
    
    @staticmethod
    def calculate_take_profit(entry_price: float, direction: int, risk_reward_ratio: float, stop_loss_pct: float) -> float:
        """Calculate take profit price"""
        if direction > 0:
            return entry_price * (1 + (stop_loss_pct * risk_reward_ratio))
        return entry_price * (1 - (stop_loss_pct * risk_reward_ratio))

class Validator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate input data"""
        try:
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                return False
            
            # Check for nulls
            if df[required_columns].isnull().any().any():
                return False
            
            # Check for price consistency
            if not all(df['high'] >= df['low']) or \
               not all(df['high'] >= df['close']) or \
               not all(df['high'] >= df['open']):
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Data validation error: {str(e)}")
            return False
```
