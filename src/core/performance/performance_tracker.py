from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from datetime import datetime

@dataclass
class RegimePerformanceMetrics:
    regime_name: str
    strategy_performance: Dict[str, float]
    risk_adjusted_returns: float
    drawdown: float
    win_rate: float
    recovery_factor: float
    timestamp: datetime

class RegimePerformanceTracker:
    def __init__(self, config):
        self.config = config
        self.performance_history = {}
        self.current_regime_performance = None
        
    async def track_regime_performance(self, 
                                     regime: 'MarketRegime',
                                     trading_results: Dict) -> RegimePerformanceMetrics:
        """Track performance metrics for current regime"""
        try:
            metrics = self._calculate_regime_metrics(regime, trading_results)
            self._update_performance_history(regime.name, metrics)
            return metrics
        except Exception as e:
            logging.error(f"Performance tracking error: {str(e)}")
            raise
