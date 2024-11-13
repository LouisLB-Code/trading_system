# File: src/core/metrics/metrics_collector.py

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass, field

@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta())
    trades_per_day: float = 0.0

@dataclass
class SystemMetrics:
    """System performance metrics"""
    latency_ms: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    event_queue_size: List[int] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    signals_generated: int = 0
    signals_executed: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    avg_position_duration: timedelta = field(default_factory=lambda: timedelta())
    risk_adjusted_return: float = 0.0

class MetricsCollector:
    """Collects and analyzes system metrics"""
    
    def __init__(self, config):
        self.config = config
        self.performance_metrics = PerformanceMetrics()
        self.system_metrics = SystemMetrics()
        self.strategy_metrics = {}  # Dict[str, StrategyMetrics]
        self.metrics_history = []
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def update_performance_metrics(self, trading_data: Dict):
        """Update trading performance metrics"""
        try:
            async with self._lock:
                trades = trading_data.get('trades', [])
                
                if not trades:
                    return
                
                # Calculate returns
                returns = [t['pnl'] for t in trades]
                positive_returns = [r for r in returns if r > 0]
                negative_returns = [r for r in returns if r < 0]
                
                # Update metrics
                self.performance_metrics.total_return = sum(returns)
                self.performance_metrics.win_rate = len(positive_returns) / len(returns)
                self.performance_metrics.profit_factor = (
                    abs(sum(positive_returns) / sum(negative_returns))
                    if negative_returns else float('inf')
                )
                
                # Calculate ratios
                self.performance_metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
                self.performance_metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
                self.performance_metrics.max_drawdown = self._calculate_max_drawdown(returns)
                
                # Record history
                self._record_metrics('performance', self.performance_metrics)
                
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {str(e)}")
    
    async def update_system_metrics(self, metrics: Dict):
        """Update system performance metrics"""
        try:
            async with self._lock:
                # Update system metrics
                self.system_metrics.latency_ms.append(metrics.get('latency_ms', 0))
                self.system_metrics.memory_usage.append(metrics.get('memory_usage', 0))
                self.system_metrics.cpu_usage.append(metrics.get('cpu_usage', 0))
                self.system_metrics.event_queue_size.append(metrics.get('event_queue_size', 0))
                
                # Keep only recent metrics
                max_history = self.config.METRICS_HISTORY_LENGTH
                self.system_metrics.latency_ms = self.system_metrics.latency_ms[-max_history:]
                self.system_metrics.memory_usage = self.system_metrics.memory_usage[-max_history:]
                self.system_metrics.cpu_usage = self.system_metrics.cpu_usage[-max_history:]
                self.system_metrics.event_queue_size = self.system_metrics.event_queue_size[-max_history:]
                
                # Record history
                self._record_metrics('system', self.system_metrics)
                
        except Exception as e:
            self.logger.error(f"System metrics update error: {str(e)}")
    
    async def update_strategy_metrics(self, strategy_name: str, metrics: Dict):
        """Update strategy-specific metrics"""
        try:
            async with self._lock:
                if strategy_name not in self.strategy_metrics:
                    self.strategy_metrics[strategy_name] = StrategyMetrics()
                
                strategy_metrics = self.strategy_metrics[strategy_name]
                
                # Update metrics
                strategy_metrics.signals_generated += metrics.get('signals_generated', 0)
                strategy_metrics.signals_executed += metrics.get('signals_executed', 0)
                strategy_metrics.successful_trades += metrics.get('successful_trades', 0)
                strategy_metrics.failed_trades += metrics.get('failed_trades', 0)
                
                if 'position_duration' in metrics:
                    strategy_metrics.avg_position_duration = metrics['position_duration']
                
                if 'risk_adjusted_return' in metrics:
                    strategy_metrics.risk_adjusted_return = metrics['risk_adjusted_return']
                
                # Record history
                self._record_metrics('strategy', {strategy_name: strategy_metrics})
                
        except Exception as e:
            self.logger.error(f"Strategy metrics update error: {str(e)}")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            'performance': self.performance_metrics,
            'system': {
                'avg_latency': np.mean(self.system_metrics.latency_ms),
                'avg_memory': np.mean(self.system_metrics.memory_usage),
                'avg_cpu': np.mean(self.system_metrics.cpu_usage),
                'error_rate': self.system_metrics.error_count / max(1, len(self.metrics_history))
            },
            'strategies': {
                name: self._get_strategy_summary(metrics)
                for name, metrics in self.strategy_metrics.items()
            }
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
            
        excess_returns = np.array(returns) - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0
            
        excess_returns = np.array(returns) - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return 0.0
            
        downside_std = np.std(downside_returns, ddof=1)
        return np.mean(excess_returns) / downside_std if downside_std != 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
            
        cumulative = np.maximum.accumulate(np.array(returns))
        drawdowns = cumulative - np.array(returns)
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _get_strategy_summary(self, metrics: StrategyMetrics) -> Dict:
        """Get summary of strategy metrics"""
        return {
            'success_rate': (
                metrics.successful_trades /
                max(1, metrics.successful_trades + metrics.failed_trades)
            ),
            'execution_rate': (
                metrics.signals_executed /
                max(1, metrics.signals_generated)
            ),
            'risk_adjusted_return': metrics.risk_adjusted_return,
            'avg_position_duration': metrics.avg_position_duration
        }
    
    def _record_metrics(self, metrics_type: str, metrics: Dict):
        """Record metrics in history"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'type': metrics_type,
            'metrics': metrics
        })
        
        # Prune old metrics
        if len(self.metrics_history) > self.config.METRICS_HISTORY_LENGTH:
            self.metrics_history = self.metrics_history[-self.config.METRICS_HISTORY_LENGTH:]
