# File: src/core/performance_analytics.py

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from .metrics.metrics_system import MetricsCollector, PerformanceMetrics
from .market_analysis.regime_detector import MarketRegime

@dataclass
class RegimePerformanceSnapshot:
    """Snapshot of performance during a specific regime"""
    regime_name: str
    start_time: datetime
    end_time: datetime
    metrics: PerformanceMetrics
    strategy_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    transition_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class StrategyRegimeStats:
    """Statistics for strategy performance in specific regime"""
    success_rate: float = 0.0
    risk_adjusted_return: float = 0.0
    avg_trade_duration: float = 0.0
    profit_factor: float = 0.0
    regime_correlation: float = 0.0

class PerformanceAnalytics:
    """Advanced performance analytics with regime-specific tracking"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.regime_performance: Dict[str, List[RegimePerformanceSnapshot]] = {}
        self.strategy_regime_stats: Dict[str, Dict[str, StrategyRegimeStats]] = {}
        self.transition_stats: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger(__name__)
        
    async def analyze_performance(self,
                                trading_data: Dict,
                                system_state: Dict,
                                current_regime: Optional[MarketRegime] = None) -> Dict:
        """Analyze system performance comprehensively"""
        try:
            # Update base metrics through MetricsCollector
            await self.metrics_collector.update_performance_metrics(trading_data)
            await self.metrics_collector.update_system_metrics(system_state)
            
            # Get current performance snapshot
            metrics = self.metrics_collector.get_metrics_summary()
            
            # If regime is provided, update regime-specific metrics
            if current_regime:
                await self._update_regime_metrics(current_regime, trading_data, metrics)
            
            # Generate analytics and recommendations
            analytics = {
                'metrics': metrics,
                'patterns': await self._analyze_patterns(trading_data, metrics),
                'regime_analytics': await self._analyze_regime_performance() if current_regime else None,
                'recommendations': await self._generate_recommendations(metrics, current_regime)
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Performance analysis error: {str(e)}")
            raise
            
    async def _update_regime_metrics(self,
                                   regime: MarketRegime,
                                   trading_data: Dict,
                                   metrics: Dict):
        """Update regime-specific performance metrics"""
        try:
            # Create performance snapshot
            snapshot = RegimePerformanceSnapshot(
                regime_name=regime.name,
                start_time=trading_data.get('start_time', datetime.now()),
                end_time=trading_data.get('end_time', datetime.now()),
                metrics=metrics['performance'],
                strategy_performance=self._extract_strategy_performance(metrics),
                risk_metrics=self._calculate_regime_risk_metrics(regime, trading_data),
                transition_metrics=self._get_transition_metrics(regime)
            )
            
            # Update regime performance history
            if regime.name not in self.regime_performance:
                self.regime_performance[regime.name] = []
            self.regime_performance[regime.name].append(snapshot)
            
            # Update strategy-regime statistics
            await self._update_strategy_regime_stats(regime.name, snapshot)
            
        except Exception as e:
            self.logger.error(f"Regime metrics update error: {str(e)}")
    
    def _extract_strategy_performance(self, metrics: Dict) -> Dict[str, float]:
        """Extract strategy-specific performance from metrics"""
        return {
            name: stats['risk_adjusted_return']
            for name, stats in metrics.get('strategies', {}).items()
        }
    
    def _calculate_regime_risk_metrics(self, regime: MarketRegime, trading_data: Dict) -> Dict[str, float]:
        """Calculate risk metrics specific to the regime"""
        try:
            trades = trading_data.get('trades', [])
            returns = [t['pnl'] for t in trades]
            
            return {
                'volatility': np.std(returns) if returns else 0.0,
                'var_95': np.percentile(returns, 5) if returns else 0.0,
                'expected_shortfall': np.mean([r for r in returns if r < 0]) if returns else 0.0,
                'regime_stability': regime.confidence
            }
        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            return {}
    
    def _get_transition_metrics(self, regime: MarketRegime) -> Dict[str, float]:
        """Get metrics related to regime transitions"""
        if not self.transition_stats:
            return {}
            
        return self.transition_stats.get(regime.name, {})
    
    async def _update_strategy_regime_stats(self, regime_name: str, snapshot: RegimePerformanceSnapshot):
        """Update statistics for strategy performance in specific regime"""
        try:
            for strategy_name, performance in snapshot.strategy_performance.items():
                if strategy_name not in self.strategy_regime_stats:
                    self.strategy_regime_stats[strategy_name] = {}
                
                if regime_name not in self.strategy_regime_stats[strategy_name]:
                    self.strategy_regime_stats[strategy_name][regime_name] = StrategyRegimeStats()
                
                stats = self.strategy_regime_stats[strategy_name][regime_name]
                
                # Update stats using exponential moving average
                alpha = 0.3  # Smoothing factor
                stats.risk_adjusted_return = (
                    (1 - alpha) * stats.risk_adjusted_return +
                    alpha * performance
                )
                
                # Update regime correlation
                stats.regime_correlation = self._calculate_regime_correlation(
                    strategy_name,
                    regime_name
                )
                
        except Exception as e:
            self.logger.error(f"Strategy regime stats update error: {str(e)}")
    
    async def _analyze_patterns(self, trading_data: Dict, metrics: Dict) -> Dict:
        """Analyze performance patterns"""
        try:
            return {
                'trend': self._detect_performance_trend(metrics),
                'anomalies': self._detect_anomalies(metrics),
                'correlations': self._analyze_correlations(trading_data)
            }
        except Exception as e:
            self.logger.error(f"Pattern analysis error: {str(e)}")
            return {}
    
    async def _analyze_regime_performance(self) -> Dict:
        """Analyze performance across different regimes"""
        try:
            regime_analytics = {}
            for regime_name, snapshots in self.regime_performance.items():
                if not snapshots:
                    continue
                
                regime_analytics[regime_name] = {
                    'overall_performance': self._calculate_regime_overall_performance(snapshots),
                    'best_strategies': self._identify_best_strategies(regime_name),
                    'risk_profile': self._analyze_regime_risk_profile(snapshots),
                    'stability': self._analyze_regime_stability(snapshots)
                }
            
            return regime_analytics
            
        except Exception as e:
            self.logger.error(f"Regime performance analysis error: {str(e)}")
            return {}
    
    def _calculate_regime_correlation(self, strategy_name: str, regime_name: str) -> float:
        """Calculate correlation between strategy performance and regime"""
        try:
            if regime_name not in self.regime_performance:
                return 0.0
                
            snapshots = self.regime_performance[regime_name]
            if not snapshots:
                return 0.0
                
            performances = [
                s.strategy_performance.get(strategy_name, 0.0)
                for s in snapshots
            ]
            
            return np.corrcoef(performances, [1] * len(performances))[0, 1]
            
        except Exception as e:
            self.logger.error(f"Correlation calculation error: {str(e)}")
            return 0.0
    
    async def _generate_recommendations(self, 
                                     metrics: Dict,
                                     current_regime: Optional[MarketRegime]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        try:
            # System performance recommendations
            system_metrics = metrics.get('system', {})
            if system_metrics.get('avg_latency', 0) > self.config.LATENCY_THRESHOLD:
                recommendations.append(
                    "High latency detected. Consider optimizing execution pipeline."
                )
            
            # Strategy recommendations
            if current_regime:
                best_strategies = self._identify_best_strategies(current_regime.name)
                if best_strategies:
                    recommendations.append(
                        f"Consider increasing allocation to {best_strategies[0]} "
                        f"in {current_regime.name} regime."
                    )
            
            # Risk recommendations
            risk_metrics = metrics.get('performance', {})
            if risk_metrics.get('max_drawdown', 0) > self.config.MAX_DRAWDOWN_THRESHOLD:
                recommendations.append(
                    "Maximum drawdown exceeded threshold. Review risk management parameters."
                )
            
        except Exception as e:
            self.logger.error(f"Recommendations generation error: {str(e)}")
        
        return recommendations
    
    def _identify_best_strategies(self, regime_name: str) -> List[str]:
        """Identify best performing strategies for a regime"""
        try:
            strategy_scores = []
            for strategy_name, regime_stats in self.strategy_regime_stats.items():
                if regime_name in regime_stats:
                    stats = regime_stats[regime_name]
                    score = (
                        stats.risk_adjusted_return * 0.4 +
                        stats.success_rate * 0.3 +
                        stats.regime_correlation * 0.3
                    )
                    strategy_scores.append((strategy_name, score))
            
            return [
                name for name, _ in sorted(
                    strategy_scores,
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            
        except Exception as e:
            self.logger.error(f"Best strategies identification error: {str(e)}")
            return []
