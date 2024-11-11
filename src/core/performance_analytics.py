# src/core/performance_analytics.py

from typing import Dict, List, Optional
from datetime import datetime
import logging
import numpy as np
from dataclasses import dataclass
from .metrics.metrics_system import MetricsCollector, PerformanceMetrics, SystemMetrics, StrategyMetrics

@dataclass
class RegimePerformanceMetrics:
    """Regime-specific performance metrics"""
    regime_name: str
    start_time: datetime
    end_time: datetime
    performance: PerformanceMetrics
    strategy_metrics: Dict[str, StrategyMetrics]
    transition_impact: float = 0.0
    stability_score: float = 0.0
    regime_duration: float = 0.0

class PerformanceAnalytics:
    """Advanced performance analytics with regime-specific tracking"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.regime_history: List[RegimePerformanceMetrics] = []
        self.current_regime: Optional[RegimePerformanceMetrics] = None
        self.logger = logging.getLogger(__name__)
        
    async def analyze_performance(self,
                                trading_data: Dict,
                                system_state: Dict,
                                current_regime: str) -> Dict:
        """Analyze system performance comprehensively"""
        try:
            # Update base metrics using existing collector
            await self.metrics_collector.update_performance_metrics(trading_data)
            await self.metrics_collector.update_system_metrics(system_state)
            
            # Get regime-specific analysis
            regime_metrics = await self._analyze_regime_performance(
                trading_data,
                system_state,
                current_regime
            )
            
            # Combine all metrics
            metrics_summary = self.metrics_collector.get_metrics_summary()
            metrics_summary['regime'] = regime_metrics
            
            return {
                'metrics': metrics_summary,
                'patterns': await self._analyze_patterns(trading_data, current_regime),
                'optimizations': await self._generate_optimizations(metrics_summary),
                'recommendations': self._generate_recommendations(metrics_summary)
            }
            
        except Exception as e:
            self.logger.error(f"Performance analysis error: {str(e)}")
            raise
            
    async def _analyze_regime_performance(self,
                                        trading_data: Dict,
                                        system_state: Dict,
                                        regime_name: str) -> Dict:
        """Analyze performance specific to current market regime"""
        try:
            # Create new regime metrics if regime changed
            if not self.current_regime or self.current_regime.regime_name != regime_name:
                if self.current_regime:
                    self.current_regime.end_time = datetime.now()
                    self.regime_history.append(self.current_regime)
                
                self.current_regime = RegimePerformanceMetrics(
                    regime_name=regime_name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    performance=PerformanceMetrics(),
                    strategy_metrics={}
                )
            
            # Update regime-specific metrics
            self._update_regime_metrics(trading_data, system_state)
            
            return {
                'current': self._get_regime_summary(self.current_regime),
                'history': self._get_regime_history_summary(),
                'transitions': self._analyze_regime_transitions()
            }
            
        except Exception as e:
            self.logger.error(f"Regime performance analysis error: {str(e)}")
            return {}
    
    def _update_regime_metrics(self, trading_data: Dict, system_state: Dict):
        """Update metrics for current regime"""
        try:
            if not self.current_regime:
                return
                
            # Update duration
            self.current_regime.regime_duration = (
                datetime.now() - self.current_regime.start_time
            ).total_seconds() / 3600  # Convert to hours
            
            # Calculate regime stability
            self.current_regime.stability_score = self._calculate_regime_stability(
                trading_data,
                system_state
            )
            
            # Update strategy metrics for current regime
            for strategy_name, strategy_data in trading_data.get('strategies', {}).items():
                if strategy_name not in self.current_regime.strategy_metrics:
                    self.current_regime.strategy_metrics[strategy_name] = StrategyMetrics()
                
                # Update strategy metrics
                strategy_metrics = self.current_regime.strategy_metrics[strategy_name]
                self._update_strategy_regime_metrics(strategy_metrics, strategy_data)
                
        except Exception as e:
            self.logger.error(f"Regime metrics update error: {str(e)}")
    
    def _calculate_regime_stability(self, trading_data: Dict, system_state: Dict) -> float:
        """Calculate stability score for current regime"""
        try:
            # Factors contributing to stability:
            # 1. Consistency of strategy performance
            # 2. Market condition stability
            # 3. System performance stability
            
            strategy_stability = self._calculate_strategy_stability(trading_data)
            market_stability = self._calculate_market_stability(trading_data)
            system_stability = self._calculate_system_stability(system_state)
            
            # Weighted average of stability factors
            weights = [0.4, 0.4, 0.2]  # Adjustable weights
            stability = np.average(
                [strategy_stability, market_stability, system_stability],
                weights=weights
            )
            
            return float(stability)
            
        except Exception as e:
            self.logger.error(f"Stability calculation error: {str(e)}")
            return 0.0
    
    def _get_regime_summary(self, regime: RegimePerformanceMetrics) -> Dict:
        """Get summary of regime performance"""
        return {
            'name': regime.regime_name,
            'duration_hours': regime.regime_duration,
            'stability': regime.stability_score,
            'performance': {
                'total_return': regime.performance.total_return,
                'sharpe_ratio': regime.performance.sharpe_ratio,
                'max_drawdown': regime.performance.max_drawdown,
                'win_rate': regime.performance.win_rate
            },
            'strategies': {
                name: self._get_strategy_regime_summary(metrics)
                for name, metrics in regime.strategy_metrics.items()
            }
        }
    
    def _get_strategy_regime_summary(self, metrics: StrategyMetrics) -> Dict:
        """Get summary of strategy performance in regime"""
        return {
            'success_rate': metrics.successful_trades / max(1, metrics.successful_trades + metrics.failed_trades),
            'risk_adjusted_return': metrics.risk_adjusted_return,
            'signals_accuracy': metrics.signals_executed / max(1, metrics.signals_generated)
        }
    
    async def _analyze_patterns(self, trading_data: Dict, current_regime: str) -> Dict:
        """Analyze performance patterns"""
        try:
            # Analyze regime-specific patterns
            regime_patterns = self._analyze_regime_patterns(current_regime)
            
            # Analyze strategy adaptation patterns
            strategy_patterns = self._analyze_strategy_patterns(trading_data)
            
            return {
                'regime_patterns': regime_patterns,
                'strategy_patterns': strategy_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Pattern analysis error: {str(e)}")
            return {}
    
    async def _generate_optimizations(self, metrics_summary: Dict) -> Dict:
        """Generate optimization recommendations"""
        try:
            regime_optimizations = self._generate_regime_optimizations(
                metrics_summary['regime']
            )
            
            strategy_optimizations = self._generate_strategy_optimizations(
                metrics_summary['strategies']
            )
            
            return {
                'regime_optimizations': regime_optimizations,
                'strategy_optimizations': strategy_optimizations
            }
            
        except Exception as e:
            self.logger.error(f"Optimization generation error: {str(e)}")
            return {}
    
    def _generate_recommendations(self, metrics_summary: Dict) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Analyze regime performance
            regime_metrics = metrics_summary.get('regime', {})
            if regime_metrics:
                if regime_metrics.get('stability', 0) < 0.5:
                    recommendations.append(
                        "Consider adjusting regime detection parameters for better stability"
                    )
            
            # Analyze strategy performance
            strategies = metrics_summary.get('strategies', {})
            for strategy_name, metrics in strategies.items():
                if metrics.get('success_rate', 0) < 0.4:
                    recommendations.append(
                        f"Review {strategy_name} parameters for current regime"
                    )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return []
