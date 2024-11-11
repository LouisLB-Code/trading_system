# File: src/core/strategy/strategy_manager.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class StrategyPerformance:
    returns: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    recovery_factor: float
    regime_performance: Dict[str, float]
    confidence: float

@dataclass
class StrategyConfig:
    name: str
    parameters: Dict
    constraints: Dict
    risk_limits: Dict
    optimization_bounds: Dict

class StrategyManager:
    """Manages multiple trading strategies with adaptive selection"""
    
    def __init__(self, config):
        self.config = config
        self.active_strategies = {}
        self.strategy_pool = {}
        self.performance_history = {}
        self.regime_performance = {}
        self.optimization_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize_strategies(self):
        """Initialize strategy pool"""
        try:
            # Initialize base strategies
            self.strategy_pool = {
                'trend_following': TrendFollowingStrategy(self.config),
                'mean_reversion': MeanReversionStrategy(self.config),
                'momentum': MomentumStrategy(self.config),
                'breakout': BreakoutStrategy(self.config),
                'volatility': VolatilityStrategy(self.config),
                'market_making': MarketMakingStrategy(self.config),
                'statistical_arbitrage': StatArbitrageStrategy(self.config)
            }
            
            # Load strategy configurations
            await self._load_strategy_configs()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            logging.info("Strategy pool initialized successfully")
            
        except Exception as e:
            logging.error(f"Strategy initialization error: {str(e)}")
            raise
            
    async def select_strategies(self, 
                              market_regime: 'MarketRegime',
                              market_data: pd.DataFrame) -> List[str]:
        """Select optimal strategies for current market conditions"""
        try:
            selected_strategies = []
            
            # Calculate strategy scores
            scores = await self._calculate_strategy_scores(
                market_regime,
                market_data
            )
            
            # Select top performing strategies
            for strategy_name, score in sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if score > self.config.STRATEGY_SELECTION_THRESHOLD:
                    selected_strategies.append(strategy_name)
                    
                if len(selected_strategies) >= self.config.MAX_ACTIVE_STRATEGIES:
                    break
            
            # Update active strategies
            await self._update_active_strategies(selected_strategies)
            
            return selected_strategies
            
        except Exception as e:
            logging.error(f"Strategy selection error: {str(e)}")
            raise
            
    async def optimize_strategies(self):
        """Optimize strategy parameters based on performance"""
        while True:
            try:
                # Get strategy for optimization
                strategy_name = await self.optimization_queue.get()
                
                # Get strategy performance data
                performance_data = self.performance_history[strategy_name]
                
                # Optimize parameters
                new_params = await self._optimize_strategy_parameters(
                    strategy_name,
                    performance_data
                )
                
                # Update strategy parameters
                await self._update_strategy_parameters(
                    strategy_name,
                    new_params
                )
                
                # Mark optimization task as done
                self.optimization_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Strategy optimization error: {str(e)}")
                
    async def evaluate_performance(self, market_data: pd.DataFrame):
        """Evaluate performance of all active strategies"""
        try:
            for strategy_name, strategy in self.active_strategies.items():
                # Calculate performance metrics
                performance = await self._calculate_strategy_performance(
                    strategy,
                    market_data
                )
                
                # Update performance history
                self._update_performance_history(
                    strategy_name,
                    performance
                )
                
                # Check if optimization is needed
                if self._needs_optimization(strategy_name, performance):
                    await self.optimization_queue.put(strategy_name)
                    
        except Exception as e:
            logging.error(f"Performance evaluation error: {str(e)}")
            raise
            
    async def _calculate_strategy_scores(self,
                                      market_regime: 'MarketRegime',
                                      market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate scores for each strategy based on multiple factors"""
        scores = {}
        
        for name, strategy in self.strategy_pool.items():
            # Get historical performance in similar regimes
            regime_performance = self.regime_performance.get(
                (name, market_regime.name),
                0.0
            )
            
            # Get overall performance
            overall_performance = self._calculate_overall_performance(
                name
            )
            
            # Get strategy fitness for current market conditions
            fitness = await strategy.calculate_fitness(
                market_regime.condition,
                market_data
            )
            
            # Calculate final score
            scores[name] = self._calculate_final_score(
                regime_performance,
                overall_performance,
                fitness
            )
            
        return scores
    
    def _calculate_final_score(self,
                             regime_performance: float,
                             overall_performance: float,
                             fitness: float) -> float:
        """Calculate final strategy score using weighted factors"""
        weights = {
            'regime_performance': 0.4,
            'overall_performance': 0.3,
            'fitness': 0.3
        }
        
        return (
            regime_performance * weights['regime_performance'] +
            overall_performance * weights['overall_performance'] +
            fitness * weights['fitness']
        )
    
    async def _optimize_strategy_parameters(self,
                                         strategy_name: str,
                                         performance_data: Dict) -> Dict:
        """Optimize strategy parameters using bayesian optimization"""
        try:
            strategy = self.strategy_pool[strategy_name]
            config = strategy.config
            
            # Define optimization bounds
            bounds = config.optimization_bounds
            
            # Prepare optimization data
            opt_data = self._prepare_optimization_data(
                strategy_name,
                performance_data
            )
            
            # Run bayesian optimization
            optimal_params = await self._run_bayesian_optimization(
                strategy,
                opt_data,
                bounds
            )
            
            return optimal_params
            
        except Exception as e:
            logging.error(f"Parameter optimization error: {str(e)}")
            raise
            
    def _needs_optimization(self,
                          strategy_name: str,
                          performance: StrategyPerformance) -> bool:
        """Determine if strategy needs optimization"""
        # Check if performance is below thresholds
        if (performance.sharpe_ratio < self.config.MIN_SHARPE_RATIO or
            performance.profit_factor < self.config.MIN_PROFIT_FACTOR):
            return True
            
        # Check if performance is degrading
        if self._is_performance_degrading(strategy_name):
            return True
            
        # Check time since last optimization
        if self._optimization_time_exceeded(strategy_name):
            return True
            
        return False
