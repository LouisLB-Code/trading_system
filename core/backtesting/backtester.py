# File: src/core/backtesting/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from dataclasses import dataclass
import asyncio

@dataclass
class BacktestResults:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[Dict]
    positions: List[Dict]
    equity_curve: pd.Series
    metrics: Dict
    regime_performance: Dict
    strategy_performance: Dict

class Backtester:
    """Advanced backtesting system with market replay"""
    
    def __init__(self, config):
        self.config = config
        self.position_manager = BacktestPositionManager(config)
        self.risk_manager = BacktestRiskManager(config)
        self.market_data = None
        self.results = None
        
    async def run_backtest(self,
                          strategy_generator: 'AdaptiveStrategyGenerator',
                          market_data: Dict[str, pd.DataFrame],
                          initial_capital: float = 100000.0) -> BacktestResults:
        """Run full backtest simulation"""
        try:
            self.market_data = market_data
            self.current_capital = initial_capital
            self.equity_curve = [initial_capital]
            self.trades = []
            
            # Initialize components
            await self._initialize_backtest(strategy_generator)
            
            # Run market replay
            await self._run_market_replay()
            
            # Calculate results
            results = self._calculate_results()
            
            # Store results
            self.results = results
            
            return results
            
        except Exception as e:
            logging.error(f"Backtest error: {str(e)}")
            raise
    
    async def _initialize_backtest(self, strategy_generator: 'AdaptiveStrategyGenerator'):
        """Initialize backtest components"""
        try:
            # Initialize strategy generator
            self.strategy_generator = strategy_generator
            
            # Initialize market analysis
            self.regime_detector = EnhancedRegimeDetector(self.config)
            
            # Initialize execution simulator
            self.execution_simulator = BacktestExecutionSimulator(
                self.config,
                self.market_data
            )
            
            # Initialize performance tracker
            self.performance_tracker = BacktestPerformanceTracker(self.config)
            
        except Exception as e:
            logging.error(f"Backtest initialization error: {str(e)}")
            raise
    
    async def _run_market_replay(self):
        """Run market replay simulation"""
        try:
            # Get common timeindex across all symbols
            timeindex = self._get_common_timeindex()
            
            # Main simulation loop
            for timestamp in timeindex:
                # Update market data
                current_data = self._get_market_snapshot(timestamp)
                
                # Detect market regime
                regime = await self.regime_detector.detect_regime(current_data)
                
                # Generate/update strategies
                strategies = await self.strategy_generator.generate_strategy(
                    current_data,
                    regime
                )
                
                # Generate signals
                signals = await self._generate_signals(
                    strategies,
                    current_data,
                    regime
                )
                
                # Validate signals
                valid_signals = await self.risk_manager.validate_signals(
                    signals,
                    self.position_manager.positions,
                    regime
                )
                
                # Execute valid signals
                if valid_signals:
                    await self._execute_signals(
                        valid_signals,
                        timestamp
                    )
                
                # Update positions
                await self._update_positions(current_data, timestamp)
                
                # Update equity curve
                self._update_equity_curve(timestamp)
                
            # Process final results
            await self._process_results()
            
        except Exception as e:
            logging.error(f"Market replay error: {str(e)}")
            raise
    
    def _get_common_timeindex(self) -> pd.DatetimeIndex:
        """Get common timeindex across all symbols"""
        try:
            indices = [
                df.index for df in self.market_data.values()
            ]
            return pd.DatetimeIndex(
                sorted(set.intersection(*[set(idx) for idx in indices]))
            )
        except Exception as e:
            logging.error(f"Timeindex error: {str(e)}")
            raise
    
    def _get_market_snapshot(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Get market data snapshot at timestamp"""
        try:
            snapshot = {}
            for symbol, df in self.market_data.items():
                if timestamp in df.index:
                    snapshot[symbol] = df.loc[timestamp]
            return pd.DataFrame(snapshot).T
        except Exception as e:
            logging.error(f"Market snapshot error: {str(e)}")
            raise
    
    async def _generate_signals(self,
                             strategies: List['BaseStrategy'],
                             market_data: pd.DataFrame,
                             regime: 'MarketRegime') -> List[Dict]:
        """Generate trading signals from strategies"""
        signals = []
        for strategy in strategies:
            try:
                signal = await strategy.generate_signals(
                    market_data,
                    regime
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logging.error(f"Signal generation error: {str(e)}")
        return signals
    
    async def _execute_signals(self,
                            signals: List[Dict],
                            timestamp: pd.Timestamp):
        """Execute trading signals in simulation"""
        try:
            for signal in signals:
                execution = await self.execution_simulator.execute_signal(
                    signal,
                    timestamp
                )
                
                if execution:
                    # Update positions
                    await self.position_manager.process_execution(execution)
                    
                    # Record trade
                    self.trades.append({
                        **execution,
                        'timestamp': timestamp
                    })
                    
        except Exception as e:
            logging.error(f"Signal execution error: {str(e)}")
            raise
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate backtest results and metrics"""
        try:
            equity_curve = pd.Series(self.equity_curve)
            returns = equity_curve.pct_change().dropna()
            
            results = BacktestResults(
                total_return=(equity_curve[-1] / equity_curve[0]) - 1,
                sharpe_ratio=self._calculate_sharpe_ratio(returns),
                max_drawdown=self._calculate_max_drawdown(equity_curve),
                win_rate=self._calculate_win_rate(self.trades),
                profit_factor=self._calculate_profit_factor(self.trades),
                trades=self.trades,
                positions=self.position_manager.position_history,
                equity_curve=equity_curve,
                metrics=self._calculate_additional_metrics(returns),
                regime_performance=self._calculate_regime_performance(),
                strategy_performance=self._calculate_strategy_performance()
            )
            
            return results
            
        except Exception as e:
            logging.error(f"Results calculation error: {str(e)}")
            raise
            
    def _calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
        return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())
