# File: src/core/enhanced_system.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass

from .event_system import EventBus, Event, EventTypes, MarketUpdateEvent, SignalEvent
from .state_management import StateManager, SystemState, TradingState, MarketState
from .meta_learning import MetaLearningNetwork
from .error_handling import AdvancedErrorHandler
from .market_analysis import EnhancedRegimeDetector, MarketConditionAnalyzer
from .strategy import StrategyManager, AdaptiveStrategyGenerator
from .risk import EnhancedRiskManager
from .data import DataManager

@dataclass
class MarketRegime:
    name: str
    volatility: float
    trend_strength: float
    liquidity: float
    correlation: float
    risk_level: float

@dataclass
class SystemState:
    capital: float
    positions: Dict
    current_risk: float
    performance_metrics: Dict
    market_regime: MarketRegime
    resource_usage: Dict
    adaptation_metrics: Dict

class EnhancedTradingSystem:
    """Advanced trading system with self-learning capabilities and event-driven architecture"""
    
    def __init__(self, config):
        # Initialize core components
        self.config = config
        self.event_bus = EventBus()
        self.state_manager = StateManager(self.event_bus)
        self.error_handler = AdvancedErrorHandler()
        
        # Initialize system components
        self.data_manager = DataManager(config)
        self.regime_detector = EnhancedRegimeDetector(config)
        self.condition_analyzer = MarketConditionAnalyzer(config)
        self.risk_manager = EnhancedRiskManager(config)
        self.strategy_manager = StrategyManager(config)
        self.meta_learner = MetaLearningNetwork()
        
        # Initialize system state
        self.state = SystemState(
            capital=config.INITIAL_CAPITAL,
            positions={},
            current_risk=0.0,
            performance_metrics={},
            market_regime=None,
            resource_usage={},
            adaptation_metrics={}
        )
        
        # Initialize monitoring
        self.logger = logging.getLogger(__name__)
        self._initialize_monitoring()
    
    async def initialize(self):
        """Initialize the enhanced system"""
        try:
            # Update system state
            await self.state_manager.update_system_state(SystemState.INITIALIZING)
            
            # Initialize neural architecture
            architecture = await self.neural_architect.search_architecture(
                self.config.ARCHITECTURE_PARAMS
            )
            
            # Initialize models with optimal architecture
            await self.model_optimizer.initialize_models(architecture)
            
            # Initialize data systems
            await self.data_manager.initialize()
            
            # Initialize analysis components
            await self._initialize_analysis_components()
            
            # Initialize strategy components
            await self._initialize_strategy_components()
            
            # Initialize risk management
            await self._initialize_risk_components()
            
            # Subscribe to events
            await self._subscribe_to_events()
            
            # Update system state
            await self.state_manager.update_system_state(SystemState.RUNNING)
            
            self.logger.info("Enhanced trading system initialized successfully")
            
        except Exception as e:
            await self.error_handler.handle_error(e, "System Initialization", {})
            await self.state_manager.update_system_state(SystemState.ERROR)
            raise
    
    async def process_market_update(self, market_data: Dict) -> Dict:
        """Process new market data with advanced analysis"""
        try:
            # Publish market update event
            await self.event_bus.publish(
                MarketUpdateEvent(
                    symbol=market_data['symbol'],
                    data=market_data
                )
            )
            
            # Detect market regime
            regime = await self.regime_detector.detect_regime(market_data)
            self.state.market_regime = regime
            
            await self.state_manager.update_market_state({
                'current_regime': regime.name,
                'regime_confidence': regime.confidence
            })
            
            # Recognize patterns
            patterns = await self.pattern_recognizer.analyze_patterns(
                market_data,
                regime
            )
            
            # Optimize resources
            resources = await self.resource_optimizer.optimize_resources(
                self.state.resource_usage,
                patterns
            )
            
            # Generate trading decisions
            decisions = await self._generate_trading_decisions(
                patterns,
                regime,
                resources
            )
            
            # Adapt system
            adaptation = await self.adaptation_engine.adapt_system(
                decisions,
                patterns,
                regime
            )
            
            # Update state
            self._update_system_state(
                decisions,
                adaptation,
                resources
            )
            
            # Update system metrics
            await self._update_system_metrics()
            
            return {
                'decisions': decisions,
                'patterns': patterns,
                'regime': regime,
                'adaptation': adaptation,
                'resources': resources,
                'state': self.state_manager.get_current_state()
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Market Update Processing", {})
            return {}
            
    async def _generate_trading_decisions(self,
                                       patterns: Dict,
                                       regime: MarketRegime,
                                       resources: Dict) -> Dict:
        """Generate trading decisions"""
        try:
            # Get meta-learning insights
            insights = await self.meta_learner.analyze(
                patterns,
                regime,
                self.state
            )
            
            # Generate base decisions
            base_decisions = await self._generate_base_decisions(
                patterns,
                regime
            )
            
            # Apply meta-learning adjustments
            adjusted_decisions = self.meta_learner.adjust_decisions(
                base_decisions,
                insights
            )
            
            # Validate with risk management
            validated_decisions = await self.risk_manager.validate_decisions(
                adjusted_decisions,
                self.state
            )
            
            return {
                'decisions': validated_decisions,
                'meta_insights': insights,
                'risk_metrics': self.risk_manager.get_current_metrics()
            }
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Decision Generation", {})
            return {}
    
    def _update_system_state(self,
                           decisions: Dict,
                           adaptation: Dict,
                           resources: Dict):
        """Update system state with new information"""
        # Update basic state
        self.state.resource_usage = resources
        self.state.adaptation_metrics = adaptation
        
        # Update performance metrics
        self.performance_tracker.update_metrics(decisions)
        
        # Update adaptation tracking
        self.adaptation_tracker.track_adaptation(adaptation)
        
        # Update risk metrics
        self.risk_manager.update_risk_metrics(decisions)
    
    async def _initialize_analysis_components(self):
        """Initialize market analysis components"""
        try:
            # Initialize regime detection
            await self.regime_detector.initialize()
            
            # Initialize condition analysis
            await self.condition_analyzer.initialize()
            
            # Initialize meta-learning
            await self.meta_learner.initialize()
            
        except Exception as e:
            self.logger.error(f"Analysis component initialization error: {str(e)}")
            raise
    
    async def _initialize_strategy_components(self):
        """Initialize strategy components"""
        try:
            # Initialize strategy manager
            await self.strategy_manager.initialize_strategies()
            
            # Initialize strategy generation
            await self.meta_learner.initialize_models()
            
        except Exception as e:
            self.logger.error(f"Strategy component initialization error: {str(e)}")
            raise
    
    async def _initialize_risk_components(self):
        """Initialize risk management components"""
        try:
            # Initialize risk manager
            await self.risk_manager.initialize()
            
            # Set initial risk limits
            await self.state_manager.update_trading_state({
                'risk_metrics': await self.risk_manager.get_initial_metrics()
            })
            
        except Exception as e:
            self.logger.error(f"Risk component initialization error: {str(e)}")
            raise
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant system events"""
        # Market events
        await self.event_bus.subscribe(EventTypes.MARKET_UPDATE, self._handle_market_update)
        await self.event_bus.subscribe(EventTypes.MARKET_REGIME_CHANGE, self._handle_regime_change)
        
        # Trading events
        await self.event_bus.subscribe(EventTypes.SIGNAL_GENERATED, self._handle_trading_signal)
        await self.event_bus.subscribe(EventTypes.ORDER_FILLED, self._handle_order_update)
        
        # System events
        await self.event_bus.subscribe(EventTypes.SYSTEM_ERROR, self._handle_system_error)
        await self.event_bus.subscribe(EventTypes.RISK_ALERT, self._handle_risk_alert)
    
    async def _handle_market_update(self, event: Event):
        """Handle market update events"""
        try:
            # Update internal state
            market_data = event.data['data']
            symbol = event.data['symbol']
            
            # Update operation metrics
            self.operation_metrics['processed_updates'] += 1
            
            # Process market update
            start_time = datetime.now()
            await self.process_market_update(market_data)
            
            # Record execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            self.operation_metrics['execution_times'].append(execution_time)
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Market Update Handler", event.data)
    
    async def _handle_regime_change(self, event: Event):
        """Handle market regime change events"""
        try:
            # Update strategy selection
            await self.strategy_manager.select_strategies(
                event.data['regime'],
                self.state_manager.market_state.market_conditions
            )
            
            # Update risk parameters
            await self.risk_manager.adapt_risk_params(event.data['regime'])
            
            # Notify meta-learner
            await self.meta_learner.adapt(event.data)
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Regime Change Handler", event.data)
    
    async def _handle_trading_signal(self, event: Event):
        """Handle trading signal events"""
        try:
            # Update metrics
            self.operation_metrics['generated_signals'] += 1
            
            # Validate signal
            validated_signal = await self.risk_manager.validate_signal(event.data['signal'])
            
            if validated_signal:
                # Update trading state
                await self.state_manager.update_trading_state({
                    'pending_signals': validated_signal
                })
                
                # Notify strategy manager
                await self.strategy_manager.process_signal(validated_signal)
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Signal Handler", event.data)
    
    async def _handle_order_update(self, event: Event):
        """Handle order update events"""
        try:
            # Update trading state
            await self.state_manager.update_trading_state({
                'pending_orders': event.data
            })
            
            # Update risk metrics
            await self.risk_manager.update_position_risk(event.data)
            
            # Notify strategy manager
            await self.strategy_manager.process_order_update(event.data)
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Order Update Handler", event.data)
    
    async def _handle_system_error(self, event: Event):
        """Handle system error events"""
        try:
            error_data = event.data
            
            # Update system metrics
            await self.state_manager.update_metrics({
                'error_count': self.state_manager.system_metrics.error_count + 1
            })
            
            # Check error threshold
            if self.state_manager.system_metrics.error_count >= self.config.MAX_ERROR_COUNT:
                await self.state_manager.update_system_state(SystemState.ERROR)
                
            # Log error
            self.logger.error(f"System error: {error_data['message']}")
            
        except Exception as e:
            self.logger.critical(f"Error handler failed: {str(e)}")
    
    async def _handle_risk_alert(self, event: Event):
        """Handle risk alert events"""
        try:
            alert_data = event.data
            
            # Update risk metrics
            await self.state_manager.update_trading_state({
                'risk_metrics': alert_data
            })
            
            # Check risk thresholds
            if alert_data['alert_type'] == 'critical':
                await self.state_manager.update_system_state(SystemState.PAUSED)
                
            # Log alert
            self.logger.warning(f"Risk alert: {alert_data['details']}")
            
        except Exception as e:
            await self.error_handler.handle_error(e, "Risk Alert Handler", event.data)
    
    def _initialize_monitoring(self):
        """Initialize system monitoring"""
        self.start_time = datetime.now()
        self.operation_metrics = {
            'processed_updates': 0,
            'generated_signals': 0,
            'execution_times': []
        }
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        try:
            metrics = {
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'event_queue_size': len(self.event_bus._subscribers),
                'processing_latency': np.mean(self.operation_metrics['execution_times'][-100:])
            }
            
            await self.state_manager.update_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {str(e)}")
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        try:
            import psutil
            return psutil.Process().memory_percent()
        except:
            return 0.0
    
    async def run_diagnostics(self) -> Dict:
        """Run system diagnostics"""
        return {
            'system_state': self.state_manager.get_current_state(),
            'performance_metrics': self.operation_metrics,
            'component_status': {
                'data_manager': await self.data_manager.get_status(),
                'strategy_manager': await self.strategy_manager.get_status(),
                'risk_manager': await self.risk_manager.get_status()
            }
        }
