# File: src/core/state_management.py

from typing import Dict, Any, Optional, List
import asyncio
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from .event_system import EventBus, Event, EventTypes

class SystemState(Enum):
    """System state enumeration"""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    MAINTENANCE = "MAINTENANCE"

@dataclass
class TradingState:
    """Trading state information"""
    active_positions: Dict[str, Dict] = field(default_factory=dict)
    pending_orders: Dict[str, Dict] = field(default_factory=dict)
    available_capital: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    risk_metrics: Dict = field(default_factory=dict)

@dataclass
class MarketState:
    """Market state information"""
    current_regime: Optional[str] = None
    regime_confidence: float = 0.0
    market_conditions: Dict = field(default_factory=dict)
    volatility_state: Dict = field(default_factory=dict)
    liquidity_state: Dict = field(default_factory=dict)
    technical_indicators: Dict = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    event_queue_size: int = 0
    processing_latency: float = 0.0
    error_count: int = 0
    warning_count: int = 0

class StateManager:
    """Manages system-wide state"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.system_state = SystemState.INITIALIZING
        self.trading_state = TradingState()
        self.market_state = MarketState()
        self.system_metrics = SystemMetrics()
        self.state_history: List[Dict] = []
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Subscribe to relevant events
        asyncio.create_task(self._subscribe_to_events())
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant system events"""
        await self.event_bus.subscribe(EventTypes.MARKET_UPDATE, self._handle_market_update)
        await self.event_bus.subscribe(EventTypes.MARKET_REGIME_CHANGE, self._handle_regime_change)
        await self.event_bus.subscribe(EventTypes.POSITION_OPENED, self._handle_position_update)
        await self.event_bus.subscribe(EventTypes.POSITION_CLOSED, self._handle_position_update)
        await self.event_bus.subscribe(EventTypes.ORDER_FILLED, self._handle_order_update)
        await self.event_bus.subscribe(EventTypes.SYSTEM_ERROR, self._handle_system_error)
    
    async def update_system_state(self, new_state: SystemState):
        """Update system state"""
        try:
            async with self._lock:
                old_state = self.system_state
                self.system_state = new_state
                
                # Record state change
                self._record_state_change(old_state, new_state)
                
                # Publish state change event
                await self.event_bus.publish(Event(
                    type="SYSTEM_STATE_CHANGE",
                    data={
                        'old_state': old_state,
                        'new_state': new_state,
                        'timestamp': datetime.now()
                    }
                ))
                
        except Exception as e:
            self._logger.error(f"Error updating system state: {str(e)}")
            raise
    
    async def update_trading_state(self, updates: Dict[str, Any]):
        """Update trading state"""
        try:
            async with self._lock:
                for key, value in updates.items():
                    if hasattr(self.trading_state, key):
                        setattr(self.trading_state, key, value)
                
                # Record state update
                self._record_state_update('trading', updates)
                
        except Exception as e:
            self._logger.error(f"Error updating trading state: {str(e)}")
            raise
    
    async def update_market_state(self, updates: Dict[str, Any]):
        """Update market state"""
        try:
            async with self._lock:
                for key, value in updates.items():
                    if hasattr(self.market_state, key):
                        setattr(self.market_state, key, value)
                
                # Record state update
                self._record_state_update('market', updates)
                
        except Exception as e:
            self._logger.error(f"Error updating market state: {str(e)}")
            raise
    
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update system metrics"""
        try:
            async with self._lock:
                for key, value in metrics.items():
                    if hasattr(self.system_metrics, key):
                        setattr(self.system_metrics, key, value)
                
                # Record metrics update
                self._record_state_update('metrics', metrics)
                
        except Exception as e:
            self._logger.error(f"Error updating system metrics: {str(e)}")
            raise
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get complete current state"""
        return {
            'system_state': self.system_state,
            'trading_state': asdict(self.trading_state),
            'market_state': asdict(self.market_state),
            'system_metrics': asdict(self.system_metrics),
            'timestamp': datetime.now()
        }
    
    async def _handle_market_update(self, event: Event):
        """Handle market update events"""
        await self.update_market_state(event.data)
    
    async def _handle_regime_change(self, event: Event):
        """Handle market regime change events"""
        await self.update_market_state({
            'current_regime': event.data['regime'],
            'regime_confidence': event.data['confidence']
        })
    
    async def _handle_position_update(self, event: Event):
        """Handle position update events"""
        await self.update_trading_state({
            'active_positions': event.data['positions']
        })
    
    async def _handle_order_update(self, event: Event):
        """Handle order update events"""
        await self.update_trading_state({
            'pending_orders': event.data['orders']
        })
    
    async def _handle_system_error(self, event: Event):
        """Handle system error events"""
        if self.system_metrics.error_count >= self.system_metrics.error_count:
            await self.update_system_state(SystemState.ERROR)
    
    def _record_state_change(self, old_state: SystemState, new_state: SystemState):
        """Record state change in history"""
        self.state_history.append({
            'timestamp': datetime.now(),
            'type': 'state_change',
            'old_state': old_state,
            'new_state': new_state
        })
    
    def _record_state_update(self, state_type: str, updates: Dict):
        """Record state update in history"""
        self.state_history.append({
            'timestamp': datetime.now(),
            'type': 'state_update',
            'state_type': state_type,
            'updates': updates
        })
