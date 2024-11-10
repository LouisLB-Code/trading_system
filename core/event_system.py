# File: src/core/event_system.py

from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
import weakref

@dataclass
class Event:
    """Base event class"""
    type: str
    data: Any
    timestamp: datetime = datetime.now()
    source: Optional[str] = None
    id: Optional[str] = None

class EventBus:
    """Central event bus for system-wide communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[weakref.WeakMethod]] = {}
        self._logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
        
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        try:
            async with self._lock:
                if event.type in self._subscribers:
                    # Get active subscribers
                    active_subscribers = []
                    for subscriber_ref in self._subscribers[event.type]:
                        subscriber = subscriber_ref()
                        if subscriber is not None:
                            active_subscribers.append(subscriber)
                    
                    # Remove dead references
                    self._subscribers[event.type] = [
                        sub for sub in self._subscribers[event.type]
                        if sub() is not None
                    ]
                    
                    # Notify subscribers
                    tasks = [
                        self._notify_subscriber(subscriber, event)
                        for subscriber in active_subscribers
                    ]
                    
                    if tasks:
                        await asyncio.gather(*tasks)
                        
        except Exception as e:
            self._logger.error(f"Error publishing event {event.type}: {str(e)}")
            raise
    
    async def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        try:
            async with self._lock:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                
                # Store weak reference to callback
                self._subscribers[event_type].append(weakref.WeakMethod(callback))
                
        except Exception as e:
            self._logger.error(f"Error subscribing to {event_type}: {str(e)}")
            raise
    
    async def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type"""
        try:
            async with self._lock:
                if event_type in self._subscribers:
                    self._subscribers[event_type] = [
                        sub for sub in self._subscribers[event_type]
                        if sub() is not callback
                    ]
                    
        except Exception as e:
            self._logger.error(f"Error unsubscribing from {event_type}: {str(e)}")
            raise
    
    async def _notify_subscriber(self, subscriber: Callable, event: Event):
        """Notify a single subscriber"""
        try:
            await subscriber(event)
        except Exception as e:
            self._logger.error(f"Error notifying subscriber for {event.type}: {str(e)}")

# Event types
class EventTypes:
    """Enum of system event types"""
    # Market events
    MARKET_UPDATE = "MARKET_UPDATE"
    MARKET_REGIME_CHANGE = "MARKET_REGIME_CHANGE"
    MARKET_CONDITION_UPDATE = "MARKET_CONDITION_UPDATE"
    
    # Trading events
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    
    # System events
    STRATEGY_UPDATED = "STRATEGY_UPDATED"
    RISK_ALERT = "RISK_ALERT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    PERFORMANCE_UPDATE = "PERFORMANCE_UPDATE"

# Event data classes
@dataclass
class MarketUpdateEvent(Event):
    """Market data update event"""
    def __init__(self, symbol: str, data: Dict):
        super().__init__(
            type=EventTypes.MARKET_UPDATE,
            data={
                'symbol': symbol,
                'data': data
            }
        )

@dataclass
class SignalEvent(Event):
    """Trading signal event"""
    def __init__(self, strategy: str, signal: Dict):
        super().__init__(
            type=EventTypes.SIGNAL_GENERATED,
            data={
                'strategy': strategy,
                'signal': signal
            }
        )

@dataclass
class OrderEvent(Event):
    """Order-related event"""
    def __init__(self, event_type: str, order: Dict):
        super().__init__(
            type=event_type,
            data=order
        )

@dataclass
class RiskAlert(Event):
    """Risk alert event"""
    def __init__(self, alert_type: str, details: Dict):
        super().__init__(
            type=EventTypes.RISK_ALERT,
            data={
                'alert_type': alert_type,
                'details': details
            }
        )
