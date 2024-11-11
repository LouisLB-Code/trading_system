# File: src/core/monitoring/monitor.py

from typing import Dict, List, Optional, Union, Callable
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..event_system import EventBus, Event, EventTypes
from ..metrics.metrics_collector import MetricsCollector

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class Alert:
    level: AlertLevel
    source: str
    message: str
    details: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class MonitoringSystem:
    """Real-time system monitoring and alerting"""
    
    def __init__(self, config, event_bus: EventBus, metrics_collector: MetricsCollector):
        self.config = config
        self.event_bus = event_bus
        self.metrics_collector = metrics_collector
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.monitoring_tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        try:
            # Start system monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_system_metrics())
            )
            
            # Start performance monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_performance_metrics())
            )
            
            # Start strategy monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_strategy_metrics())
            )
            
            # Start resource monitoring
            self.monitoring_tasks.append(
                asyncio.create_task(self._monitor_resources())
            )
            
            self.logger.info("Monitoring system started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring system: {str(e)}")
            raise
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
    
    def register_alert_handler(self, level: AlertLevel, handler: Callable):
        """Register handler for specific alert level"""
        self.alert_handlers[level].append(handler)
    
    async def _monitor_system_metrics(self):
        """Monitor system health metrics"""
        while True:
            try:
                metrics = self.metrics_collector.system_metrics
                
                # Check CPU usage
                if metrics.cpu_usage and np.mean(metrics.cpu_usage[-5:]) > self.config.CPU_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "System",
                        "High CPU usage detected",
                        {'cpu_usage': metrics.cpu_usage[-5:]}
                    )
                
                # Check memory usage
                if metrics.memory_usage and np.mean(metrics.memory_usage[-5:]) > self.config.MEMORY_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "System",
                        "High memory usage detected",
                        {'memory_usage': metrics.memory_usage[-5:]}
                    )
                
                # Check latency
                if metrics.latency_ms and np.mean(metrics.latency_ms[-5:]) > self.config.LATENCY_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "System",
                        "High latency detected",
                        {'latency': metrics.latency_ms[-5:]}
                    )
                
                await asyncio.sleep(self.config.SYSTEM_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitoring error: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _monitor_performance_metrics(self):
        """Monitor trading performance metrics"""
        while True:
            try:
                metrics = self.metrics_collector.performance_metrics
                
                # Check drawdown
                if metrics.max_drawdown > self.config.MAX_DRAWDOWN_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        "Performance",
                        "Maximum drawdown threshold exceeded",
                        {'max_drawdown': metrics.max_drawdown}
                    )
                
                # Check win rate
                if metrics.win_rate < self.config.MIN_WIN_RATE:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Performance",
                        "Win rate below threshold",
                        {'win_rate': metrics.win_rate}
                    )
                
                await asyncio.sleep(self.config.PERFORMANCE_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _monitor_strategy_metrics(self):
        """Monitor strategy-specific metrics"""
        while True:
            try:
                for strategy_name, metrics in self.metrics_collector.strategy_metrics.items():
                    # Check execution rate
                    execution_rate = metrics.signals_executed / max(1, metrics.signals_generated)
                    if execution_rate < self.config.MIN_EXECUTION_RATE:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            f"Strategy_{strategy_name}",
                            "Low signal execution rate",
                            {'execution_rate': execution_rate}
                        )
                    
                    # Check success rate
                    success_rate = metrics.successful_trades / max(1, metrics.successful_trades + metrics.failed_trades)
                    if success_rate < self.config.MIN_STRATEGY_SUCCESS_RATE:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            f"Strategy_{strategy_name}",
                            "Low strategy success rate",
                            {'success_rate': success_rate}
                        )
                
                await asyncio.sleep(self.config.STRATEGY_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Strategy monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _monitor_resources(self):
        """Monitor system resources"""
        while True:
            try:
                # Check disk space
                if self._check_disk_space() < self.config.MIN_DISK_SPACE:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Resources",
                        "Low disk space",
                        {'available_space': self._check_disk_space()}
                    )
                
                # Check memory availability
                if self._check_memory() < self.config.MIN_MEMORY:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Resources",
                        "Low memory availability",
                        {'available_memory': self._check_memory()}
                    )
                
                await asyncio.sleep(self.config.RESOURCE_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _create_alert(self, level: AlertLevel, source: str, message: str, details: Dict):
        """Create and handle new alert"""
        try:
            # Create alert
            alert = Alert(
                level=level,
                source=source,
                message=message,
                details=details
            )
            
            # Add to active alerts
            self.active_alerts.append(alert)
            
            # Notify handlers
            for handler in self.alert_handlers[level]:
                await handler(alert)
            
            # Publish alert event
            await self.event_bus.publish(Event(
                type=EventTypes.RISK_ALERT,
                data={
                    'level': level.value,
                    'source': source,
                    'message': message,
                    'details': details
                }
            ))
            
            # Log alert
            self.logger.warning(f"Alert: {message} from {source}")
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")
    
    def _check_disk_space(self) -> float:
        """Check available disk space"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent
        except:
            return 0.0
    
    def _check_memory(self) -> float:
        """Check available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.available / memory.total * 100
        except:
            return 0.0
