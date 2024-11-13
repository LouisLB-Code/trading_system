```python
from typing import Dict, List, Optional, Callable, Set
import asyncio
import logging
import psutil
import gc
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import socket
import resource
from collections import defaultdict

from ..event_system import EventBus, Event, EventTypes
from .collectors import MetricsCollector, SystemMetrics, TradingMetrics, PerformanceMetrics
from .handlers import AlertHandlerManager
from .recovery import RecoverySystem
from ..logging_config import get_logger

# Enhanced enums and data classes
class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class ResourceType(Enum):
    CPU = "CPU"
    MEMORY = "MEMORY"
    DISK = "DISK"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    TASK = "TASK"

@dataclass
class Alert:
    level: AlertLevel
    source: str
    message: str
    details: Dict
    resource_type: Optional[ResourceType] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    io_read_bytes: int
    io_write_bytes: int
    open_file_descriptors: int
    thread_count: int
    process_time: float
    system_load: List[float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class NetworkMetrics:
    latency: float
    packet_loss: float
    bandwidth_usage: float
    connection_count: int
    error_count: int
    retry_count: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MemoryMetrics:
    total_allocated: int
    peak_allocated: int
    gc_count: List[int]
    gc_threshold: List[int]
    uncollectable_objects: int
    object_counts: Dict[str, int]
    memory_growth_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TaskMetrics:
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_duration: float
    blocked_tasks: int
    queue_sizes: Dict[str, int]
    long_running_tasks: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class MonitoringSystem:
    """Enhanced real-time system monitoring and alerting"""
    
    def __init__(self, config, event_bus: EventBus, trading_system):
        self.config = config
        self.event_bus = event_bus
        self.trading_system = trading_system
        
        # Initialize components
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertHandlerManager(config)
        self.recovery_system = RecoverySystem(config, trading_system, event_bus)
        
        # Initialize state
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.monitoring_tasks: List[asyncio.Task] = []
        self.recovering: bool = False
        
        # Setup logging
        self.logger = get_logger('monitoring')
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        
        # Enhanced monitoring state
        self.resource_metrics_history: List[ResourceMetrics] = []
        self.network_metrics_history: List[NetworkMetrics] = []
        self.memory_metrics_history: List[MemoryMetrics] = []
        self.task_metrics_history: List[TaskMetrics] = []
        self.blocked_tasks: Set[str] = set()
        self.long_running_tasks: Dict[str, datetime] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_gc_collection = 0
        self.baseline_memory_usage = self._get_process_memory()
        self.baseline_object_counts = self._get_object_counts()
        
        # Alert rate limiting
        self.alert_counts: Dict[str, int] = {}
        self.last_alert_cleanup = datetime.now()
        
        # Network monitoring
        self.connection_history: Dict[str, List[datetime]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_network_check = defaultdict(datetime.now)
```
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        try:
            self.monitoring_tasks.extend([
                # Core monitoring
                asyncio.create_task(self._monitor_system_metrics()),
                asyncio.create_task(self._monitor_performance_metrics()),
                asyncio.create_task(self._monitor_strategy_metrics()),
                asyncio.create_task(self._monitor_resources()),
                
                # Enhanced monitoring
                asyncio.create_task(self._monitor_memory_usage()),
                asyncio.create_task(self._monitor_network_connectivity()),
                asyncio.create_task(self._monitor_tasks()),
                asyncio.create_task(self._monitor_system_health()),
                
                # Maintenance tasks
                asyncio.create_task(self._cleanup_old_alerts()),
                asyncio.create_task(self._cleanup_alert_counts()),
                asyncio.create_task(self._perform_gc_maintenance())
            ])
            
            self.logger.info_with_context(
                "Enhanced monitoring system started successfully",
                task_count=len(self.monitoring_tasks),
                monitoring_types=["system", "performance", "memory", "network", "tasks", "health"]
            )
            
        except Exception as e:
            self.logger.error_with_context(
                "Error starting monitoring system",
                error=str(e),
                traceback=True
            )
            raise

    async def _monitor_memory_usage(self):
        """Monitor memory usage and detect memory leaks"""
        while True:
            try:
                current_metrics = self._collect_memory_metrics()
                self.memory_metrics_history.append(current_metrics)
                
                # Trim history to configured length
                if len(self.memory_metrics_history) > self.config.METRICS_HISTORY_LENGTH:
                    self.memory_metrics_history = self.memory_metrics_history[-self.config.METRICS_HISTORY_LENGTH:]
                
                # Analyze memory growth
                if len(self.memory_metrics_history) >= 10:
                    growth_rate = self._calculate_memory_growth_rate()
                    if growth_rate > self.config.MEMORY_GROWTH_THRESHOLD:
                        leaked_objects = self._find_leaked_objects()
                        await self._create_alert(
                            AlertLevel.WARNING,
                            "Memory",
                            "Potential memory leak detected",
                            {
                                'growth_rate': growth_rate,
                                'leaked_objects': leaked_objects,
                                'current_usage': current_metrics.total_allocated,
                                'peak_usage': current_metrics.peak_allocated
                            },
                            ResourceType.MEMORY
                        )
                
                # Check for memory pressure
                if current_metrics.total_allocated > self.config.MEMORY_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        "Memory",
                        "High memory usage detected",
                        {
                            'current_usage': current_metrics.total_allocated,
                            'threshold': self.config.MEMORY_THRESHOLD,
                            'gc_stats': current_metrics.gc_count
                        },
                        ResourceType.MEMORY
                    )
                
                await asyncio.sleep(self.config.MEMORY_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error_with_context(
                    "Memory monitoring error",
                    error=str(e),
                    traceback=True
                )
                await asyncio.sleep(5)

    def _collect_memory_metrics(self) -> MemoryMetrics:
        """Collect comprehensive memory metrics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get garbage collection statistics
            gc_counts = gc.get_count()
            gc_thresholds = gc.get_threshold()
            
            # Get object counts
            current_objects = self._get_object_counts()
            
            # Calculate memory growth rate
            current_usage = memory_info.rss
            growth_rate = self._calculate_memory_growth_rate()
            
            return MemoryMetrics(
                total_allocated=current_usage,
                peak_allocated=memory_info.peak_wset,
                gc_count=list(gc_counts),
                gc_threshold=list(gc_thresholds),
                uncollectable_objects=len(gc.garbage),
                object_counts=current_objects,
                memory_growth_rate=growth_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error_with_context(
                "Error collecting memory metrics",
                error=str(e),
                traceback=True
            )
            return None

    def _get_object_counts(self) -> Dict[str, int]:
        """Get counts of Python objects by type"""
        counts: Dict[str, int] = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1
        return counts

    def _find_leaked_objects(self) -> Dict[str, int]:
        """Find potentially leaked objects by comparing with baseline"""
        current_counts = self._get_object_counts()
        leaked_objects = {}
        
        for obj_type, count in current_counts.items():
            baseline = self.baseline_object_counts.get(obj_type, 0)
            if count > baseline * self.config.OBJECT_COUNT_THRESHOLD:
                leaked_objects[obj_type] = count - baseline
                
        return leaked_objects

    def _calculate_memory_growth_rate(self) -> float:
        """Calculate memory growth rate from history"""
        if len(self.memory_metrics_history) < 2:
            return 0.0
            
        recent_metrics = self.memory_metrics_history[-10:]
        initial_memory = recent_metrics[0].total_allocated
        final_memory = recent_metrics[-1].total_allocated
        time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
        
        if time_diff == 0:
            return 0.0
            
        return (final_memory - initial_memory) / time_diff

    async def _perform_gc_maintenance(self):
        """Perform garbage collection maintenance"""
        while True:
            try:
                # Check if GC is needed
                current_memory = self._get_process_memory()
                memory_threshold_exceeded = current_memory > self.config.MEMORY_THRESHOLD
                time_threshold_exceeded = (
                    time.time() - self.last_gc_collection > 
                    self.config.GC_INTERVAL_SECONDS
                )
                
                if memory_threshold_exceeded or time_threshold_exceeded:
                    self.logger.info_with_context(
                        "Initiating garbage collection",
                        memory_usage=current_memory,
                        last_gc=self.last_gc_collection
                    )
                    
                    # Perform garbage collection
                    gc.collect()
                    self.last_gc_collection = time.time()
                    
                    # Update metrics
                    new_memory = self._get_process_memory()
                    memory_freed = current_memory - new_memory
                    
                    self.logger.info_with_context(
                        "Garbage collection completed",
                        memory_freed=memory_freed,
                        current_memory=new_memory
                    )
                
                await asyncio.sleep(self.config.GC_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error_with_context(
                    "GC maintenance error",
                    error=str(e),
                    traceback=True
                )
                await asyncio.sleep(5)

    def _get_process_memory(self) -> int:
        """Get current process memory usage"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0
    async def _monitor_network_connectivity(self):
        """Monitor network connectivity and performance"""
        while True:
            try:
                network_metrics = await self._collect_network_metrics()
                self.network_metrics_history.append(network_metrics)
                
                # Trim history
                if len(self.network_metrics_history) > self.config.METRICS_HISTORY_LENGTH:
                    self.network_metrics_history = self.network_metrics_history[-self.config.METRICS_HISTORY_LENGTH:]
                
                # Check latency
                if network_metrics.latency > self.config.MAX_LATENCY_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Network",
                        "High network latency detected",
                        {
                            'latency': network_metrics.latency,
                            'threshold': self.config.MAX_LATENCY_THRESHOLD,
                            'packet_loss': network_metrics.packet_loss
                        },
                        ResourceType.NETWORK
                    )
                
                # Check packet loss
                if network_metrics.packet_loss > self.config.MAX_PACKET_LOSS_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        "Network",
                        "High packet loss detected",
                        {
                            'packet_loss': network_metrics.packet_loss,
                            'threshold': self.config.MAX_PACKET_LOSS_THRESHOLD
                        },
                        ResourceType.NETWORK
                    )
                
                # Check connection stability
                await self._check_exchange_connectivity()
                
                # Check bandwidth usage
                if network_metrics.bandwidth_usage > self.config.MAX_BANDWIDTH_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Network",
                        "High bandwidth usage detected",
                        {
                            'bandwidth_usage': network_metrics.bandwidth_usage,
                            'threshold': self.config.MAX_BANDWIDTH_THRESHOLD
                        },
                        ResourceType.NETWORK
                    )
                
                await asyncio.sleep(self.config.NETWORK_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error_with_context(
                    "Network monitoring error",
                    error=str(e),
                    traceback=True
                )
                await asyncio.sleep(5)

    async def _collect_network_metrics(self) -> NetworkMetrics:
        """Collect comprehensive network metrics"""
        try:
            # Get network counters
            net_io = psutil.net_io_counters()
            
            # Measure latency to exchanges
            latencies = await self._measure_exchange_latencies()
            avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0
            
            # Calculate packet loss
            packet_loss = await self._measure_packet_loss()
            
            # Get connection count
            connections = len(psutil.net_connections())
            
            # Calculate bandwidth usage
            bandwidth_usage = self._calculate_bandwidth_usage(net_io)
            
            return NetworkMetrics(
                latency=avg_latency,
                packet_loss=packet_loss,
                bandwidth_usage=bandwidth_usage,
                connection_count=connections,
                error_count=sum(self.error_counts.values()),
                retry_count=self._get_retry_count(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error_with_context(
                "Error collecting network metrics",
                error=str(e),
                traceback=True
            )
            return None

    async def _measure_exchange_latencies(self) -> Dict[str, float]:
        """Measure latency to all configured exchanges"""
        latencies = {}
        for exchange_name, exchange_config in self.config.EXCHANGES.items():
            try:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(exchange_config['ping_endpoint']) as response:
                        if response.status == 200:
                            latency = (time.time() - start_time) * 1000  # Convert to ms
                            latencies[exchange_name] = latency
                        else:
                            self.error_counts[exchange_name] += 1
                            self.logger.warning_with_context(
                                "Exchange ping failed",
                                exchange=exchange_name,
                                status=response.status
                            )
            except Exception as e:
                self.error_counts[exchange_name] += 1
                self.logger.error_with_context(
                    "Exchange latency measurement error",
                    exchange=exchange_name,
                    error=str(e)
                )
        
        return latencies

    async def _measure_packet_loss(self) -> float:
        """Measure packet loss to exchanges"""
        try:
            total_packets = 10
            successful_packets = 0
            
            for exchange_config in self.config.EXCHANGES.values():
                hostname = exchange_config['hostname']
                for _ in range(total_packets):
                    if await self._ping_host(hostname):
                        successful_packets += 1
                        
            return 1 - (successful_packets / (total_packets * len(self.config.EXCHANGES)))
            
        except Exception as e:
            self.logger.error_with_context(
                "Packet loss measurement error",
                error=str(e)
            )
            return 0.0

    async def _ping_host(self, hostname: str) -> bool:
        """Ping a host and return success/failure"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1', hostname,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    def _calculate_bandwidth_usage(self, net_io: psutil._common.snetio) -> float:
        """Calculate current bandwidth usage in MB/s"""
        try:
            if not hasattr(self, '_last_net_io'):
                self._last_net_io = net_io
                self._last_net_io_time = time.time()
                return 0.0
                
            time_diff = time.time() - self._last_net_io_time
            if time_diff == 0:
                return 0.0
                
            bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
            bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
            
            bandwidth = (bytes_sent + bytes_recv) / time_diff / 1024 / 1024  # Convert to MB/s
            
            self._last_net_io = net_io
            self._last_net_io_time = time.time()
            
            return bandwidth
            
        except Exception as e:
            self.logger.error_with_context(
                "Bandwidth calculation error",
                error=str(e)
            )
            return 0.0

    async def _check_exchange_connectivity(self):
        """Check connectivity to all exchanges"""
        for exchange_name, exchange_config in self.config.EXCHANGES.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(exchange_config['health_endpoint']) as response:
                        if response.status != 200:
                            await self._create_alert(
                                AlertLevel.CRITICAL,
                                "Network",
                                f"Lost connection to {exchange_name}",
                                {
                                    'exchange': exchange_name,
                                    'status': response.status,
                                    'response': await response.text()
                                },
                                ResourceType.NETWORK
                            )
                        else:
                            # Update connection history
                            self.connection_history[exchange_name].append(datetime.now())
                            # Keep only recent history
                            self.connection_history[exchange_name] = [
                                t for t in self.connection_history[exchange_name]
                                if (datetime.now() - t).total_seconds() < self.config.CONNECTION_HISTORY_SECONDS
                            ]
                            
            except Exception as e:
                self.logger.error_with_context(
                    "Exchange connectivity check error",
                    exchange=exchange_name,
                    error=str(e)
                )
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    "Network",
                    f"Connection error to {exchange_name}",
                    {
                        'exchange': exchange_name,
                        'error': str(e)
                    },
                    ResourceType.NETWORK
                )

    def _get_retry_count(self) -> int:
        """Get total number of connection retries"""
        total_retries = 0
        for exchange_name in self.config.EXCHANGES:
            connection_times = self.connection_history[exchange_name]
            if len(connection_times) >= 2:
                # Count gaps in connection history as retries
                for i in range(1, len(connection_times)):
                    if (connection_times[i] - connection_times[i-1]).total_seconds() > self.config.CONNECTION_RETRY_THRESHOLD:
                        total_retries += 1
        return total_retries
        async def _monitor_tasks(self):
        """Monitor asyncio tasks for deadlocks and performance issues"""
        while True:
            try:
                task_metrics = await self._collect_task_metrics()
                self.task_metrics_history.append(task_metrics)
                
                # Trim history
                if len(self.task_metrics_history) > self.config.METRICS_HISTORY_LENGTH:
                    self.task_metrics_history = self.task_metrics_history[-self.config.METRICS_HISTORY_LENGTH:]
                
                # Check for blocked tasks
                if task_metrics.blocked_tasks > 0:
                    await self._create_alert(
                        AlertLevel.CRITICAL,
                        "Tasks",
                        "Blocked tasks detected",
                        {
                            'blocked_count': task_metrics.blocked_tasks,
                            'blocked_tasks': list(self.blocked_tasks),
                            'task_details': self._get_task_details()
                        },
                        ResourceType.TASK
                    )
                
                # Check for long-running tasks
                if task_metrics.long_running_tasks:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Tasks",
                        "Long-running tasks detected",
                        {
                            'task_count': len(task_metrics.long_running_tasks),
                            'tasks': task_metrics.long_running_tasks,
                            'durations': self._get_task_durations()
                        },
                        ResourceType.TASK
                    )
                
                # Check task queues
                for queue_name, size in task_metrics.queue_sizes.items():
                    if size > self.config.MAX_QUEUE_SIZE:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            "Tasks",
                            f"Queue {queue_name} exceeding size limit",
                            {
                                'queue': queue_name,
                                'size': size,
                                'threshold': self.config.MAX_QUEUE_SIZE
                            },
                            ResourceType.TASK
                        )
                
                await asyncio.sleep(self.config.TASK_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error_with_context(
                    "Task monitoring error",
                    error=str(e),
                    traceback=True
                )
                await asyncio.sleep(5)

    async def _collect_task_metrics(self) -> TaskMetrics:
        """Collect comprehensive task metrics"""
        try:
            # Get all tasks
            all_tasks = asyncio.all_tasks()
            active_tasks = [t for t in all_tasks if not t.done()]
            completed_tasks = [t for t in all_tasks if t.done() and not t.cancelled()]
            failed_tasks = [t for t in completed_tasks if t.exception() is not None]
            
            # Check for blocked tasks
            self.blocked_tasks = set()
            for task in active_tasks:
                if self._is_task_blocked(task):
                    self.blocked_tasks.add(task.get_name())
            
            # Update long-running tasks
            current_time = datetime.now()
            for task in active_tasks:
                task_name = task.get_name()
                if task_name not in self.long_running_tasks:
                    self.long_running_tasks[task_name] = current_time
                elif (current_time - self.long_running_tasks[task_name]).total_seconds() > self.config.TASK_TIMEOUT_SECONDS:
                    if task_name not in self.blocked_tasks:  # Don't double-count blocked tasks
                        self.blocked_tasks.add(task_name)
            
            # Calculate average task duration
            durations = self._get_task_durations().values()
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Get queue sizes
            queue_sizes = await self._get_queue_sizes()
            
            return TaskMetrics(
                active_tasks=len(active_tasks),
                completed_tasks=len(completed_tasks),
                failed_tasks=len(failed_tasks),
                average_duration=avg_duration,
                blocked_tasks=len(self.blocked_tasks),
                queue_sizes=queue_sizes,
                long_running_tasks=list(self.long_running_tasks.keys()),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error_with_context(
                "Error collecting task metrics",
                error=str(e),
                traceback=True
            )
            return None

    def _is_task_blocked(self, task: asyncio.Task) -> bool:
        """Check if a task is blocked"""
        try:
            # Get task stack
            stack = task.get_stack()
            if not stack:
                return False
                
            # Check for common blocking patterns
            blocking_patterns = [
                'asyncio.queues',
                'asyncio.locks',
                'socket.recv',
                'aiohttp.client',
                'asyncpg.connection'
            ]
            
            frame_summary = stack[-1]
            for pattern in blocking_patterns:
                if pattern in frame_summary.filename:
                    return True
                    
            return False
            
        except Exception:
            return False

    async def _get_queue_sizes(self) -> Dict[str, int]:
        """Get sizes of all monitored queues"""
        try:
            queues = {
                'events': self.event_bus._queue.qsize() if hasattr(self.event_bus, '_queue') else 0,
                'orders': self.trading_system.order_queue.qsize() if hasattr(self.trading_system, 'order_queue') else 0,
                'signals': self.trading_system.signal_queue.qsize() if hasattr(self.trading_system, 'signal_queue') else 0
            }
            return queues
        except Exception:
            return {}

    def _get_task_durations(self) -> Dict[str, float]:
        """Get durations of all tasks"""
        durations = {}
        current_time = datetime.now()
        for task_name, start_time in self.long_running_tasks.items():
            duration = (current_time - start_time).total_seconds()
            durations[task_name] = duration
        return durations

    def _get_task_details(self) -> Dict[str, Dict]:
        """Get detailed information about tasks"""
        details = {}
        for task in asyncio.all_tasks():
            task_name = task.get_name()
            details[task_name] = {
                'state': task._state,
                'stack': [str(f) for f in task.get_stack()] if not task.done() else [],
                'duration': (datetime.now() - self.long_running_tasks.get(task_name, datetime.now())).total_seconds(),
                'cancelled': task.cancelled(),
                'exception': str(task.exception()) if task.done() and task.exception() is not None else None
            }
        return details

    async def _monitor_system_health(self):
        """Monitor overall system health metrics"""
        while True:
            try:
                health_metrics = self._collect_health_metrics()
                
                # Check CPU load
                if health_metrics.system_load[0] > self.config.MAX_LOAD_THRESHOLD:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "System",
                        "High system load detected",
                        {
                            'load': health_metrics.system_load[0],
                            'threshold': self.config.MAX_LOAD_THRESHOLD
                        },
                        ResourceType.CPU
                    )
                
                # Check file descriptors
                if health_metrics.open_file_descriptors > self.config.MAX_FILE_DESCRIPTORS:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "System",
                        "High number of open file descriptors",
                        {
                            'count': health_metrics.open_file_descriptors,
                            'threshold': self.config.MAX_FILE_DESCRIPTORS
                        }
                    )
                
                # Check I/O operations
                io_rate = (health_metrics.io_read_bytes + health_metrics.io_write_bytes) / self.config.IO_CHECK_INTERVAL
                if io_rate > self.config.MAX_IO_RATE:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "System",
                        "High I/O rate detected",
                        {
                            'rate': io_rate,
                            'threshold': self.config.MAX_IO_RATE
                        }
                    )
                
                await asyncio.sleep(self.config.HEALTH_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error_with_context(
                    "System health monitoring error",
                    error=str(e),
                    traceback=True
                )
                await asyncio.sleep(5)

    def _collect_health_metrics(self) -> ResourceMetrics:
        """Collect system health metrics"""
        try:
            process = psutil.Process()
            
            return ResourceMetrics(
                cpu_percent=process.cpu_percent(),
                memory_percent=process.memory_percent(),
                disk_percent=psutil.disk_usage('/').percent,
                network_bytes_sent=psutil.net_io_counters().bytes_sent,
                network_bytes_recv=psutil.net_io_counters().bytes_recv,
                io_read_bytes=process.io_counters().read_bytes,
                io_write_bytes=process.io_counters().write_bytes,
                open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                thread_count=process.num_threads(),
                process_time=time.time() - self.start_time,
                system_load=os.getloadavg(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error_with_context(
                "Error collecting health metrics",
                error=str(e),
                traceback=True
            )
            return None
    async def _monitor_resources(self):
        """Enhanced resource monitoring with performance optimization"""
        while True:
            try:
                # Collect comprehensive resource metrics
                resource_metrics = await self._collect_resource_metrics()
                self.resource_metrics_history.append(resource_metrics)
                
                # Perform resource analysis
                await self._analyze_resource_usage(resource_metrics)
                
                # Check performance bottlenecks
                await self._check_performance_bottlenecks(resource_metrics)
                
                # Optimize resource usage
                await self._optimize_resources(resource_metrics)
                
                # Monitor system calls
                await self._monitor_system_calls()
                
                # Check for resource leaks
                await self._check_resource_leaks()
                
                await asyncio.sleep(self.config.RESOURCE_CHECK_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error_with_context(
                    "Resource monitoring error",
                    error=str(e),
                    traceback=True
                )
                await asyncio.sleep(5)

    async def _collect_resource_metrics(self) -> Dict:
        """Collect detailed resource metrics"""
        try:
            process = psutil.Process()
            
            # CPU metrics
            cpu_metrics = {
                'cpu_percent': process.cpu_percent(interval=1),
                'cpu_times': process.cpu_times()._asdict(),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'cpu_stats': psutil.cpu_stats()._asdict(),
                'load_avg': psutil.getloadavg()
            }
            
            # Memory metrics
            memory_metrics = {
                'memory_info': process.memory_info()._asdict(),
                'memory_percent': process.memory_percent(),
                'memory_maps': [m._asdict() for m in process.memory_maps()],
                'swap_memory': psutil.swap_memory()._asdict()
            }
            
            # Disk metrics
            disk_metrics = {
                'disk_usage': psutil.disk_usage('/')._asdict(),
                'disk_io_counters': psutil.disk_io_counters()._asdict(),
                'disk_partitions': [p._asdict() for p in psutil.disk_partitions()]
            }
            
            # Network metrics
            network_metrics = {
                'net_io_counters': psutil.net_io_counters()._asdict(),
                'net_connections': len(process.connections()),
                'net_if_stats': {name: stats._asdict() 
                                for name, stats in psutil.net_if_stats().items()}
            }
            
            # Process metrics
            process_metrics = {
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
                'num_handles': process.num_handles() if hasattr(process, 'num_handles') else None,
                'io_counters': process.io_counters()._asdict(),
                'ctx_switches': process.num_ctx_switches()._asdict()
            }
            
            return {
                'cpu': cpu_metrics,
                'memory': memory_metrics,
                'disk': disk_metrics,
                'network': network_metrics,
                'process': process_metrics,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error_with_context(
                "Error collecting resource metrics",
                error=str(e),
                traceback=True
            )
            return {}

    async def _analyze_resource_usage(self, metrics: Dict):
        """Analyze resource usage patterns and trends"""
        try:
            # Analyze CPU usage patterns
            cpu_pattern = self._analyze_cpu_pattern(metrics['cpu'])
            if cpu_pattern['high_usage_duration'] > self.config.CPU_ALERT_THRESHOLD:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Resources",
                    "Sustained high CPU usage",
                    {
                        'duration': cpu_pattern['high_usage_duration'],
                        'pattern': cpu_pattern['pattern'],
                        'affected_cores': cpu_pattern['affected_cores']
                    },
                    ResourceType.CPU
                )
            
            # Analyze memory usage patterns
            memory_pattern = self._analyze_memory_pattern(metrics['memory'])
            if memory_pattern['growth_rate'] > self.config.MEMORY_GROWTH_THRESHOLD:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Resources",
                    "Abnormal memory growth detected",
                    {
                        'growth_rate': memory_pattern['growth_rate'],
                        'pattern': memory_pattern['pattern'],
                        'largest_allocations': memory_pattern['largest_allocations']
                    },
                    ResourceType.MEMORY
                )
            
            # Analyze disk I/O patterns
            io_pattern = self._analyze_io_pattern(metrics['disk'])
            if io_pattern['bottleneck_severity'] > self.config.IO_ALERT_THRESHOLD:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Resources",
                    "I/O bottleneck detected",
                    {
                        'severity': io_pattern['bottleneck_severity'],
                        'pattern': io_pattern['pattern'],
                        'affected_operations': io_pattern['affected_operations']
                    },
                    ResourceType.DISK
                )
            
        except Exception as e:
            self.logger.error_with_context(
                "Resource analysis error",
                error=str(e),
                traceback=True
            )

    async def _check_performance_bottlenecks(self, metrics: Dict):
        """Check for performance bottlenecks"""
        try:
            bottlenecks = []
            
            # Check CPU bottlenecks
            if metrics['cpu']['cpu_percent'] > 80:
                bottlenecks.append({
                    'type': 'CPU',
                    'severity': 'high',
                    'details': self._analyze_cpu_bottleneck(metrics['cpu'])
                })
            
            # Check memory bottlenecks
            if metrics['memory']['memory_percent'] > 80:
                bottlenecks.append({
                    'type': 'Memory',
                    'severity': 'high',
                    'details': self._analyze_memory_bottleneck(metrics['memory'])
                })
            
            # Check I/O bottlenecks
            io_stats = self._analyze_io_bottleneck(metrics['disk'])
            if io_stats['bottleneck_detected']:
                bottlenecks.append({
                    'type': 'I/O',
                    'severity': io_stats['severity'],
                    'details': io_stats
                })
            
            # Check network bottlenecks
            net_stats = self._analyze_network_bottleneck(metrics['network'])
            if net_stats['bottleneck_detected']:
                bottlenecks.append({
                    'type': 'Network',
                    'severity': net_stats['severity'],
                    'details': net_stats
                })
            
            if bottlenecks:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Performance",
                    "Performance bottlenecks detected",
                    {
                        'bottlenecks': bottlenecks,
                        'recommendations': self._generate_optimization_recommendations(bottlenecks)
                    }
                )
                
                # Attempt automatic optimization if enabled
                if self.config.AUTO_OPTIMIZATION_ENABLED:
                    await self._optimize_resources(metrics, bottlenecks)
                    
        except Exception as e:
            self.logger.error_with_context(
                "Performance bottleneck check error",
                error=str(e),
                traceback=True
            )

    async def _optimize_resources(self, metrics: Dict, bottlenecks: List[Dict] = None):
        """Optimize resource usage based on current metrics"""
        try:
            optimizations_applied = []
            
            # Optimize memory usage
            if await self._optimize_memory_usage(metrics['memory']):
                optimizations_applied.append('memory')
            
            # Optimize CPU usage
            if await self._optimize_cpu_usage(metrics['cpu']):
                optimizations_applied.append('cpu')
            
            # Optimize I/O operations
            if await self._optimize_io_operations(metrics['disk']):
                optimizations_applied.append('io')
            
            # Optimize network usage
            if await self._optimize_network_usage(metrics['network']):
                optimizations_applied.append('network')
            
            if optimizations_applied:
                self.logger.info_with_context(
                    "Resource optimizations applied",
                    optimizations=optimizations_applied,
                    metrics_after=await self._collect_resource_metrics()
                )
                
        except Exception as e:
            self.logger.error_with_context(
                "Resource optimization error",
                error=str(e),
                traceback=True
            )

    async def _monitor_system_calls(self):
        """Monitor and analyze system calls"""
        try:
            if not hasattr(self, '_syscall_tracker'):
                self._syscall_tracker = defaultdict(int)
            
            process = psutil.Process()
            
            # Get system call statistics
            if hasattr(process, 'num_ctx_switches'):
                ctx_switches = process.num_ctx_switches()
                
                # Track context switches
                voluntary_switches = ctx_switches.voluntary
                involuntary_switches = ctx_switches.involuntary
                
                # Analyze for abnormal patterns
                if involuntary_switches > self.config.MAX_INVOLUNTARY_SWITCHES:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Performance",
                        "High number of involuntary context switches",
                        {
                            'involuntary_switches': involuntary_switches,
                            'voluntary_switches': voluntary_switches,
                            'threshold': self.config.MAX_INVOLUNTARY_SWITCHES
                        }
                    )
            
        except Exception as e:
            self.logger.error_with_context(
                "System call monitoring error",
                error=str(e),
                traceback=True
            )

    async def _check_resource_leaks(self):
        """Check for resource leaks"""
        try:
            process = psutil.Process()
            
            # Check file descriptor leaks
            if hasattr(process, 'num_fds'):
                num_fds = process.num_fds()
                if num_fds > self.config.MAX_FILE_DESCRIPTORS:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        "Resources",
                        "Possible file descriptor leak",
                        {
                            'current_fds': num_fds,
                            'threshold': self.config.MAX_FILE_DESCRIPTORS,
                            'open_files': process.open_files()
                        }
                    )
            
            # Check socket leaks
            connections = process.connections()
            if len(connections) > self.config.MAX_CONNECTIONS:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Resources",
                    "Possible socket leak",
                    {
                        'current_connections': len(connections),
                        'threshold': self.config.MAX_CONNECTIONS,
                        'connection_details': [conn._asdict() for conn in connections]
                    }
                )
            
            # Check thread leaks
            if process.num_threads() > self.config.MAX_THREADS:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Resources",
                    "Possible thread leak",
                    {
                        'current_threads': process.num_threads(),
                        'threshold': self.config.MAX_THREADS,
                        'thread_details': self._get_thread_details()
                    }
                )
                
        except Exception as e:
            self.logger.error_with_context(
                "Resource leak check error",
                error=str(e),
                traceback=True
            )
     async def _perform_advanced_recovery(self, alert: Alert) -> bool:
        """Perform advanced system recovery based on alert type"""
        try:
            self.logger.info_with_context(
                "Starting advanced recovery procedure",
                alert_type=alert.level.value,
                alert_source=alert.source,
                resource_type=alert.resource_type.value if alert.resource_type else None
            )
            
            recovery_success = False
            
            if alert.resource_type == ResourceType.CPU:
                recovery_success = await self._recover_cpu_issues(alert)
            elif alert.resource_type == ResourceType.MEMORY:
                recovery_success = await self._recover_memory_issues(alert)
            elif alert.resource_type == ResourceType.DISK:
                recovery_success = await self._recover_disk_issues(alert)
            elif alert.resource_type == ResourceType.NETWORK:
                recovery_success = await self._recover_network_issues(alert)
            elif alert.resource_type == ResourceType.TASK:
                recovery_success = await self._recover_task_issues(alert)
            
            self.logger.info_with_context(
                "Advanced recovery procedure completed",
                success=recovery_success,
                alert_source=alert.source,
                recovery_actions_taken=self._get_recovery_actions()
            )
            
            return recovery_success
            
        except Exception as e:
            self.logger.error_with_context(
                "Advanced recovery error",
                error=str(e),
                alert_details=alert.__dict__,
                traceback=True
            )
            return False

    async def _recover_cpu_issues(self, alert: Alert) -> bool:
        """Recover from CPU-related issues"""
        try:
            actions_taken = []
            
            # Check for CPU-intensive tasks
            cpu_intensive_tasks = self._identify_cpu_intensive_tasks()
            if cpu_intensive_tasks:
                # Throttle or suspend non-critical tasks
                await self._throttle_tasks(cpu_intensive_tasks)
                actions_taken.append('throttled_cpu_intensive_tasks')
            
            # Adjust thread pool size
            if self.trading_system.thread_pool:
                optimal_size = self._calculate_optimal_thread_pool_size()
                await self._adjust_thread_pool(optimal_size)
                actions_taken.append('adjusted_thread_pool')
            
            # Optimize task scheduling
            await self._optimize_task_scheduling()
            actions_taken.append('optimized_task_scheduling')
            
            self.logger.info_with_context(
                "CPU recovery actions completed",
                actions_taken=actions_taken
            )
            
            return True
            
        except Exception as e:
            self.logger.error_with_context(
                "CPU recovery error",
                error=str(e),
                traceback=True
            )
            return False

    async def _recover_memory_issues(self, alert: Alert) -> bool:
        """Recover from memory-related issues"""
        try:
            actions_taken = []
            
            # Force garbage collection
            gc.collect()
            actions_taken.append('garbage_collection')
            
            # Clear caches
            await self._clear_system_caches()
            actions_taken.append('cleared_caches')
            
            # Release unused memory
            if hasattr(self.trading_system, 'release_memory'):
                await self.trading_system.release_memory()
                actions_taken.append('released_trading_system_memory')
            
            # Restart memory-intensive components if needed
            if alert.details.get('memory_percent', 0) > self.config.CRITICAL_MEMORY_THRESHOLD:
                await self._restart_memory_intensive_components()
                actions_taken.append('restarted_components')
            
            self.logger.info_with_context(
                "Memory recovery actions completed",
                actions_taken=actions_taken
            )
            
            return True
            
        except Exception as e:
            self.logger.error_with_context(
                "Memory recovery error",
                error=str(e),
                traceback=True
            )
            return False

    async def _recover_network_issues(self, alert: Alert) -> bool:
        """Recover from network-related issues"""
        try:
            actions_taken = []
            
            # Reset connections
            await self._reset_connections()
            actions_taken.append('reset_connections')
            
            # Switch to backup endpoints if available
            if self.config.BACKUP_ENDPOINTS_ENABLED:
                await self._switch_to_backup_endpoints()
                actions_taken.append('switched_to_backup_endpoints')
            
            # Adjust network timeouts
            self._adjust_network_timeouts()
            actions_taken.append('adjusted_timeouts')
            
            # Implement exponential backoff for retries
            self._configure_exponential_backoff()
            actions_taken.append('configured_backoff')
            
            self.logger.info_with_context(
                "Network recovery actions completed",
                actions_taken=actions_taken
            )
            
            return True
            
        except Exception as e:
            self.logger.error_with_context(
                "Network recovery error",
                error=str(e),
                traceback=True
            )
            return False

    async def _recover_task_issues(self, alert: Alert) -> bool:
        """Recover from task-related issues"""
        try:
            actions_taken = []
            
            # Cancel stuck tasks
            stuck_tasks = self._identify_stuck_tasks()
            if stuck_tasks:
                await self._cancel_stuck_tasks(stuck_tasks)
                actions_taken.append('cancelled_stuck_tasks')
            
            # Reset task queues
            await self._reset_task_queues()
            actions_taken.append('reset_task_queues')
            
            # Restart task scheduler if needed
            if self._should_restart_scheduler():
                await self._restart_task_scheduler()
                actions_taken.append('restarted_scheduler')
            
            self.logger.info_with_context(
                "Task recovery actions completed",
                actions_taken=actions_taken
            )
            
            return True
            
        except Exception as e:
            self.logger.error_with_context(
                "Task recovery error",
                error=str(e),
                traceback=True
            )
            return False

    async def _integrate_external_monitoring(self):
        """Integrate with external monitoring services"""
        try:
            # Configure external monitoring services
            if self.config.DATADOG_ENABLED:
                await self._setup_datadog_monitoring()
            
            if self.config.PROMETHEUS_ENABLED:
                await self._setup_prometheus_monitoring()
            
            if self.config.GRAFANA_ENABLED:
                await self._setup_grafana_monitoring()
            
            if self.config.CLOUDWATCH_ENABLED:
                await self._setup_cloudwatch_monitoring()
                
            self.logger.info_with_context(
                "External monitoring integration completed",
                enabled_services=self._get_enabled_monitoring_services()
            )
            
        except Exception as e:
            self.logger.error_with_context(
                "External monitoring integration error",
                error=str(e),
                traceback=True
            )

    async def _setup_datadog_monitoring(self):
        """Setup Datadog monitoring"""
        try:
            from datadog import initialize, statsd
            
            initialize(
                api_key=self.config.DATADOG_API_KEY,
                app_key=self.config.DATADOG_APP_KEY
            )
            
            # Setup metrics
            self._setup_datadog_metrics(statsd)
            
            # Setup events
            self._setup_datadog_events()
            
            # Setup alerts
            self._setup_datadog_alerts()
            
        except Exception as e:
            self.logger.error_with_context(
                "Datadog setup error",
                error=str(e),
                traceback=True
            )

    async def _setup_prometheus_monitoring(self):
        """Setup Prometheus monitoring"""
        try:
            from prometheus_client import start_http_server, Counter, Gauge, Histogram
            
            # Start Prometheus HTTP server
            start_http_server(self.config.PROMETHEUS_PORT)
            
            # Setup metrics
            self.prometheus_metrics = {
                'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage'),
                'memory_usage': Gauge('memory_usage_percent', 'Memory usage percentage'),
                'active_tasks': Gauge('active_tasks', 'Number of active tasks'),
                'alert_count': Counter('alert_count', 'Number of alerts generated'),
                'request_latency': Histogram('request_latency_seconds', 'Request latency')
            }
            
        except Exception as e:
            self.logger.error_with_context(
                "Prometheus setup error",
                error=str(e),
                traceback=True
            )

    async def _setup_grafana_monitoring(self):
        """Setup Grafana monitoring"""
        try:
            # Setup Grafana datasources
            await self._setup_grafana_datasources()
            
            # Setup dashboards
            await self._setup_grafana_dashboards()
            
            # Setup alerts
            await self._setup_grafana_alerts()
            
        except Exception as e:
            self.logger.error_with_context(
                "Grafana setup error",
                error=str(e),
                traceback=True
            )

    async def _setup_cloudwatch_monitoring(self):
        """Setup AWS CloudWatch monitoring"""
        try:
            import boto3
            
            # Initialize CloudWatch client
            self.cloudwatch = boto3.client('cloudwatch')
            
            # Setup metrics
            self._setup_cloudwatch_metrics()
            
            # Setup alarms
            self._setup_cloudwatch_alarms()
            
            # Setup dashboards
            self._setup_cloudwatch_dashboards()
            
        except Exception as e:
            self.logger.error_with_context(
                "CloudWatch setup error",
                error=str(e),
                traceback=True
            )

    def _update_external_metrics(self, metrics: Dict):
        """Update metrics in external monitoring services"""
        try:
            # Update Datadog metrics
            if self.config.DATADOG_ENABLED:
                self._update_datadog_metrics(metrics)
            
            # Update Prometheus metrics
            if self.config.PROMETHEUS_ENABLED:
                self._update_prometheus_metrics(metrics)
            
            # Update CloudWatch metrics
            if self.config.CLOUDWATCH_ENABLED:
                self._update_cloudwatch_metrics(metrics)
                
        except Exception as e:
            self.logger.error_with_context(
                "External metrics update error",
                error=str(e),
                traceback=True
            ) 

    async def generate_system_report(self) -> Dict:
        """Generate comprehensive system health and performance report"""
        try:
            # Collect all metrics
            metrics = await self._collect_all_metrics()
            
            # Generate health score
            health_score = self._calculate_system_health_score(metrics)
            
            # Analyze trends
            trend_analysis = self._analyze_system_trends()
            
            # Generate performance profile
            performance_profile = await self._generate_performance_profile()
            
            # Compile recommendations
            recommendations = self._generate_system_recommendations(
                metrics,
                health_score,
                trend_analysis
            )
            
            report = {
                'timestamp': datetime.now(),
                'overall_health': {
                    'score': health_score,
                    'status': self._get_health_status(health_score),
                    'critical_issues': self._get_critical_issues()
                },
                'performance_metrics': {
                    'current': metrics,
                    'trends': trend_analysis,
                    'profile': performance_profile
                },
                'resource_utilization': {
                    'current': self._get_resource_utilization(),
                    'bottlenecks': self._identify_bottlenecks(),
                    'optimization_opportunities': self._find_optimization_opportunities()
                },
                'system_stability': {
                    'uptime': self._get_system_uptime(),
                    'error_rate': self._calculate_error_rate(),
                    'recovery_rate': self._calculate_recovery_rate()
                },
                'recommendations': recommendations,
                'alerts': {
                    'active': len(self.active_alerts),
                    'history': self._get_alert_statistics(),
                    'most_frequent': self._get_most_frequent_alerts()
                }
            }
            
            # Store report
            await self._store_system_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error_with_context(
                "Report generation error",
                error=str(e),
                traceback=True
            )
            return {}

    def _calculate_system_health_score(self, metrics: Dict) -> float:
        """Calculate overall system health score"""
        try:
            scores = []
            
            # CPU health (0-1)
            cpu_score = 1.0 - (metrics['cpu']['cpu_percent'] / 100.0)
            scores.append(cpu_score * self.config.CPU_HEALTH_WEIGHT)
            
            # Memory health (0-1)
            memory_score = 1.0 - (metrics['memory']['memory_percent'] / 100.0)
            scores.append(memory_score * self.config.MEMORY_HEALTH_WEIGHT)
            
            # Disk health (0-1)
            disk_score = 1.0 - (metrics['disk']['disk_usage']['percent'] / 100.0)
            scores.append(disk_score * self.config.DISK_HEALTH_WEIGHT)
            
            # Network health (0-1)
            network_health = self._calculate_network_health(metrics['network'])
            scores.append(network_health * self.config.NETWORK_HEALTH_WEIGHT)
            
            # Task health (0-1)
            task_health = self._calculate_task_health()
            scores.append(task_health * self.config.TASK_HEALTH_WEIGHT)
            
            # Error rate impact
            error_rate = self._calculate_error_rate()
            error_impact = max(0, 1 - (error_rate * self.config.ERROR_RATE_IMPACT))
            scores.append(error_impact * self.config.ERROR_HEALTH_WEIGHT)
            
            return sum(scores) * 100  # Return as percentage
            
        except Exception as e:
            self.logger.error_with_context(
                "Health score calculation error",
                error=str(e),
                traceback=True
            )
            return 0.0

    async def _generate_performance_profile(self) -> Dict:
        """Generate detailed performance profile"""
        try:
            # Start profiling
            import cProfile
            import pstats
            import io
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Run key operations
            await self._profile_key_operations()
            
            # Stop profiling
            profiler.disable()
            
            # Analyze results
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats()
            
            # Parse profiling data
            profile_data = self._parse_profile_data(s.getvalue())
            
            # Analyze bottlenecks
            bottlenecks = self._analyze_performance_bottlenecks(profile_data)
            
            return {
                'execution_times': profile_data,
                'bottlenecks': bottlenecks,
                'recommendations': self._generate_performance_recommendations(bottlenecks)
            }
            
        except Exception as e:
            self.logger.error_with_context(
                "Performance profile generation error",
                error=str(e),
                traceback=True
            )
            return {}

    def _analyze_system_trends(self) -> Dict:
        """Analyze system performance trends"""
        try:
            trends = {}
            
            # Analyze resource usage trends
            for metric_type in ['cpu', 'memory', 'disk', 'network']:
                trends[metric_type] = self._calculate_metric_trends(metric_type)
            
            # Analyze performance trends
            trends['performance'] = {
                'latency': self._analyze_latency_trend(),
                'throughput': self._analyze_throughput_trend(),
                'error_rate': self._analyze_error_rate_trend()
            }
            
            # Analyze task trends
            trends['tasks'] = {
                'completion_rate': self._analyze_task_completion_trend(),
                'queue_size': self._analyze_queue_size_trend()
            }
            
            # Calculate trend indicators
            trends['indicators'] = {
                'stability': self._calculate_stability_indicator(trends),
                'degradation': self._calculate_degradation_indicator(trends),
                'improvement': self._calculate_improvement_indicator(trends)
            }
            
            return trends
            
        except Exception as e:
            self.logger.error_with_context(
                "Trend analysis error",
                error=str(e),
                traceback=True
            )
            return {}

    def _calculate_metric_trends(self, metric_type: str) -> Dict:
        """Calculate trends for specific metric type"""
        try:
            history = self._get_metric_history(metric_type)
            if not history:
                return {}
                
            # Calculate basic statistics
            mean = np.mean(history)
            std = np.std(history)
            
            # Calculate trend line
            x = np.arange(len(history))
            z = np.polyfit(x, history, 1)
            slope = z[0]
            
            # Determine trend direction and strength
            trend_direction = 'increasing' if slope > 0 else 'decreasing'
            trend_strength = abs(slope) / mean if mean != 0 else 0
            
            # Calculate volatility
            volatility = std / mean if mean != 0 else 0
            
            # Detect anomalies
            anomalies = self._detect_anomalies(history, mean, std)
            
            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'volatility': volatility,
                'anomalies': anomalies,
                'forecast': self._forecast_metric(history)
            }
            
        except Exception as e:
            self.logger.error_with_context(
                "Metric trend calculation error",
                error=str(e),
                traceback=True
            )
            return {}

    def _detect_anomalies(self, data: List[float], mean: float, std: float) -> List[Dict]:
        """Detect anomalies in metric data"""
        try:
            anomalies = []
            z_threshold = self.config.ANOMALY_THRESHOLD
            
            for i, value in enumerate(data):
                z_score = (value - mean) / std if std != 0 else 0
                
                if abs(z_score) > z_threshold:
                    anomalies.append({
                        'index': i,
                        'value': value,
                        'z_score': z_score,
                        'deviation': abs(value - mean)
                    })
                    
            return anomalies
            
        except Exception as e:
            self.logger.error_with_context(
                "Anomaly detection error",
                error=str(e),
                traceback=True
            )
            return []

    def _forecast_metric(self, history: List[float]) -> Dict:
        """Generate simple forecast for metric"""
        try:
            if len(history) < self.config.MIN_FORECAST_POINTS:
                return {}
                
            # Simple linear regression for forecast
            x = np.arange(len(history))
            y = np.array(history)
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Forecast next n points
            future_points = self.config.FORECAST_POINTS
            forecast_x = np.arange(len(history), len(history) + future_points)
            forecast_y = p(forecast_x)
            
            # Calculate confidence intervals
            residuals = y - p(x)
            std_residuals = np.std(residuals)
            
            confidence_interval = std_residuals * 1.96  # 95% confidence interval
            
            return {
                'points': forecast_y.tolist(),
                'confidence_interval': confidence_interval,
                'trend_coefficient': z[0],
                'r_squared': self._calculate_r_squared(y, p(x))
            }
            
        except Exception as e:
            self.logger.error_with_context(
                "Forecast generation error",
                error=str(e),
                traceback=True
            )
            return {}

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value for forecast accuracy"""
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
        except Exception:
            return 0.0

    def _generate_system_recommendations(self, 
                                      metrics: Dict, 
                                      health_score: float,
                                      trend_analysis: Dict) -> List[Dict]:
        """Generate system optimization recommendations"""
        try:
            recommendations = []
            
            # Resource optimization recommendations
            if health_score < self.config.HEALTH_SCORE_THRESHOLD:
                recommendations.extend(self._get_resource_recommendations(metrics))
            
            # Performance optimization recommendations
            if trend_analysis.get('indicators', {}).get('degradation', 0) > self.config.DEGRADATION_THRESHOLD:
                recommendations.extend(self._get_performance_recommendations(trend_analysis))
            
            # Stability recommendations
            if trend_analysis.get('indicators', {}).get('stability', 0) < self.config.STABILITY_THRESHOLD:
                recommendations.extend(self._get_stability_recommendations(metrics))
            
            return recommendations
            
        except Exception as e:
            self.logger.error_with_context(
                "Recommendation generation error",
                error=str(e),
                traceback=True
            )
            return []

    async def _perform_system_maintenance(self):
        """Perform regular system maintenance tasks"""
        try:
            maintenance_results = {
                'database_cleanup': await self._cleanup_database(),
                'log_rotation': await self._rotate_logs(),
                'cache_cleanup': await self._cleanup_caches(),
                'temp_file_cleanup': await self._cleanup_temp_files(),
                'metrics_archival': await self._archive_old_metrics(),
                'connection_cleanup': await self._cleanup_stale_connections(),
                'memory_defrag': await self._defragment_memory()
            }
            
            self.logger.info_with_context(
                "System maintenance completed",
                maintenance_results=maintenance_results
            )
            
            # Report maintenance metrics
            await self._report_maintenance_metrics(maintenance_results)
            
            return maintenance_results
            
        except Exception as e:
            self.logger.error_with_context(
                "System maintenance error",
                error=str(e),
                traceback=True
            )
            return {}

    async def validate_configuration(self) -> bool:
        """Validate system configuration"""
        try:
            validation_results = {
                'thresholds': self._validate_thresholds(),
                'intervals': self._validate_intervals(),
                'connections': await self._validate_connections(),
                'permissions': self._validate_permissions(),
                'dependencies': self._validate_dependencies(),
                'paths': self._validate_paths()
            }
            
            is_valid = all(validation_results.values())
            
            if not is_valid:
                self.logger.error_with_context(
                    "Configuration validation failed",
                    validation_results=validation_results
                )
            
            return is_valid
            
        except Exception as e:
            self.logger.error_with_context(
                "Configuration validation error",
                error=str(e),
                traceback=True
            )
            return False

    def _validate_thresholds(self) -> bool:
        """Validate monitoring thresholds"""
        try:
            required_thresholds = {
                'CPU_THRESHOLD',
                'MEMORY_THRESHOLD',
                'DISK_USAGE_THRESHOLD',
                'LATENCY_THRESHOLD',
                'MAX_QUEUE_SIZE'
            }
            
            # Check for missing thresholds
            missing = required_thresholds - set(dir(self.config))
            if missing:
                self.logger.error_with_context(
                    "Missing required thresholds",
                    missing_thresholds=list(missing)
                )
                return False
            
            # Validate threshold values
            if not (0 <= self.config.CPU_THRESHOLD <= 100):
                return False
            if not (0 <= self.config.MEMORY_THRESHOLD <= 100):
                return False
            if not (0 <= self.config.DISK_USAGE_THRESHOLD <= 100):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error_with_context(
                "Threshold validation error",
                error=str(e)
            )
            return False

    async def _cleanup_database(self) -> Dict:
        """Clean up old database records"""
        try:
            results = {
                'metrics_cleaned': 0,
                'alerts_cleaned': 0,
                'logs_cleaned': 0
            }
            
            # Clean metrics
            cutoff_date = datetime.now() - timedelta(days=self.config.METRICS_RETENTION_DAYS)
            results['metrics_cleaned'] = await self._delete_old_metrics(cutoff_date)
            
            # Clean alerts
            alerts_cutoff = datetime.now() - timedelta(days=self.config.ALERTS_RETENTION_DAYS)
            results['alerts_cleaned'] = await self._delete_old_alerts(alerts_cutoff)
            
            # Clean logs
            logs_cutoff = datetime.now() - timedelta(days=self.config.LOGS_RETENTION_DAYS)
            results['logs_cleaned'] = await self._delete_old_logs(logs_cutoff)
            
            return results
            
        except Exception as e:
            self.logger.error_with_context(
                "Database cleanup error",
                error=str(e),
                traceback=True
            )
            return {'error': str(e)}

    async def _rotate_logs(self) -> Dict:
        """Rotate and archive log files"""
        try:
            import glob
            import shutil
            
            results = {
                'rotated_files': 0,
                'archived_files': 0,
                'space_freed': 0
            }
            
            log_dir = self.config.LOG_DIRECTORY
            archive_dir = self.config.LOG_ARCHIVE_DIRECTORY
            
            # Create archive directory if it doesn't exist
            os.makedirs(archive_dir, exist_ok=True)
            
            # Rotate logs
            log_files = glob.glob(f"{log_dir}/*.log")
            for log_file in log_files:
                if os.path.getsize(log_file) > self.config.MAX_LOG_SIZE:
                    archived_name = f"{os.path.basename(log_file)}.{datetime.now():%Y%m%d}"
                    shutil.move(log_file, os.path.join(archive_dir, archived_name))
                    results['rotated_files'] += 1
            
            # Clean old archives
            cutoff_date = datetime.now() - timedelta(days=self.config.LOG_ARCHIVE_DAYS)
            for archived_file in glob.glob(f"{archive_dir}/*.log.*"):
                file_date = datetime.strptime(archived_file[-8:], "%Y%m%d")
                if file_date < cutoff_date:
                    file_size = os.path.getsize(archived_file)
                    os.remove(archived_file)
                    results['archived_files'] += 1
                    results['space_freed'] += file_size
            
            return results
            
        except Exception as e:
            self.logger.error_with_context(
                "Log rotation error",
                error=str(e),
                traceback=True
            )
            return {'error': str(e)}

    async def _defragment_memory(self) -> Dict:
        """Defragment system memory"""
        try:
            initial_memory = self._get_process_memory()
            
            # Force garbage collection
            gc.collect()
            
            # Compact memory pools
            if hasattr(gc, 'collect_with_compact'):  # Available in some Python implementations
                gc.collect_with_compact()
            
            # Release unused memory back to the system
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
            
            final_memory = self._get_process_memory()
            
            return {
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'memory_freed': initial_memory - final_memory
            }
            
        except Exception as e:
            self.logger.error_with_context(
                "Memory defragmentation error",
                error=str(e),
                traceback=True
            )
            return {'error': str(e)}

    async def _archive_old_metrics(self) -> Dict:
        """Archive old metrics data"""
        try:
            results = {
                'metrics_archived': 0,
                'file_size': 0
            }
            
            cutoff_date = datetime.now() - timedelta(days=self.config.METRICS_ARCHIVE_DAYS)
            
            # Prepare archive data
            archive_data = {
                'system_metrics': [m for m in self.resource_metrics_history if m.timestamp < cutoff_date],
                'network_metrics': [m for m in self.network_metrics_history if m.timestamp < cutoff_date],
                'memory_metrics': [m for m in self.memory_metrics_history if m.timestamp < cutoff_date],
                'task_metrics': [m for m in self.task_metrics_history if m.timestamp < cutoff_date]
            }
            
            # Create archive file
            archive_path = os.path.join(
                self.config.METRICS_ARCHIVE_DIRECTORY,
                f"metrics_{datetime.now():%Y%m%d}.json.gz"
            )
            
            # Archive data
            import gzip
            import json
            with gzip.open(archive_path, 'wt') as f:
                json.dump(archive_data, f, default=str)
            
            # Update results
            results['metrics_archived'] = sum(len(m) for m in archive_data.values())
            results['file_size'] = os.path.getsize(archive_path)
            
            # Clean up original data
            self._cleanup_archived_metrics(cutoff_date)
            
            return results
            
        except Exception as e:
            self.logger.error_with_context(
                "Metrics archival error",
                error=str(e),
                traceback=True
            )
            return {'error': str(e)}

    def _cleanup_archived_metrics(self, cutoff_date: datetime):
        """Remove archived metrics from memory"""
        try:
            self.resource_metrics_history = [m for m in self.resource_metrics_history if m.timestamp >= cutoff_date]
            self.network_metrics_history = [m for m in self.network_metrics_history if m.timestamp >= cutoff_date]
            self.memory_metrics_history = [m for m in self.memory_metrics_history if m.timestamp >= cutoff_date]
            self.task_metrics_history = [m for m in self.task_metrics_history if m.timestamp >= cutoff_date]
            
        except Exception as e:
            self.logger.error_with_context(
                "Metrics cleanup error",
                error=str(e),
                traceback=True
            )

    def __str__(self) -> str:
        """String representation of monitoring system"""
        return f"MonitoringSystem(active_alerts={len(self.active_alerts)}, " \
               f"tasks={len(self.monitoring_tasks)}, " \
               f"status={self._determine_system_status(None)})"

    def __repr__(self) -> str:
        """Detailed representation of monitoring system"""
        return f"MonitoringSystem(" \
               f"active_alerts={len(self.active_alerts)}, " \
               f"alert_history={len(self.alert_history)}, " \
               f"tasks={len(self.monitoring_tasks)}, " \
               f"recovering={self.recovering}, " \
               f"status={self._determine_system_status(None)})"            