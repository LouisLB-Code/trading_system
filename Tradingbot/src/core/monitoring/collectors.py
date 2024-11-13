
import psutil
import gc
import os
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from ..logging_config import get_logger

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_latency: float
    event_queue_size: int
    thread_count: int
    process_time: float
    io_counters: Dict
    file_descriptors: int
    system_load: List[float]
    gc_stats: Dict[str, int]
    timestamp: datetime

@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    latency: float
    packet_loss: float
    bandwidth_usage: float
    connection_count: int
    error_count: int
    retry_count: int
    bytes_sent: int
    bytes_received: int
    connection_fails: int
    timestamp: datetime

class MetricsCollector:
    """Enhanced metrics collection system"""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_logger('metrics_collector')
        self.start_time = time.time()
        self.last_io_counters = None
        self.last_network_counters = None
        self.connection_history: Dict[str, List[datetime]] = {}
        self.error_counts: Dict[str, int] = {}
        
    async def collect_all_metrics(self) -> Dict:
        """Collect all system metrics"""
        try:
            return {
                'system': await self.collect_system_metrics(),
                'network': await self.collect_network_metrics(),
                'memory': await self.collect_memory_metrics(),
                'tasks': await self.collect_task_metrics()
            }
        except Exception as e:
            self.logger.error_with_context(
                "Metrics collection error",
                error=str(e),
                traceback=True
            )
            return {}

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        try:
            process = psutil.Process()
            
            # Get IO counters
            io_counters = process.io_counters()
            
            # Get GC statistics
            gc_stats = {
                'collections': gc.get_count(),
                'threshold': gc.get_threshold(),
                'objects': len(gc.get_objects()),
                'garbage': len(gc.garbage)
            }
            
            metrics = SystemMetrics(
                cpu_percent=process.cpu_percent(),
                memory_percent=process.memory_percent(),
                disk_usage=psutil.disk_usage('/').percent,
                network_latency=await self._measure_network_latency(),
                event_queue_size=self._get_queue_size(),
                thread_count=process.num_threads(),
                process_time=time.time() - self.start_time,
                io_counters={
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count
                },
                file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                system_load=os.getloadavg(),
                gc_stats=gc_stats,
                timestamp=datetime.now()
            )
            
            # Log warnings if thresholds exceeded
            self._check_system_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error_with_context(
                "System metrics collection error",
                error=str(e),
                traceback=True
            )
            return None

    async def collect_network_metrics(self) -> NetworkMetrics:
        """Collect network metrics"""
        try:
            net_io = psutil.net_io_counters()
            
            # Calculate bandwidth usage
            bytes_sent = 0
            bytes_received = 0
            if self.last_network_counters:
                time_diff = time.time() - self.last_network_time
                if time_diff > 0:
                    bytes_sent = (net_io.bytes_sent - self.last_network_counters.bytes_sent) / time_diff
                    bytes_received = (net_io.bytes_recv - self.last_network_counters.bytes_recv) / time_diff
            
            self.last_network_counters = net_io
            self.last_network_time = time.time()
            
            # Count active connections
            connections = len(psutil.net_connections())
            
            metrics = NetworkMetrics(
                latency=await self._measure_network_latency(),
                packet_loss=await self._measure_packet_loss(),
                bandwidth_usage=(bytes_sent + bytes_received) / (1024 * 1024),  # Convert to MB/s
                connection_count=connections,
                error_count=sum(self.error_counts.values()),
                retry_count=self._get_retry_count(),
                bytes_sent=bytes_sent,
                bytes_received=bytes_received,
                connection_fails=self._count_connection_failures(),
                timestamp=datetime.now()
            )
            
            # Log warnings if thresholds exceeded
            self._check_network_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error_with_context(
                "Network metrics collection error",
                error=str(e),
                traceback=True
            )
            return None

    def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        if metrics.cpu_percent > self.config.CPU_THRESHOLD:
            self.logger.warning_with_context(
                "CPU usage exceeds threshold",
                current=metrics.cpu_percent,
                threshold=self.config.CPU_THRESHOLD
            )
            
        if metrics.memory_percent > self.config.MEMORY_THRESHOLD:
            self.logger.warning_with_context(
                "Memory usage exceeds threshold",
                current=metrics.memory_percent,
                threshold=self.config.MEMORY_THRESHOLD
            )
            
        if metrics.disk_usage > self.config.DISK_USAGE_THRESHOLD:
            self.logger.warning_with_context(
                "Disk usage exceeds threshold",
                current=metrics.disk_usage,
                threshold=self.config.DISK_USAGE_THRESHOLD
            )

    def _check_network_thresholds(self, metrics: NetworkMetrics):
        """Check network metrics against thresholds"""
        if metrics.latency > self.config.MAX_NETWORK_LATENCY:
            self.logger.warning_with_context(
                "Network latency exceeds threshold",
                current=metrics.latency,
                threshold=self.config.MAX_NETWORK_LATENCY
            )
            
        if metrics.packet_loss > self.config.MAX_PACKET_LOSS:
            self.logger.warning_with_context(
                "Packet loss exceeds threshold",
                current=metrics.packet_loss,
                threshold=self.config.MAX_PACKET_LOSS
            )
            
        if metrics.connection_count > self.config.MAX_CONNECTIONS:
            self.logger.warning_with_context(
                "Connection count exceeds threshold",
                current=metrics.connection_count,
                threshold=self.config.MAX_CONNECTIONS
            )

    async def _measure_network_latency(self) -> float:
        """Measure network latency"""
        try:
            import socket
            start_time = time.time()
            socket.create_connection(("8.8.8.8", 53), timeout=self.config.CONNECTION_TIMEOUT)
            return (time.time() - start_time) * 1000  # Convert to milliseconds
        except:
            return float('inf')

    async def _measure_packet_loss(self) -> float:
        """Measure packet loss rate"""
        try:
            import subprocess
            result = subprocess.run(['ping', '-c', '10', '8.8.8.8'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            output = result.stdout.decode()
            if 'packet loss' in output:
                loss = float(output.split('%')[0].split(' ')[-1])
                return loss / 100
            return 0.0
        except:
            return 0.0

    def _get_retry_count(self) -> int:
        """Get total number of connection retries"""
        count = 0
        for timestamps in self.connection_history.values():
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    if (timestamps[i] - timestamps[i-1]).total_seconds() < self.config.CONNECTION_TIMEOUT:
                        count += 1
        return count

    def _count_connection_failures(self) -> int:
        """Count recent connection failures"""
        cutoff = datetime.now() - timedelta(minutes=5)
        return sum(1 for timestamps in self.connection_history.values()
                  for t in timestamps
                  if t > cutoff)

    def _get_queue_size(self) -> int:
        """Get current event queue size"""
        try:
            import asyncio
            return len(asyncio.all_tasks())
        except:
            return 0