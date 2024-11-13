import logging
import asyncio
import gc
import os
import psutil
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .monitor import Alert, AlertLevel, ResourceType

class RecoveryAction(Enum):
    """Recovery actions available to the system"""
    RESTART_STRATEGY = "restart_strategy"
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"
    REDUCE_POSITION_SIZES = "reduce_position_sizes"
    CLOSE_POSITIONS = "close_positions"
    CLEAR_QUEUE = "clear_queue"
    RESET_CONNECTION = "reset_connection"
    FREE_MEMORY = "free_memory"
    DISK_CLEANUP = "disk_cleanup"
    OPTIMIZE_RESOURCES = "optimize_resources"
    RESTART_COMPONENT = "restart_component"
    THROTTLE_TASKS = "throttle_tasks"
    CLEAR_CACHE = "clear_cache"
    REBALANCE_LOAD = "rebalance_load"

@dataclass
class RecoveryResult:
    """Result of a recovery action"""
    success: bool
    action: RecoveryAction
    details: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

class RecoverySystem:
    """Enhanced system for handling automated recoveries"""
    
    def __init__(self, config, trading_system, event_bus):
        self.config = config
        self.trading_system = trading_system
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.recovery_history: List[RecoveryResult] = []
        self.active_recoveries: Set[str] = set()
        self.last_recovery_time: Dict[RecoveryAction, datetime] = {}
        self.recovery_metrics: Dict[str, int] = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }
        
    async def handle_alert(self, alert: Alert) -> Optional[RecoveryResult]:
        """Handle alert and execute recovery if needed"""
        try:
            if not self._should_attempt_recovery(alert):
                return None

            recovery_id = f"{alert.source}_{alert.level.value}_{datetime.now().timestamp()}"
            
            if recovery_id in self.active_recoveries:
                self.logger.warning(f"Recovery already in progress for alert: {recovery_id}")
                return None
                
            self.active_recoveries.add(recovery_id)
            self.recovery_metrics['total_attempts'] += 1
            
            try:
                result = await self._execute_recovery(alert)
                if result and result.success:
                    self.recovery_metrics['successful_recoveries'] += 1
                else:
                    self.recovery_metrics['failed_recoveries'] += 1
                return result
            finally:
                self.active_recoveries.remove(recovery_id)
                
        except Exception as e:
            self.logger.error(f"Recovery handling error: {str(e)}")
            self.recovery_metrics['failed_recoveries'] += 1
            return None

    def _should_attempt_recovery(self, alert: Alert) -> bool:
        """Check if recovery should be attempted"""
        if not self.config.AUTO_RECOVERY_ENABLED:
            return False
            
        # Skip recovery for INFO level alerts
        if alert.level == AlertLevel.INFO:
            return False
            
        # Check cooldown period
        action = self._get_primary_recovery_action(alert)
        if action in self.last_recovery_time:
            time_since_last = (datetime.now() - self.last_recovery_time[action]).total_seconds()
            if time_since_last < self.config.RECOVERY_COOLDOWN:
                return False
                
        # Check attempt limits
        recent_attempts = [
            r for r in self.recovery_history[-self.config.MAX_RECOVERY_ATTEMPTS:]
            if r.action == action
        ]
        if len(recent_attempts) >= self.config.MAX_RECOVERY_ATTEMPTS:
            return False
            
        return True

    def _get_primary_recovery_action(self, alert: Alert) -> RecoveryAction:
        """Get primary recovery action for alert type"""
        resource_type_actions = {
            ResourceType.CPU: RecoveryAction.OPTIMIZE_RESOURCES,
            ResourceType.MEMORY: RecoveryAction.FREE_MEMORY,
            ResourceType.DISK: RecoveryAction.DISK_CLEANUP,
            ResourceType.NETWORK: RecoveryAction.RESET_CONNECTION,
            ResourceType.TASK: RecoveryAction.RESTART_COMPONENT,
            ResourceType.DATABASE: RecoveryAction.CLEAR_CACHE
        }
        
        if alert.resource_type:
            return resource_type_actions.get(alert.resource_type, RecoveryAction.RESTART_COMPONENT)
        
        # Default actions based on alert level
        level_actions = {
            AlertLevel.WARNING: RecoveryAction.OPTIMIZE_RESOURCES,
            AlertLevel.CRITICAL: RecoveryAction.RESTART_COMPONENT,
            AlertLevel.EMERGENCY: RecoveryAction.PAUSE_TRADING
        }
        
        return level_actions.get(alert.level, RecoveryAction.OPTIMIZE_RESOURCES)

    async def _execute_recovery(self, alert: Alert) -> RecoveryResult:
        """Execute recovery based on alert"""
        try:
            # Determine recovery actions
            actions = self._determine_recovery_actions(alert)
            if not actions:
                return None
                
            self.logger.info(f"Starting recovery sequence for alert: {alert.source}")
            results = []
            
            for action in actions:
                self.logger.info(f"Executing recovery action: {action.value}")
                
                # Execute recovery action
                result = await self._execute_action(action, alert)
                results.append(result)
                
                # Update last recovery time
                self.last_recovery_time[action] = datetime.now()
                
                if result.success:
                    self.logger.info(f"Recovery action {action.value} successful")
                    break
                else:
                    self.logger.warning(f"Recovery action {action.value} failed: {result.error}")
            
            # Get final result
            final_result = self._get_final_result(results)
            
            # Store in history
            self.recovery_history.append(final_result)
            
            # Trim history if needed
            if len(self.recovery_history) > self.config.MAX_METRICS_HISTORY:
                self.recovery_history = self.recovery_history[-self.config.MAX_METRICS_HISTORY:]
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Recovery execution error: {str(e)}")
            return RecoveryResult(
                success=False,
                action=actions[0] if actions else None,
                details={'error': str(e)},
                error=str(e)
            )

    def _determine_recovery_actions(self, alert: Alert) -> List[RecoveryAction]:
        """Determine appropriate recovery actions based on alert"""
        if alert.resource_type == ResourceType.CPU:
            return [
                RecoveryAction.THROTTLE_TASKS,
                RecoveryAction.OPTIMIZE_RESOURCES,
                RecoveryAction.REBALANCE_LOAD,
                RecoveryAction.RESTART_COMPONENT
            ]
            
        elif alert.resource_type == ResourceType.MEMORY:
            return [
                RecoveryAction.FREE_MEMORY,
                RecoveryAction.CLEAR_CACHE,
                RecoveryAction.OPTIMIZE_RESOURCES,
                RecoveryAction.RESTART_COMPONENT
            ]
            
        elif alert.resource_type == ResourceType.DISK:
            return [
                RecoveryAction.DISK_CLEANUP,
                RecoveryAction.CLEAR_CACHE,
                RecoveryAction.OPTIMIZE_RESOURCES
            ]
            
        elif alert.resource_type == ResourceType.NETWORK:
            return [
                RecoveryAction.RESET_CONNECTION,
                RecoveryAction.REBALANCE_LOAD,
                RecoveryAction.RESTART_COMPONENT
            ]
            
        elif alert.resource_type == ResourceType.TASK:
            return [
                RecoveryAction.CLEAR_QUEUE,
                RecoveryAction.THROTTLE_TASKS,
                RecoveryAction.RESTART_COMPONENT
            ]
            
        # Default actions based on alert level
        return self._get_default_recovery_actions(alert)

    def _get_default_recovery_actions(self, alert: Alert) -> List[RecoveryAction]:
        """Get default recovery actions based on alert level"""
        if alert.level == AlertLevel.WARNING:
            return [
                RecoveryAction.OPTIMIZE_RESOURCES,
                RecoveryAction.CLEAR_CACHE
            ]
        elif alert.level == AlertLevel.CRITICAL:
            return [
                RecoveryAction.PAUSE_TRADING,
                RecoveryAction.RESTART_COMPONENT,
                RecoveryAction.RESUME_TRADING
            ]
        elif alert.level == AlertLevel.EMERGENCY:
            return [
                RecoveryAction.PAUSE_TRADING,
                RecoveryAction.CLOSE_POSITIONS,
                RecoveryAction.RESTART_COMPONENT
            ]
        return [RecoveryAction.OPTIMIZE_RESOURCES]

    async def _execute_action(self, action: RecoveryAction, alert: Alert) -> RecoveryResult:
        """Execute specific recovery action"""
        try:
            action_methods = {
                RecoveryAction.RESTART_STRATEGY: self._restart_strategy,
                RecoveryAction.PAUSE_TRADING: self._pause_trading,
                RecoveryAction.RESUME_TRADING: self._resume_trading,
                RecoveryAction.REDUCE_POSITION_SIZES: self._reduce_position_sizes,
                RecoveryAction.CLOSE_POSITIONS: self._close_positions,
                RecoveryAction.CLEAR_QUEUE: self._clear_queue,
                RecoveryAction.RESET_CONNECTION: self._reset_connection,
                RecoveryAction.FREE_MEMORY: self._free_memory,
                RecoveryAction.DISK_CLEANUP: self._disk_cleanup,
                RecoveryAction.OPTIMIZE_RESOURCES: self._optimize_resources,
                RecoveryAction.RESTART_COMPONENT: self._restart_component,
                RecoveryAction.THROTTLE_TASKS: self._throttle_tasks,
                RecoveryAction.CLEAR_CACHE: self._clear_cache,
                RecoveryAction.REBALANCE_LOAD: self._rebalance_load
            }
            
            if action not in action_methods:
                raise ValueError(f"Unknown recovery action: {action}")
                
            success = await action_methods[action](alert.details)
            
            return RecoveryResult(
                success=success,
                action=action,
                details=alert.details
            )
            
        except Exception as e:
            self.logger.error(f"Action execution error: {str(e)}")
            return RecoveryResult(
                success=False,
                action=action,
                details=alert.details,
                error=str(e)
            )

    async def _restart_strategy(self, details: Dict) -> bool:
        """Restart a trading strategy"""
        try:
            strategy_name = details.get('strategy_name', '')
            await self.trading_system.restart_strategy(strategy_name)
            return True
        except Exception as e:
            self.logger.error(f"Strategy restart error: {str(e)}")
            return False

    async def _pause_trading(self, details: Dict) -> bool:
        """Pause all trading activities"""
        try:
            await self.trading_system.pause_trading()
            return True
        except Exception as e:
            self.logger.error(f"Trading pause error: {str(e)}")
            return False

    async def _resume_trading(self, details: Dict) -> bool:
        """Resume trading activities"""
        try:
            await self.trading_system.resume_trading()
            return True
        except Exception as e:
            self.logger.error(f"Trading resume error: {str(e)}")
            return False

    async def _reduce_position_sizes(self, details: Dict) -> bool:
        """Reduce position sizes for risk management"""
        try:
            reduction_factor = details.get('reduction_factor', 0.5)
            await self.trading_system.reduce_position_sizes(reduction_factor)
            return True
        except Exception as e:
            self.logger.error(f"Position size reduction error: {str(e)}")
            return False

    async def _close_positions(self, details: Dict) -> bool:
        """Close all open positions"""
        try:
            await self.trading_system.close_all_positions()
            return True
        except Exception as e:
            self.logger.error(f"Position closing error: {str(e)}")
            return False

    async def _clear_queue(self, details: Dict) -> bool:
        """Clear event queue"""
        try:
            await self.event_bus.clear_queue()
            return True
        except Exception as e:
            self.logger.error(f"Queue clearing error: {str(e)}")
            return False

    async def _reset_connection(self, details: Dict) -> bool:
        """Reset exchange connection"""
        try:
            await self.trading_system.reset_connection()
            return True
        except Exception as e:
            self.logger.error(f"Connection reset error: {str(e)}")
            return False

    async def _free_memory(self, details: Dict) -> bool:
        """Free up system memory"""
        try:
            gc.collect()
            
            if 'memory_threshold' in details:
                # More aggressive memory cleanup if above threshold
                if psutil.Process().memory_percent() > details['memory_threshold']:
                    import ctypes
                    libc = ctypes.CDLL('libc.so.6')
                    libc.malloc_trim(0)
                    
            return True
        except Exception as e:
            self.logger.error(f"Memory cleanup error: {str(e)}")
            return False

    async def _disk_cleanup(self, details: Dict) -> bool:
        """Clean up disk space"""
        try:
            # Clean up old log files
            if os.path.exists(self.config.LOG_DIRECTORY):
                await self._cleanup_old_files(
                    self.config.LOG_DIRECTORY,
                    days_old=details.get('log_retention_days', 30)
                )
                
            # Clean up old metrics files
            if os.path.exists(self.config.METRICS_ARCHIVE_DIRECTORY):
                await self._cleanup_old_files(
                    self.config.METRICS_ARCHIVE_DIRECTORY,
                    days_old=details.get('metrics_retention_days', 30)
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Disk cleanup error: {str(e)}")
            return False

    async def _optimize_resources(self, details: Dict) -> bool:
        """Optimize system resources"""
        try:
            # Optimize memory usage
            gc.collect()
            
            # Adjust thread pool size if needed
            if hasattr(self.trading_system, 'adjust_thread_pool'):
                await self.trading_system.adjust_thread_pool()
                
            # Optimize database connections if needed
            if hasattr(self.trading_system, 'optimize_db_connections'):
                await self.trading_system.optimize_db_connections()
                
            return True
        except Exception as e:
            self.logger.error(f"Resource optimization error: {str(e)}")
            return False

    async def _restart_component(self, details: Dict) -> bool:
        """Restart a system component"""
        try:
            component_name = details.get('component_name', '')
            
            # Stop component
            if hasattr(self.trading_system, f'stop_{component_name}'):
                await getattr(self.trading_system, f'stop_{component_name}')()
                
            # Wait for cleanup
            await asyncio.sleep(2)
            
            # Start component
            if hasattr(self.trading_system, f'start_{component_name}'):
                await getattr(self.trading_system, f'start_{component_name}')()
                
            return True
        except Exception as e:
            self.logger.error(f"Component restart error: {str(e)}")
            return False

    async def _throttle_tasks(self, details: Dict) -> bool:
        """Throttle task execution"""
        try:
            throttle_factor = details.get('throttle_factor', 0.5)
            max_concurrent = details.get('max_concurrent', None)
            
            if hasattr(self.trading_system, 'throttle_tasks'):
                await self.trading_system.throttle_tasks(
                    factor=throttle_factor,
                    max_concurrent=max_concurrent
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Task throttling error: {str(e)}")
            return False

    async def _clear_cache(self, details: Dict) -> bool:
        """Clear system caches"""
        try:
            # Clear memory cache
            if hasattr(self.trading_system, 'clear_memory_cache'):
                await self.trading_system.clear_memory_cache()
                
            # Clear disk cache
            if hasattr(self.trading_system, 'clear_disk_cache'):
                await self.trading_system.clear_disk_cache()
                
            # Clear database cache
            if hasattr(self.trading_system, 'clear_db_cache'):
                await self.trading_system.clear_db_cache()
                
            return True
        except Exception as e:
            self.logger.error(f"Cache clearing error: {str(e)}")
            return False

    async def _rebalance_load(self, details: Dict) -> bool:
        """Rebalance system load"""
        try:
            # Rebalance tasks
            if hasattr(self.trading_system, 'rebalance_tasks'):
                await self.trading_system.rebalance_tasks()
                
            # Rebalance connections
            if hasattr(self.trading_system, 'rebalance_connections'):
                await self.trading_system.rebalance_connections()
                
            # Rebalance resources
            if hasattr(self.trading_system, 'rebalance_resources'):
                await self.trading_system.rebalance_resources()
                
            return True
        except Exception as e:
            self.logger.error(f"Load rebalancing error: {str(e)}")
            return False

    async def _cleanup_old_files(self, directory: str, days_old: int) -> None:
        """Clean up old files from directory"""
        try:
            current_time = datetime.now()
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (current_time - file_time).days > days_old:
                        os.remove(file_path)
        except Exception as e:
            self.logger.error(f"File cleanup error: {str(e)}")

    def _get_final_result(self, results: List[RecoveryResult]) -> RecoveryResult:
        """Get final recovery result from multiple attempts"""
        if not results:
            return RecoveryResult(
                success=False,
                action=None,
                details={},
                error="No recovery actions attempted"
            )
        
        # Return first successful result or last failed result
        for result in results:
            if result.success:
                return result
        return results[-1]

    def get_recovery_statistics(self) -> Dict:
        """Get recovery system statistics"""
        return {
            'metrics': self.recovery_metrics,
            'active_recoveries': len(self.active_recoveries),
            'total_history': len(self.recovery_history),
            'recent_success_rate': self._calculate_recent_success_rate()
        }

    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate of recent recoveries"""
        try:
            recent_recoveries = self.recovery_history[-100:]  # Last 100 recoveries
            if not recent_recoveries:
                return 0.0
                
            successful = sum(1 for r in recent_recoveries if r.success)
            return successful / len(recent_recoveries)
        except Exception:
            return 0.0

    def clear_history(self):
        """Clear recovery history"""
        self.recovery_history.clear()
        self.last_recovery_time.clear()
        self.recovery_metrics = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }