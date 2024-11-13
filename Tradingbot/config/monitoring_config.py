from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    
    # Existing thresholds
    CPU_THRESHOLD: float = 80.0
    MEMORY_THRESHOLD: float = 80.0
    DISK_USAGE_THRESHOLD: float = 90.0
    MIN_DISK_SPACE: float = 10.0
    MIN_MEMORY: float = 20.0
    MAX_QUEUE_SIZE: int = 1000
    LATENCY_THRESHOLD: float = 1000.0
    MAX_NETWORK_LATENCY: float = 500.0
    
    # Existing performance thresholds
    MAX_DRAWDOWN_THRESHOLD: float = 0.20
    MIN_WIN_RATE: float = 0.40
    MIN_SHARPE_RATIO: float = 1.0
    MIN_EXECUTION_RATE: float = 0.80
    MIN_STRATEGY_SUCCESS_RATE: float = 0.50
    
    # Existing intervals
    SYSTEM_CHECK_INTERVAL: int = 60
    PERFORMANCE_CHECK_INTERVAL: int = 300
    STRATEGY_CHECK_INTERVAL: int = 120
    RESOURCE_CHECK_INTERVAL: int = 180
    
    # Existing history settings
    MAX_METRICS_HISTORY: int = 1000
    MAX_ALERT_HISTORY: int = 500
    
    # Existing alert settings
    ALERT_COOLDOWN_PERIOD: int = 300
    MAX_ACTIVE_ALERTS: int = 100
    
    # Existing recovery settings
    AUTO_RECOVERY_ENABLED: bool = True
    MAX_RECOVERY_ATTEMPTS: int = 3
    RECOVERY_COOLDOWN: int = 600

    # New External monitoring settings
    DATADOG_ENABLED: bool = False
    PROMETHEUS_ENABLED: bool = False
    GRAFANA_ENABLED: bool = False
    CLOUDWATCH_ENABLED: bool = False
    
    # New Path settings
    LOG_DIRECTORY: str = "logs"
    LOG_ARCHIVE_DIRECTORY: str = "logs/archive"
    METRICS_ARCHIVE_DIRECTORY: str = "data/metrics"
    
    # New Retention settings
    METRICS_RETENTION_DAYS: int = 30
    ALERTS_RETENTION_DAYS: int = 90
    LOGS_RETENTION_DAYS: int = 30
    
    # New Network monitoring settings
    MAX_PACKET_LOSS: float = 0.1
    MAX_CONNECTIONS: int = 1000
    CONNECTION_TIMEOUT: int = 30
    RETRY_ATTEMPTS: int = 3
    
    # New Task monitoring settings
    MAX_TASK_DURATION: int = 300
    MAX_BLOCKED_TASKS: int = 5
    TASK_TIMEOUT_SECONDS: int = 600
    
    # New Analysis settings
    HEALTH_SCORE_THRESHOLD: float = 80.0
    MIN_FORECAST_POINTS: int = 30
    FORECAST_POINTS: int = 10
    ANOMALY_THRESHOLD: float = 3.0
    
    # New System maintenance settings
    GC_INTERVAL_SECONDS: int = 600
    MAX_LOG_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_FILE_DESCRIPTORS: int = 1000
    
    # Recovery strategies
    RECOVERY_STRATEGIES: Dict[str, List[str]] = {
        "memory": ["gc_collect", "clear_caches", "restart_component"],
        "cpu": ["throttle_tasks", "optimize_scheduling"],
        "network": ["reset_connections", "switch_endpoints"],
        "task": ["cancel_stuck", "reset_queues", "restart_scheduler"]
    }

    # Feature flags
    DEBUG: bool = False
    DETAILED_LOGGING: bool = True
    AUTO_OPTIMIZATION_ENABLED: bool = True