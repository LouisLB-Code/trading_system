# tests/conftest.py
import pytest
from src.core.enhanced_system import EnhancedSystem
from src.core.state_management import StateManager
from src.core.strategy.strategy_manager import StrategyManager
from config import TestConfig
from src.core.error_handling import (
    AdvancedErrorHandler,
    ErrorPerformanceTracker,
    ErrorPatternAnalyzer,
    RecoveryStrategyOptimizer
)

@pytest.fixture
def test_config():
    """Global test configuration"""
    return TestConfig()

@pytest.fixture
def trading_system(test_config):
    """Initialize trading system for tests"""
    return EnhancedSystem(test_config)

@pytest.fixture
def state_manager(test_config):
    """Initialize state manager for tests"""
    return StateManager(test_config)

@pytest.fixture
def strategy_manager(test_config):
    """Initialize strategy manager for tests"""
    return StrategyManager(test_config)

@pytest.fixture
def error_handler():
    """Initialize error handler for tests"""
    return AdvancedErrorHandler()

@pytest.fixture
def performance_tracker():
    """Initialize performance tracker for tests"""
    return ErrorPerformanceTracker()

@pytest.fixture
def pattern_analyzer():
    """Initialize pattern analyzer for tests"""
    return ErrorPatternAnalyzer()

@pytest.fixture
def strategy_optimizer():
    """Initialize strategy optimizer for tests"""
    return RecoveryStrategyOptimizer()

@pytest.fixture
def mock_system_state():
    """Create a mock system state"""
    return {
        "status": "active",
        "active_orders": [],
        "positions": {},
        "last_update": datetime.now(),
        "error_count": 0
    }
