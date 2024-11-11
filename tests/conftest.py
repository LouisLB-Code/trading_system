# tests/conftest.py
import pytest
from src.core.enhanced_system import EnhancedSystem
from src.core.state_management import StateManager
from src.core.strategy.strategy_manager import StrategyManager
from config import TestConfig

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
