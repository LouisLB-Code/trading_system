# src/exceptions.py
class DataValidationError(Exception):
    """Exception raised for data validation errors"""
    pass

class OrderExecutionError(Exception):
    """Exception raised for order execution errors"""
    pass

class StrategyError(Exception):
    """Exception raised for strategy-related errors"""
    pass

class SystemError(Exception):
    """Exception raised for system-level errors"""
    pass
