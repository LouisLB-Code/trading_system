# tests/unit/test_data_validation.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.exceptions import DataValidationError  # You'll need to create this

class TestDataValidation:
    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        self.config = TestConfig()
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
    def test_market_data_validation(self, setup):
        """Test validation of market data input"""
        # Test valid data
        valid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 1100]
        }, index=pd.date_range(start='2024-01-01', periods=2))
        
        assert self.validate_market_data(valid_data) == True
        
    def test_missing_columns(self, setup):
        """Test detection of missing required columns"""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'close': [101, 102],  # Missing high, low, volume
        })
        
        with pytest.raises(DataValidationError) as exc_info:
            self.validate_market_data(invalid_data)
        assert "Missing required columns" in str(exc_info.value)
        
    def test_invalid_data_types(self, setup):
        """Test detection of invalid data types"""
        invalid_types_data = pd.DataFrame({
            'open': [100, 'invalid'],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        with pytest.raises(DataValidationError) as exc_info:
            self.validate_market_data(invalid_types_data)
        assert "Invalid data type" in str(exc_info.value)
        
    def test_data_consistency(self, setup):
        """Test data consistency validation"""
        # Test high < low
        inconsistent_data = pd.DataFrame({
            'open': [100, 101],
            'high': [98, 99],    # High is less than low
            'low': [102, 103],
            'close': [101, 102],
            'volume': [1000, 1100]
        }, index=pd.date_range(start='2024-01-01', periods=2))
        
        with pytest.raises(DataValidationError) as exc_info:
            self.validate_market_data(inconsistent_data)
        assert "Price consistency error" in str(exc_info.value)
        
    def test_negative_values(self, setup):
        """Test validation of negative values"""
        negative_values_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, -100]  # Negative volume
        }, index=pd.date_range(start='2024-01-01', periods=2))
        
        with pytest.raises(DataValidationError) as exc_info:
            self.validate_market_data(negative_values_data)
        assert "Negative values not allowed" in str(exc_info.value)
        
    def validate_market_data(self, data):
        """Helper method to validate market data"""
        # Check for required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise DataValidationError(f"Invalid data type in column {col}")
                
        # Check price consistency
        if any(data['high'] < data['low']):
            raise DataValidationError("Price consistency error: high < low")
            
        # Check for negative values
        if any(data[col].lt(0).any() for col in self.required_columns):
            raise DataValidationError("Negative values not allowed")
            
        return True
