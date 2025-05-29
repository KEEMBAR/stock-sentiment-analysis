"""
Tests for the data loader module.
"""

import pytest
import pandas as pd
import os
from src.data_loader import load_data, validate_data, DataValidationError

@pytest.fixture
def sample_valid_df():
    """Create a sample valid DataFrame for testing."""
    return pd.DataFrame({
        'headline': ['Test headline'],
        'url': ['http://test.com'],
        'publisher': ['Test Publisher'],
        'date': ['2023-01-01 10:00:00-04:00'],
        'stock': ['AAPL']
    })

def test_validate_data_valid(sample_valid_df):
    """Test validation with valid data."""
    is_valid, error_msg = validate_data(sample_valid_df)
    assert is_valid
    assert error_msg is None

def test_validate_data_missing_columns():
    """Test validation with missing columns."""
    df = pd.DataFrame({'headline': ['Test']})  # Missing required columns
    is_valid, error_msg = validate_data(df)
    assert not is_valid
    assert "Missing required columns" in error_msg

def test_validate_data_null_values(sample_valid_df):
    """Test validation with null values."""
    df = sample_valid_df.copy()
    df.loc[0, 'headline'] = None
    is_valid, error_msg = validate_data(df)
    assert not is_valid
    assert "Found null values" in error_msg

def test_load_data_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")

def test_load_data_invalid_date():
    """Test loading data with invalid date format."""
    # This test would need a temporary file with invalid date format
    pass  # Implement when needed

# Add more tests as needed 