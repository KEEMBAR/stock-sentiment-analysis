"""
Tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from scripts.preprocess import (
    standardize_column_names,
    parse_dates,
    handle_missing_values,
    preprocess_data
)

@pytest.fixture
def sample_raw_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Headline': ['Test headline 1', 'Test headline 2'],
        'URL': ['http://test1.com', 'http://test2.com'],
        'Publisher': ['Publisher 1', 'Publisher 2'],
        'Date': ['2023-01-01 10:00:00-04:00', '2023-01-02 11:00:00-04:00'],
        'Stock': ['AAPL', 'GOOGL']
    })

def test_standardize_column_names(sample_raw_df):
    """Test column name standardization."""
    df = standardize_column_names(sample_raw_df)
    expected_columns = ['headline', 'url', 'publisher', 'date', 'stock']
    assert all(col in df.columns for col in expected_columns)
    assert all(col.islower() for col in df.columns)
    assert not any(' ' in col for col in df.columns)

def test_parse_dates(sample_raw_df):
    """Test date parsing."""
    df = standardize_column_names(sample_raw_df)
    df = parse_dates(df)
    assert pd.api.types.is_datetime64_dtype(df['date'])

def test_parse_dates_invalid_column():
    """Test date parsing with invalid column name."""
    df = pd.DataFrame({'wrong_col': ['2023-01-01']})
    with pytest.raises(ValueError, match="Date column 'date' not found"):
        parse_dates(df)

def test_handle_missing_values():
    """Test handling of missing values."""
    df = pd.DataFrame({
        'date': ['2023-01-01', None],
        'headline': ['Test', 'Test'],
        'stock': ['AAPL', 'GOOGL']
    })
    df = handle_missing_values(df)
    assert len(df) == 1  # One row should be dropped
    assert df['date'].isna().sum() == 0

def test_preprocess_data_empty_df():
    """Test preprocessing with empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        preprocess_data(df)

def test_preprocess_data_full_pipeline(sample_raw_df):
    """Test the full preprocessing pipeline."""
    df = preprocess_data(sample_raw_df)
    
    # Check if all preprocessing steps were applied
    assert all(col.islower() for col in df.columns)
    assert pd.api.types.is_datetime64_dtype(df['date'])
    assert df.isna().sum().sum() == 0  # No missing values
    assert len(df) == len(sample_raw_df)  # No rows should be dropped 