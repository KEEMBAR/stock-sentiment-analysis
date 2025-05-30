"""
Data loading module for stock sentiment analysis project.

This module provides functions to load and validate financial news data,
with proper error handling and data validation.
"""

import pandas as pd
import os
import logging
from typing import Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate the loaded data for required columns and data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame to validate
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    required_columns = ['headline', 'url', 'publisher', 'date', 'stock']
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        return False, f"Found null values in columns: {null_counts[null_counts > 0].to_dict()}"
    
    # Validate date format robustly
    parsed_dates = pd.to_datetime(df['date'], errors='coerce')
    if parsed_dates.isnull().any():
        bad_dates = df.loc[parsed_dates.isnull(), 'date'].unique()
        return False, f"Some dates could not be parsed: {bad_dates[:5]}{'...' if len(bad_dates) > 5 else ''}"
        
    return True, None

def load_data(path: str, validate: bool = True) -> pd.DataFrame:
    """
    Load financial news data from CSV file with validation.
    
    Args:
        path (str): Path to the CSV file
        validate (bool): Whether to validate the data after loading
        
    Returns:
        pd.DataFrame: Loaded and validated DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        DataValidationError: If data validation fails
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        logger.info(f"Loading data from {path}")
        
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load the data
        df = pd.read_csv(path)
        
        if df.empty:
            raise pd.errors.EmptyDataError("The loaded file is empty")
        
        # Validate data if requested
        if validate:
            is_valid, error_msg = validate_data(df)
            if not is_valid:
                raise DataValidationError(error_msg)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

