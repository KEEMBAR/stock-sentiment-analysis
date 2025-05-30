

"""
Data preprocessing module for stock sentiment analysis.

This module provides functions to clean and preprocess the financial news data,
including date parsing, column standardization, and handling missing values.
"""

import pandas as pd
import logging
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    return df.rename(columns=lambda x: str(x).lower().replace(" ", "_"))

def parse_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Parse date column to datetime format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_col (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with parsed dates
        
    Raises:
        ValueError: If date column is not found in DataFrame
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    try:
        # Convert to datetime and remove timezone info to ensure datetime64[ns] dtype
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        logger.info(f"Successfully parsed {date_col} column to datetime")
    except Exception as e:
        logger.error(f"Error parsing dates: {str(e)}")
        raise
    
    return df

def handle_missing_values(df: pd.DataFrame, required_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_cols (list, optional): List of columns that must not have missing values
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if required_cols is None:
        required_cols = ['date', 'headline', 'stock']
    
    # Drop rows with missing values in required columns
    initial_rows = len(df)
    df = df.dropna(subset=required_cols)
    rows_dropped = initial_rows - len(df)
    
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows with missing values in {required_cols}")
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main preprocessing function that applies all preprocessing steps.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
        
    Raises:
        ValueError: If input DataFrame is empty or required columns are missing
    """
    try:
        logger.info("Starting data preprocessing")
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Apply preprocessing steps
        df = standardize_column_names(df)
        df = parse_dates(df)
        df = handle_missing_values(df)
        
        # Assuming df_clean is your preprocessed DataFrame
        df['headline_length'] = df['headline'].str.len()

        # Basic statistics
        print(df['headline_length'].describe())

        # Optional: Visualize
        plt.figure(figsize=(10, 5))
        sns.histplot(df['headline_length'], bins=50, kde=True)
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length')
        plt.ylabel('Frequency')
        plt.show()
        
        publisher_counts = df['publisher'].value_counts()
        print(publisher_counts.head(10))  # Top 10 publishers

        # Optional: Visualize
        plt.figure(figsize=(12, 5))
        sns.barplot(x=publisher_counts.head(10).index, y=publisher_counts.head(10).values)
        plt.title('Top 10 Most Active Publishers')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45, ha='right')
        plt.show()
        
        # Ensure 'date' is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Count articles per day
        daily_counts = df.groupby(df['date'].dt.date).size()

        plt.figure(figsize=(15, 5))
        daily_counts.plot()
        plt.title('Number of Articles Published per Day')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.show()
        
        logger.info("Data preprocessing completed successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
