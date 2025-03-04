import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def clean_beer_data(df):
    """
    Clean beer dataset by handling missing values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Fill missing values in specific columns
    if 'Weighted Distribution Any Promo' in cleaned_df.columns:
        cleaned_df["Weighted Distribution Any Promo"].fillna(0, inplace=True)
    
    # Drop remaining rows with any null values
    cleaned_df = cleaned_df.dropna(axis=0, how='any')
    
    return cleaned_df

def encode_categorical_variables(df, columns_to_encode=None):
    """
    Encode categorical variables using LabelEncoder
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns_to_encode : list or None
        List of columns to encode. If None, no encoding is performed.
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with encoded variables
    """
    if df is None or columns_to_encode is None:
        return df
        
    # Create a copy to avoid modifying the original
    encoded_df = df.copy()
    
    # Initialize encoder
    le = LabelEncoder()
    
    # Encode specified columns
    for col in columns_to_encode:
        if col in encoded_df.columns:
            encoded_df[col] = le.fit_transform(encoded_df[col])
    
    return encoded_df

def get_numeric_columns(df):
    """
    Get list of numeric columns from dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
        
    Returns:
    --------
    list
        List of numeric column names
    """
    if df is None:
        return []
        
    return df.select_dtypes(include=[np.number]).columns.tolist()

def detect_column_types(df):
    """
    Automatically detect categorical and numerical columns in a DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to analyze column types
    
    Returns:
    --------
    tuple
        A tuple containing two lists: (categorical_columns, numeric_columns)
    """
    # Import required libraries
    import numpy as np
    import pandas as pd

    # Check for input validity
    if df is None or df.empty:
        return [], []

    # Detect column types
    categorical_columns = []
    numeric_columns = []

    for col in df.columns:
        # Get the column's data type
        dtype = df[col].dtype

        # Check for categorical conditions
        is_categorical = (
            # Object/string columns
            dtype == 'object' or 
            # Columns with few unique values relative to total rows
            (df[col].nunique() < len(df) * 0.1) or 
            # Boolean columns
            dtype == 'bool'
        )

        # Columns that are not categorical are considered numeric
        if is_categorical:
            categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    return categorical_columns, numeric_columns