import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load customer behavior data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df


def check_missing_values(df):
    """
    Check and report missing values in the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value counts and percentages
    """
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create summary dataframe
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_percentage.round(2)
    })
    
    # Only return columns with missing values
    missing_info = missing_info[missing_info['Missing Values'] > 0]
    
    if len(missing_info) > 0:
        print("Missing values found:")
        return missing_info
    else:
        print("No missing values found.")
        return missing_info


def handle_missing_values(df):
    """
    Handle missing values in the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    print("Handling missing values...")
    df_clean = df.copy()
    
    # Check if there are any rows with multiple missing values
    rows_with_multiple_missing = df_clean[df_clean.isnull().sum(axis=1) > 1]
    if len(rows_with_multiple_missing) > 0:
        print(f"Removing {len(rows_with_multiple_missing)} rows with multiple missing values")
        df_clean = df_clean.dropna(thresh=df_clean.shape[1]-1)
    
    # For the remaining missing values, fill with median for numerical columns
    for column in df_clean.columns:
        if df_clean[column].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[column]):
                median_value = df_clean[column].median()
                df_clean[column].fillna(median_value, inplace=True)
                print(f"Filled missing values in '{column}' with median: {median_value}")
            else:
                mode_value = df_clean[column].mode()[0]
                df_clean[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in '{column}' with mode: {mode_value}")
    
    return df_clean


def detect_outliers(df, columns=None):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to check for outliers
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and outlier indices as values
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    outliers = {}
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        
        if len(outlier_indices) > 0:
            outliers[column] = outlier_indices
            print(f"Found {len(outlier_indices)} outliers in '{column}'")
    
    return outliers


def normalize_data(df, exclude_columns=None):
    """
    Normalize numerical data using StandardScaler
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_columns : list, optional
        List of columns to exclude from normalization
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with normalized features
        
    StandardScaler
        Fitted scaler for future transformations
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Create a copy of the dataframe
    df_normalized = df.copy()
    
    # Select numeric columns for normalization
    numeric_columns = df_normalized.select_dtypes(include=['number']).columns
    columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
    
    # Apply normalization
    scaler = StandardScaler()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
    
    print(f"Normalized {len(columns_to_normalize)} columns: {columns_to_normalize}")
    
    return df_normalized, scaler


def preprocess_data(file_path):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    tuple
        (original_df, preprocessed_df, scaler)
    """
    # Load the data
    df = load_data(file_path)
    
    # Store the original dataframe
    original_df = df.copy()
    
    # Check missing values
    missing_info = check_missing_values(df)
    
    # Handle missing values
    df_clean = handle_missing_values(df)
    
    # Detect outliers
    outliers = detect_outliers(df_clean)
    
    # For this segmentation task, we'll keep the outliers as they might be valid customer behaviors
    
    # We'll exclude customer_id from normalization as it's an identifier
    exclude_from_normalization = ['customer_id']
    
    # Normalize the data
    df_normalized, scaler = normalize_data(df_clean, exclude_columns=exclude_from_normalization)
    
    print("Preprocessing completed successfully!")
    
    return original_df, df_normalized, scaler


if __name__ == "__main__":
    # Example usage
    file_path = "../data/customer_behavior_analytcis.csv"
    original_df, df_normalized, scaler = preprocess_data(file_path)
    
    print("\nOriginal Data:")
    print(original_df.head())
    
    print("\nPreprocessed Data:")
    print(df_normalized.head())