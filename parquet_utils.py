#!/usr/bin/env python3
"""
Utility functions for working with parquet files
"""

import pandas as pd
import os
from pathlib import Path

def read_parquet_file(file_path):
    """
    Read a parquet file and return basic information about it
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        dict: Dictionary containing file information
    """
    try:
        df = pd.read_parquet(file_path)
        info = {
            'file_path': file_path,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'is_empty': df.empty,
            'sample_data': df.head(3).to_dict('records') if not df.empty else None
        }
        return info, df
    except Exception as e:
        return {'error': str(e)}, None

def analyze_parquet_directory(directory_path):
    """
    Analyze all parquet files in a directory
    
    Args:
        directory_path (str): Path to directory containing parquet files
        
    Returns:
        list: List of file information dictionaries
    """
    results = []
    directory = Path(directory_path)
    
    for parquet_file in directory.glob("*.parquet"):
        info, _ = read_parquet_file(str(parquet_file))
        results.append(info)
    
    return results

def save_to_parquet(df, file_path):
    """
    Save a pandas DataFrame to parquet format
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        file_path (str): Output file path
    """
    df.to_parquet(file_path, index=False)
    print(f"Saved DataFrame to {file_path}")

def filter_parquet_data(file_path, conditions=None, columns=None):
    """
    Filter parquet data based on conditions
    
    Args:
        file_path (str): Path to parquet file
        conditions (dict): Dictionary of column: value pairs for filtering
        columns (list): List of columns to select
        
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    df = pd.read_parquet(file_path)
    
    if conditions:
        for column, value in conditions.items():
            if column in df.columns:
                df = df[df[column] == value]
    
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
    
    return df

def main():
    """Example usage of the utility functions"""
    
    # Example 1: Read and analyze a specific parquet file
    print("=== Analyzing job_3414290.parquet ===")
    info, df = read_parquet_file('results/job_3414290.parquet')
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n=== Analyzing all parquet files in results directory ===")
    all_files_info = analyze_parquet_directory('results')
    for file_info in all_files_info:
        print(f"\nFile: {file_info.get('file_path', 'Unknown')}")
        print(f"Shape: {file_info.get('shape', 'Unknown')}")
        print(f"Empty: {file_info.get('is_empty', 'Unknown')}")
        if file_info.get('columns'):
            print(f"Columns: {file_info['columns']}")

if __name__ == "__main__":
    main()
