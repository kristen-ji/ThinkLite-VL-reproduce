#!/usr/bin/env python3
"""
Example usage of parquet files
"""

import pandas as pd
import numpy as np
from parquet_utils import read_parquet_file, analyze_parquet_directory

def basic_parquet_operations():
    """Demonstrate basic parquet operations"""
    
    print("=== Basic Parquet Operations ===\n")
    
    # 1. Reading a parquet file
    print("1. Reading parquet file:")
    try:
        df = pd.read_parquet('results/job_3414290.parquet')
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Empty: {df.empty}")
    except Exception as e:
        print(f"   Error reading file: {e}")
    
    # 2. Creating and saving a sample parquet file
    print("\n2. Creating and saving sample data:")
    sample_data = {
        'job_id': [1, 2, 3, 4, 5],
        'status': ['completed', 'running', 'failed', 'completed', 'pending'],
        'duration': [120, 45, 0, 180, 0],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
    }
    sample_df = pd.DataFrame(sample_data)
    
    output_file = 'results/sample_job_data.parquet'
    sample_df.to_parquet(output_file, index=False)
    print(f"   Saved sample data to {output_file}")
    print(f"   Sample data shape: {sample_df.shape}")
    
    # 3. Reading the sample file back
    print("\n3. Reading the sample file back:")
    loaded_df = pd.read_parquet(output_file)
    print(f"   Loaded data shape: {loaded_df.shape}")
    print(f"   First few rows:")
    print(loaded_df.head())
    
    # 4. Filtering and querying
    print("\n4. Filtering data:")
    completed_jobs = loaded_df[loaded_df['status'] == 'completed']
    print(f"   Completed jobs: {len(completed_jobs)}")
    print(completed_jobs[['job_id', 'duration']])
    
    # 5. Aggregations
    print("\n5. Aggregations:")
    status_counts = loaded_df['status'].value_counts()
    print(f"   Status distribution:")
    print(status_counts)
    
    avg_duration = loaded_df['duration'].mean()
    print(f"   Average duration: {avg_duration:.2f}")

def analyze_existing_files():
    """Analyze existing parquet files in the results directory"""
    
    print("\n=== Analyzing Existing Files ===\n")
    
    # Use the utility function to analyze all parquet files
    files_info = analyze_parquet_directory('results')
    
    for file_info in files_info:
        print(f"File: {file_info.get('file_path', 'Unknown')}")
        print(f"  Shape: {file_info.get('shape', 'Unknown')}")
        print(f"  Empty: {file_info.get('is_empty', 'Unknown')}")
        
        if file_info.get('columns'):
            print(f"  Columns: {file_info['columns']}")
        
        if file_info.get('sample_data'):
            print(f"  Sample data: {file_info['sample_data']}")
        
        print()

def advanced_operations():
    """Demonstrate more advanced parquet operations"""
    
    print("=== Advanced Operations ===\n")
    
    # Create a larger dataset for demonstration
    print("1. Creating larger dataset:")
    np.random.seed(42)
    large_data = {
        'job_id': range(1, 1001),
        'user_id': np.random.randint(1, 51, 1000),
        'priority': np.random.choice(['low', 'medium', 'high'], 1000),
        'cpu_usage': np.random.normal(50, 20, 1000),
        'memory_usage': np.random.normal(60, 15, 1000),
        'status': np.random.choice(['running', 'completed', 'failed', 'pending'], 1000),
        'created_at': pd.date_range('2024-01-01', periods=1000, freq='10min')
    }
    
    large_df = pd.DataFrame(large_data)
    large_df.to_parquet('results/large_job_data.parquet', index=False)
    print(f"   Created large dataset with {len(large_df)} rows")
    
    # 2. Reading with specific columns
    print("\n2. Reading specific columns:")
    selected_columns = ['job_id', 'priority', 'status']
    partial_df = pd.read_parquet('results/large_job_data.parquet', columns=selected_columns)
    print(f"   Loaded {len(partial_df)} rows with columns: {partial_df.columns.tolist()}")
    
    # 3. Filtering during read (if supported by your pandas version)
    print("\n3. Filtering data:")
    high_priority = large_df[large_df['priority'] == 'high']
    print(f"   High priority jobs: {len(high_priority)}")
    
    # 4. Grouping and aggregations
    print("\n4. Grouping and aggregations:")
    status_stats = large_df.groupby('status').agg({
        'cpu_usage': ['mean', 'std'],
        'memory_usage': ['mean', 'std'],
        'job_id': 'count'
    }).round(2)
    print(status_stats)
    
    # 5. Time-based operations
    print("\n5. Time-based operations:")
    large_df['hour'] = large_df['created_at'].dt.hour
    hourly_jobs = large_df.groupby('hour')['job_id'].count()
    print(f"   Jobs created by hour:")
    print(hourly_jobs.head(10))

if __name__ == "__main__":
    basic_parquet_operations()
    analyze_existing_files()
    advanced_operations()
