# How to Use Parquet Files

## Overview
Parquet files are a columnar storage format that's efficient for data analysis and processing. In this project, parquet files are used to store job-related data and results.

## Current Status
- `job_3414290.parquet` is currently empty (0 rows, 0 columns)
- Other parquet files in the `results/` directory contain actual data

## Basic Usage

### 1. Reading a Parquet File
```python
import pandas as pd

# Read a parquet file
df = pd.read_parquet('results/job_3414290.parquet')

# Check basic information
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Empty: {df.empty}")
```

### 2. Creating and Saving Parquet Files
```python
import pandas as pd

# Create sample data
data = {
    'job_id': [1, 2, 3],
    'status': ['completed', 'running', 'failed'],
    'duration': [120, 45, 0]
}
df = pd.DataFrame(data)

# Save to parquet
df.to_parquet('results/my_job_data.parquet', index=False)
```

### 3. Using the Utility Scripts

#### Analyze all parquet files:
```bash
python parquet_utils.py
```

#### Run the example usage:
```bash
python example_parquet_usage.py
```

## Advanced Operations

### Filtering Data
```python
# Read and filter
df = pd.read_parquet('results/job_789992.parquet')
completed_jobs = df[df['status'] == 'completed']
```

### Reading Specific Columns
```python
# Only load specific columns (more memory efficient)
selected_columns = ['job_id', 'status', 'duration']
df = pd.read_parquet('results/job_789992.parquet', columns=selected_columns)
```

### Aggregations
```python
# Group by status and calculate statistics
stats = df.groupby('status').agg({
    'duration': ['mean', 'count'],
    'job_id': 'count'
})
```

## Available Tools

### 1. `parquet_utils.py`
- `read_parquet_file(file_path)`: Read and analyze a single parquet file
- `analyze_parquet_directory(directory_path)`: Analyze all parquet files in a directory
- `save_to_parquet(df, file_path)`: Save DataFrame to parquet format
- `filter_parquet_data(file_path, conditions, columns)`: Filter data based on conditions

### 2. `example_parquet_usage.py`
- Demonstrates basic operations
- Shows how to create sample data
- Includes advanced operations like grouping and time-based analysis

## Common Use Cases

### 1. Job Monitoring
```python
# Monitor job status
df = pd.read_parquet('results/job_789992.parquet')
status_counts = df['status'].value_counts()
print(f"Job status distribution:\n{status_counts}")
```

### 2. Performance Analysis
```python
# Analyze job performance
df = pd.read_parquet('results/job_789992.parquet')
if 'duration' in df.columns:
    avg_duration = df['duration'].mean()
    print(f"Average job duration: {avg_duration:.2f} seconds")
```

### 3. Data Export
```python
# Export to other formats
df = pd.read_parquet('results/job_789992.parquet')
df.to_csv('results/job_789992.csv', index=False)
df.to_json('results/job_789992.json', orient='records')
```

## Troubleshooting

### Empty Parquet Files
If a parquet file is empty (like `job_3414290.parquet`):
1. Check if the job that should populate it completed successfully
2. Verify the file creation process in your job scripts
3. Consider regenerating the data

### Memory Issues
For large parquet files:
1. Use `columns` parameter to load only needed columns
2. Use `chunksize` for iterative processing
3. Consider using `dask` for very large datasets

### File Not Found
Ensure the file path is correct and the file exists:
```python
import os
if os.path.exists('results/job_3414290.parquet'):
    df = pd.read_parquet('results/job_3414290.parquet')
else:
    print("File not found")
```

## Next Steps
1. Run `python parquet_utils.py` to analyze your current parquet files
2. Use `python example_parquet_usage.py` to see practical examples
3. Modify the example scripts to work with your specific data structure
4. Integrate parquet operations into your existing job processing pipeline
