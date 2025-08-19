#!/usr/bin/env python3
"""
Script to combine results from multiple chunk jobs into a single file.
Usage: python combine_chunk_results.py <results_directory> <output_file>
"""

import os
import sys
import pandas as pd
import glob
from pathlib import Path

def combine_chunk_results(results_dir, output_file):
    """Combine all chunk results into a single file"""
    
    # Find all chunk result files
    chunk_pattern = os.path.join(results_dir, "chunk_*_job_*.parquet")
    chunk_files = glob.glob(chunk_pattern)
    
    if not chunk_files:
        print(f"No chunk files found in {results_dir}")
        print(f"Looking for pattern: {chunk_pattern}")
        return
    
    print(f"Found {len(chunk_files)} chunk files:")
    for f in sorted(chunk_files):
        print(f"  - {os.path.basename(f)}")
    
    # Read and combine all chunks
    all_dataframes = []
    total_samples = 0
    
    for chunk_file in sorted(chunk_files):
        try:
            df = pd.read_parquet(chunk_file)
            all_dataframes.append(df)
            total_samples += len(df)
            print(f"Loaded {len(df)} samples from {os.path.basename(chunk_file)}")
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
            continue
    
    if not all_dataframes:
        print("No valid chunk files found!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save combined results
    combined_df.to_parquet(output_file, index=False)
    
    print(f"\nCombined {len(all_dataframes)} chunks into {output_file}")
    print(f"Total samples: {total_samples}")
    print(f"Combined file shape: {combined_df.shape}")
    
    # Show sample statistics
    print(f"\nSample statistics:")
    print(f"  - Total samples: {len(combined_df)}")
    if 'iters' in combined_df.columns:
        print(f"  - Average iterations: {combined_df['iters'].mean():.2f}")
    if 'solution' in combined_df.columns:
        print(f"  - Samples with solutions: {combined_df['solution'].notna().sum()}")
    
    return combined_df

def main():
    if len(sys.argv) != 3:
        print("Usage: python combine_chunk_results.py <results_directory> <output_file>")
        print("Example: python combine_chunk_results.py results combined_results.parquet")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist!")
        sys.exit(1)
    
    print(f"Combining chunk results from {results_dir} into {output_file}")
    combine_chunk_results(results_dir, output_file)

if __name__ == "__main__":
    main()
