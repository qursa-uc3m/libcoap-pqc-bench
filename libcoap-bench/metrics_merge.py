import pandas as pd
import sys
import os
import glob
import numpy as np
import argparse
from datetime import datetime

# Function to read a CSV and calculate the mode for each column
def calculate_modes(input_csv):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv, sep=';')
    
    # Calculate the mode for each column
    modes = df.mode().iloc[0]
    
    return modes

# Function to append the modes to another CSV
def append_modes_to_csv(modes, output_csv):
    # Create a DataFrame with the modes
    mode_df = pd.DataFrame([modes], columns=modes.index)
    
    # Read the output CSV file into a DataFrame
    output_df = pd.read_csv(output_csv, sep=';')
    
    # Append the mode_df to the output_df for each row
    num_rows = len(output_df)
    mode_df = pd.concat([mode_df] * num_rows, ignore_index=True)
    mode_df.iloc[-1] = 0
    mode_df.iloc[-3] = "------------"   
    # Add column names with '_mode' suffix to avoid collision
    # mode_df.columns = [f'{col}' for col in mode_df.columns]
    
    # Concatenate the original DataFrame with the modes DataFrame
    result_df = pd.concat([output_df, mode_df], axis=1)
    
    # Write the result to the same output CSV file
    result_df.to_csv(output_csv, index=False, sep=";")
    result_df.columns = result_df.columns.map(lambda x: ';'.join(x))

# Function to aggregate metrics across multiple iterations
def aggregate_metrics(iteration_dirs, file_pattern, output_dir):
    """
    Aggregate metrics from the same CSV file across multiple iteration directories.
    
    Args:
        iteration_dirs: List of iteration directory paths
        file_pattern: Pattern of CSV files to aggregate (e.g., "udp_rasp_conv_stats_*.csv")
        output_dir: Directory where aggregated results will be saved
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all unique file names across all iteration directories
    all_files = set()
    for dir_path in iteration_dirs:
        files = glob.glob(os.path.join(dir_path, file_pattern))
        all_files.update([os.path.basename(f) for f in files])
    
    # Process each unique file
    for filename in sorted(all_files):
        # Collect DataFrames from each iteration
        iteration_dfs = []
        iteration_means = []
        iteration_stds = []
        
        # Track if file exists in all iterations
        found_in_all_iterations = True
        
        for dir_path in iteration_dirs:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                try:
                    # Read CSV with correct type inference
                    df = pd.read_csv(file_path, sep=';')
                    
                    # Convert separator row to string explicitly to avoid numeric conversion issues
                    separator_idx = df[df.iloc[:, 0] == '------------'].index
                    if len(separator_idx) > 0:
                        for col in df.columns:
                            df.loc[separator_idx, col] = '------------'
                    
                    # Store the full DataFrame
                    iteration_dfs.append(df)
                    
                    # Extract mean values (second to last row)
                    means = df.iloc[-2].copy()
                    
                    # Convert to numeric, errors='coerce' will convert non-numeric to NaN
                    means = means.apply(pd.to_numeric, errors='coerce')
                    iteration_means.append(means)
                    
                    # Extract standard deviation values (last row)
                    stds = df.iloc[-1].copy()
                    
                    # Convert to numeric, errors='coerce' will convert non-numeric to NaN
                    stds = stds.apply(pd.to_numeric, errors='coerce')
                    iteration_stds.append(stds)
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    found_in_all_iterations = False
                    break
            else:
                print(f"Warning: File {filename} not found in {dir_path}")
                found_in_all_iterations = False
                break
        
        # Skip files that don't exist in all iterations
        if not found_in_all_iterations or len(iteration_dfs) == 0:
            print(f"Skipping {filename} - not present in all iterations or error reading")
            continue
            
        # Take the first DataFrame as a template for the aggregated result
        template_df = iteration_dfs[0].copy()
        
        # Remove the last two rows (mean and std dev)
        if len(template_df) >= 2:
            data_rows = template_df.iloc[:-2].copy()
        else:
            data_rows = pd.DataFrame(columns=template_df.columns)
        
        # Convert means and stds to DataFrame for easier calculations
        means_df = pd.DataFrame(iteration_means)
        stds_df = pd.DataFrame(iteration_stds)
        
        # Get number of iterations and samples per iteration
        k = len(iteration_dfs)  # Number of iterations
        
        if len(data_rows) > 0:
            n = len(data_rows)    # Number of samples per iteration (assuming equal)
        else:
            # If there are no data rows, use default value
            n = 1
        
        N = n * k  # Total number of samples
        
        # Calculate the overall mean (mean of means)
        overall_mean = means_df.mean(numeric_only=True)
        
        # Calculate the average variance across iterations
        avg_variance = (stds_df ** 2).mean(numeric_only=True)
        
        # Calculate the variance of means
        if k > 1:
            means_variance = means_df.var(ddof=1, numeric_only=True)  # Use ddof=1 for sample variance
        else:
            means_variance = pd.Series(0, index=means_df.columns)
            for col in means_df.columns:
                if pd.api.types.is_numeric_dtype(means_df[col]):
                    means_variance[col] = 0
        
        # Calculate the total variance using the formula
        # Total Variance = ((n-1) * Average Variance + n * Variance of Means) / (N-1)
        total_variance = pd.Series(0, index=avg_variance.index, dtype='float64')
        if N > 1:
            # Handle edge case where n=1 (only one data point per iteration)
            if n > 1:
                for col in avg_variance.index:
                    if col in means_variance.index:
                        total_variance[col] = ((n-1) * avg_variance[col] + n * means_variance[col]) / (N-1)
            else:
                # If n=1, we only have between-iteration variance
                total_variance = means_variance
        else:
            total_variance = avg_variance
        
        # Calculate the overall standard deviation
        overall_std = np.sqrt(total_variance)
        
        # Create the aggregated DataFrame
        aggregated_df = data_rows.copy()
        
        # Make sure we preserve column types from the template
        col_types = template_df.dtypes
        
        # Add separator row
        separator_row = pd.Series("------------", index=aggregated_df.columns)
        aggregated_df = pd.concat([aggregated_df, separator_row.to_frame().T], ignore_index=True)
        
        # Add overall mean row
        mean_row = pd.Series(index=aggregated_df.columns)
        for col in aggregated_df.columns:
            if col in overall_mean.index:
                mean_row[col] = overall_mean[col]
            else:
                mean_row[col] = "N/A"  # For non-numeric columns
        aggregated_df = pd.concat([aggregated_df, mean_row.to_frame().T], ignore_index=True)
        
        # Add overall std dev row
        std_row = pd.Series(index=aggregated_df.columns)
        for col in aggregated_df.columns:
            if col in overall_std.index:
                std_row[col] = overall_std[col]
            else:
                std_row[col] = 0  # For non-numeric columns
        aggregated_df = pd.concat([aggregated_df, std_row.to_frame().T], ignore_index=True)
        
        # Save the aggregated results
        output_path = os.path.join(output_dir, filename)
        aggregated_df.to_csv(output_path, index=False, sep=';')
        print(f"Created aggregated file: {output_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Metrics processing tool')
    
    # Create a mutually exclusive group for the mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--merge', action='store_true', help='Merge mode (default)')
    mode_group.add_argument('--aggregate', action='store_true', help='Aggregate mode')
    
    # Arguments for merge mode
    parser.add_argument('--input', type=str, help='Input CSV file (for merge mode)')
    parser.add_argument('--output', type=str, help='Output CSV file (for merge mode)')
    
    # Arguments for aggregate mode
    parser.add_argument('--session', type=str, help='Session ID for aggregation')
    parser.add_argument('--iterations', type=int, help='Number of iterations to aggregate')
    parser.add_argument('--data-dir', type=str, default='./bench-data', 
                       help='Base directory for iteration data (default: ./bench-data)')
    parser.add_argument('--pattern', type=str, default='udp_rasp_conv_stats_*.csv',
                       help='File pattern to aggregate (default: udp_rasp_conv_stats_*.csv)')
    parser.add_argument('--output-dir', type=str, default='./bench-data-agg',
                       help='Output directory for aggregated results (default: ./bench-data-agg)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check if no mode was specified - default to merge mode
    if not args.merge and not args.aggregate:
        args.merge = True
    
    if args.merge:
        # Original merge functionality
        # Check if the correct number of arguments are provided
        if args.input is None or args.output is None:
            if len(sys.argv) != 3:
                print("Usage in merge mode: python metrics_merge.py <input_csv> <output_csv>")
                print("   or: python metrics_merge.py --merge --input <input_csv> --output <output_csv>")
                sys.exit(1)
            
            # Get the input and output file paths from the command-line arguments
            input_csv = sys.argv[1] if args.input is None else args.input
            output_csv = sys.argv[2] if args.output is None else args.output
        else:
            input_csv = args.input
            output_csv = args.output
        
        # Calculate modes from the input CSV
        modes = calculate_modes(input_csv)
        
        # Append the modes to the output CSV
        append_modes_to_csv(modes, output_csv)
        
        print(f"Successfully merged data from {input_csv} into {output_csv}")
        
    elif args.aggregate:
        # New aggregation functionality
        if args.session is None:
            print("Error: Session ID (--session) is required for aggregate mode")
            sys.exit(1)
        
        if args.iterations is None:
            print("Error: Number of iterations (--iterations) is required for aggregate mode")
            sys.exit(1)
        
        # Construct the list of iteration directories
        iteration_dirs = []
        for i in range(1, args.iterations + 1):
            dir_name = f"{args.data_dir}-{args.session}-{i}"
            if os.path.exists(dir_name):
                iteration_dirs.append(dir_name)
            else:
                print(f"Warning: Directory {dir_name} not found")
        
        if not iteration_dirs:
            print("Error: No valid iteration directories found")
            sys.exit(1)
        
        # Create timestamp-based output directory if not specified
        output_dir = args.output_dir
        if output_dir == './bench-data-agg':
            output_dir = f"{args.output_dir}-{args.session}"
        
        print(f"Aggregating metrics from {len(iteration_dirs)} iterations...")
        print(f"Session ID: {args.session}")
        print(f"Output directory: {output_dir}")
        
        # Perform the aggregation
        aggregate_metrics(iteration_dirs, args.pattern, output_dir)
        
        print(f"Aggregation completed. Results saved to {output_dir}")