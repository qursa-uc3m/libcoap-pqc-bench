#!/usr/bin/env python3
"""
Unified Benchmark Data Manager

This module provides a comprehensive solution for managing benchmark data:
- Processes timing, CPU cycles, and network statistics
- Standardizes data formats
- Handles merging of different data types
- Supports data aggregation across multiple iterations
- Manages consistent file naming and organization

Usage:
  python benchmark_data_manager.py process [options]
  python benchmark_data_manager.py merge [options]
  python benchmark_data_manager.py aggregate [options]
"""

import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

class BenchmarkConfig:
    """Represents the configuration of a benchmark test"""
    
    def __init__(self, 
                 security_mode: str = None,
                 algorithm: str = None, 
                 cert_type: str = None,
                 clients: int = None,
                 observer_time: int = None,
                 parallelization: str = None,
                 scenario: str = None,
                 rasp: bool = False,
                 client_auth: bool = False):
        self.security_mode = security_mode  # pki, psk, nosec
        self.algorithm = algorithm          # KYBER_LEVEL5, etc.
        self.cert_type = cert_type          # DILITHIUM_LEVEL3, RSA_2048, etc.
        self.clients = clients              # Number of clients
        self.observer_time = observer_time  # Observer time in seconds
        self.parallelization = parallelization  # background, parallel
        self.scenario = scenario            # A, B, C
        self.rasp = rasp                    # True if using Raspberry Pi
        self.client_auth = client_auth      # True if client authentication enabled
    
    @classmethod
    def from_filename(cls, filename: str) -> 'BenchmarkConfig':
        """Parse a benchmark configuration from a filename"""
        config = cls()
        
        # Extract basic properties
        config.rasp = "_rasp_" in filename
        
        # Extract security mode
        if "_pki" in filename:
            config.security_mode = "pki"
            config.client_auth = "_client-auth" in filename
        elif "_psk" in filename:
            config.security_mode = "psk"
        elif "_nosec" in filename:
            config.security_mode = "nosec"
            
        # Extract scenario
        scenario_match = re.search(r'_scenario([A-C])', filename)
        if scenario_match:
            config.scenario = scenario_match.group(1)
        
        # Extract clients count
        clients_match = re.search(r'_n(\d+)', filename)
        if clients_match:
            config.clients = int(clients_match.group(1))
            
        # Extract observer time
        observer_match = re.search(r'_s(\d+)', filename)
        if observer_match:
            config.observer_time = int(observer_match.group(1))
            
        # Extract parallelization
        if "_background" in filename:
            config.parallelization = "background"
        elif "_parallel" in filename:
            config.parallelization = "parallel"
            
        # Extract algorithm (only for pki/psk)
        if config.security_mode in ("pki", "psk"):
            # This is a simplification - actual parsing would need to be more robust
            # to handle different algorithm naming patterns
            algorithm_match = re.search(r'stats_([A-Z0-9_]+)_', filename)
            if algorithm_match:
                config.algorithm = algorithm_match.group(1)
        
        # Extract certificate type (only for pki)
        if config.security_mode == "pki":
            # Again, this is simplified and would need to be more robust
            cert_match = re.search(r'_([A-Z0-9_]+)_n', filename)
            if cert_match:
                config.cert_type = cert_match.group(1)
                
        return config
    
    def to_filename(self) -> str:
        """Generate a standardized filename from this configuration"""
        parts = []
        
        # Base prefix
        prefix = "udp_rasp" if self.rasp else "udp"
        parts.append(prefix)
        
        # Add algorithm and cert type for pki/psk
        if self.security_mode in ("pki", "psk") and self.algorithm:
            parts.append(self.algorithm)
            
            if self.security_mode == "pki" and self.cert_type:
                parts.append(self.cert_type)
        
        # Add clients
        if self.clients:
            parts.append(f"n{self.clients}")
        
        # Add observer time and parallelization if applicable
        if self.observer_time:
            parts.append(f"s{self.observer_time}")
            if self.parallelization:
                parts.append(self.parallelization)
        
        # Add security mode and client auth
        parts.append(self.security_mode)
        if self.security_mode == "pki" and self.client_auth:
            parts.append("client-auth")
            
        # Add scenario
        if self.scenario:
            parts.append(f"scenario{self.scenario}")
            
        return "_".join(parts)
    
    def __str__(self) -> str:
        """Return a human-readable representation of the configuration"""
        parts = []
        
        if self.security_mode:
            parts.append(f"Security: {self.security_mode}")
        
        if self.algorithm:
            parts.append(f"Algorithm: {self.algorithm}")
            
        if self.cert_type:
            parts.append(f"Certificate: {self.cert_type}")
            
        if self.clients:
            parts.append(f"Clients: {self.clients}")
            
        if self.observer_time:
            parts.append(f"Observer: {self.observer_time}s")
            
        if self.parallelization:
            parts.append(f"Parallelization: {self.parallelization}")
            
        if self.scenario:
            parts.append(f"Scenario: {self.scenario}")
            
        if self.rasp:
            parts.append("Raspberry Pi")
            
        if self.client_auth:
            parts.append("Client Auth")
            
        return ", ".join(parts)


class BenchmarkData:
    """Represents the data collected from a benchmark run"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.raw_data = {}  # Raw data from various sources
        self.metrics = {}   # Processed metrics
        
        # Standard metrics we'll always calculate
        self.metrics['duration'] = None
        self.metrics['cpu_cycles'] = None
        
        # Network metrics
        self.metrics['frames_sent'] = None
        self.metrics['bytes_sent'] = None
        self.metrics['frames_received'] = None
        self.metrics['bytes_received'] = None
        self.metrics['total_frames'] = None
        self.metrics['total_bytes'] = None
        
        # Energy metrics (if available)
        self.metrics['power'] = None
        self.metrics['max_power'] = None
        self.metrics['energy'] = None
        
        # Statistical calculations
        self.mean_values = {}
        self.std_values = {}
    
    def load_time_data(self, time_file: str) -> bool:
        """Load timing data from time_output.txt file"""
        try:
            with open(time_file, 'r') as file:
                times = [float(line.strip()) for line in file]
                self.raw_data['times'] = times
                self.metrics['duration'] = np.mean(times)
                return True
        except Exception as e:
            print(f"Error reading time file {time_file}: {e}")
            return False
    
    def load_cycles_data(self, cycles_file: str) -> bool:
        """Load CPU cycles data from cycles_output.txt file"""
        try:
            with open(cycles_file, 'r') as file:
                cycles = int(file.read().strip())
                self.raw_data['cycles'] = cycles
                self.metrics['cpu_cycles'] = cycles
                return True
        except Exception as e:
            print(f"Error reading cycles file {cycles_file}: {e}")
            return False
            
    def load_network_stats(self, stats_file: str) -> bool:
        """Load network statistics from Wireshark/tshark output"""
        try:
            with open(stats_file, 'r') as file:
                lines = file.readlines()
                
                # Parse the relevant lines to extract network metrics
                data = []
                for line in lines:
                    # Skip irrelevant lines
                    if not line.strip() or "<->" not in line:
                        continue
                    
                    try:
                        # Split the line into parts
                        parts = line.strip().split()
                        
                        # Check if there are enough parts to parse
                        if len(parts) < 12:
                            print(f"Warning: Not enough parts in line: {line.strip()}")
                            continue
                        
                        # Function to convert value with unit to bytes
                        def convert_to_bytes(value_str, unit):
                            # Handle thousand separators (e.g., "2.649 bytes" → 2649)
                            if unit == "bytes" and '.' in value_str:
                                value_str = value_str.replace('.', '')  # Remove thousand separator
                            
                            # Replace commas with dots for decimal parsing (e.g., "0,0143" → 0.0143)
                            value_str = value_str.replace(',', '.')
                            
                            value = float(value_str)
                            unit = unit.strip().lower()
                            if unit == 'kb':
                                return value * 1000
                            elif unit == 'kib':
                                return value * 1024
                            elif unit == 'mb':
                                return value * 1e6
                            elif unit == 'mib':
                                return value * 1024**2
                            elif unit == 'gb':
                                return value * 1e9
                            elif unit == 'gib':
                                return value * 1024**3
                            elif unit in ('bytes', 'b'):
                                return value
                            else:
                                print(f"Warning: Unknown unit '{unit}' in line: {line.strip()}")
                                return value  # Assume bytes if unknown
                        
                        # Parse out direction (A -> B)
                        frames_out = float(parts[3].replace(',', '.'))
                        bytes_out = convert_to_bytes(parts[4], parts[5])
                        
                        # Parse in direction (B -> A)
                        frames_in = float(parts[6].replace(',', '.'))
                        bytes_in = convert_to_bytes(parts[7], parts[8])
                        
                        # Parse total
                        total_frames = float(parts[9].replace(',', '.'))
                        total_bytes = convert_to_bytes(parts[10], parts[11])
                        
                        data.append([frames_out, bytes_out, frames_in, bytes_in, total_frames, total_bytes])
                        
                    except (IndexError, ValueError, KeyError) as e:
                        print(f"Warning: Could not parse line: {line.strip()}: {e}")
                        continue
        
                if not data:
                    print(f"Warning: No valid network statistics found in {stats_file}")
                    return False
                    
                # Convert to numpy array for easier calculations
                data = np.array(data)
                
                # Store the extracted metrics
                self.metrics['frames_sent'] = np.mean(data[:, 0])
                self.metrics['bytes_sent'] = np.mean(data[:, 1])
                self.metrics['frames_received'] = np.mean(data[:, 2])
                self.metrics['bytes_received'] = np.mean(data[:, 3])
                self.metrics['total_frames'] = np.mean(data[:, 4])
                self.metrics['total_bytes'] = np.mean(data[:, 5])
                
                # Store raw data for potential further analysis
                self.raw_data['network_stats'] = data
                
                return True
        except Exception as e:
            print(f"Error processing network stats file {stats_file}: {e}")
            return False
    
    def load_energy_data(self, energy_file: str) -> bool:
        """Load energy measurements if available"""
        try:
            # Check if the file exists
            if not os.path.exists(energy_file):
                print(f"Energy file {energy_file} not found")
                return False
            
            # Determine the delimiter (semicolon or comma)
            with open(energy_file, 'r') as f:
                first_line = f.readline().strip()
                delimiter = ';' if ';' in first_line else ','
            
            # Read the energy data
            df = pd.read_csv(energy_file, delimiter=delimiter)
            
            # Find the relevant rows (mean and std deviation)
            # This assumes the second-to-last row contains means and the last row contains std deviations
            try:
                # Try to identify the mean row (second to last)
                mean_row_idx = len(df) - 2
                
                # Extract the energy metrics
                self.metrics['power'] = float(df.iloc[mean_row_idx].get("Power (W)") or df.iloc[mean_row_idx].get("power"))
                self.metrics['max_power'] = float(df.iloc[mean_row_idx].get("Max Power (W)") or df.iloc[mean_row_idx].get("max_power"))
                self.metrics['energy'] = float(df.iloc[mean_row_idx].get("Energy (Wh)") or df.iloc[mean_row_idx].get("energy"))
                
                # Store raw data
                self.raw_data['energy'] = {
                    'power': self.metrics['power'],
                    'max_power': self.metrics['max_power'],
                    'energy': self.metrics['energy']
                }
                
                return True
            except Exception as e:
                print(f"Error parsing energy values from {energy_file}: {e}")
                return False
                
        except Exception as e:
            print(f"Error reading energy file {energy_file}: {e}")
            return False
    
    def calculate_statistics(self) -> None:
        """Calculate mean and standard deviation for all metrics"""
        # Calculate means (most are already means from the source data)
        self.mean_values = {k: v for k, v in self.metrics.items() if v is not None}
        
        # Calculate standard deviations from raw data where available
        self.std_values = {}
        
        if 'times' in self.raw_data:
            self.std_values['duration'] = np.std(self.raw_data['times'])
        
        if 'network_stats' in self.raw_data:
            data = self.raw_data['network_stats']
            self.std_values['frames_sent'] = np.std(data[:, 0])
            self.std_values['bytes_sent'] = np.std(data[:, 1])
            self.std_values['frames_received'] = np.std(data[:, 2])
            self.std_values['bytes_received'] = np.std(data[:, 3])
            self.std_values['total_frames'] = np.std(data[:, 4])
            self.std_values['total_bytes'] = np.std(data[:, 5])
    
    def save_csv(self, output_file: str) -> bool:
        """Save the benchmark data to a CSV file"""
        try:
            # Define column headers (all possible metrics)
            columns = [
                'duration', 'cpu_cycles', 
                'frames_sent', 'bytes_sent', 'frames_received', 'bytes_received', 
                'total_frames', 'total_bytes',
                'Power (W)', 'Max Power (W)', 'Energy (Wh)'
            ]
            
            # Prepare data rows
            rows = []
            
            # Add raw data if available
            if 'times' in self.raw_data:
                for t in self.raw_data['times']:
                    row = {'duration': t}
                    rows.append(row)
            
            # Add separator row
            rows.append({col: '------------' for col in columns})
            
            # Add mean values row
            mean_row = {}
            for k, v in self.mean_values.items():
                if k in columns:
                    mean_row[k] = v
                elif k == 'power':
                    mean_row['Power (W)'] = v
                elif k == 'max_power':
                    mean_row['Max Power (W)'] = v
                elif k == 'energy':
                    mean_row['Energy (Wh)'] = v
            rows.append(mean_row)
            
            # Add standard deviation row
            std_row = {}
            for k, v in self.std_values.items():
                if k in columns:
                    std_row[k] = v
                elif k == 'power':
                    std_row['Power (W)'] = v
                elif k == 'max_power':
                    std_row['Max Power (W)'] = v
                elif k == 'energy':
                    std_row['Energy (Wh)'] = v
            rows.append(std_row)
            
            # Write to CSV
            df = pd.DataFrame(rows)
            df.to_csv(output_file, sep=';', index=False)
            
            print(f"Data saved to {output_file}")
            return True
        
        except Exception as e:
            print(f"Error saving data to {output_file}: {e}")
            return False


class BenchmarkDataManager:
    """Manages the processing, merging, and aggregation of benchmark data"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), "bench-data")
        
    def process_benchmark(self, time_file: str = None, cycles_file: str = None,
                         stats_file: str = None, energy_file: str = None,
                         config: BenchmarkConfig = None, output_file: str = None) -> Optional[BenchmarkData]:
        """Process a single benchmark run from various input files"""
        benchmark = BenchmarkData(config)
        
        # Load data from available sources
        success = True
        if time_file and os.path.exists(time_file):
            success = success and benchmark.load_time_data(time_file)
        
        if cycles_file and os.path.exists(cycles_file):
            success = success and benchmark.load_cycles_data(cycles_file)
            
        if stats_file and os.path.exists(stats_file):
            success = success and benchmark.load_network_stats(stats_file)
            
        if energy_file and os.path.exists(energy_file):
            success = success and benchmark.load_energy_data(energy_file)
            
        if not success:
            print("Warning: Some data files could not be processed")
            
        # Calculate statistics
        benchmark.calculate_statistics()
        
        # Save to file if output_file is provided
        if output_file:
            benchmark.save_csv(output_file)
            
        return benchmark
    
    def process_from_directory(self, input_dir: str = None, 
                              output_dir: str = None) -> List[BenchmarkData]:
        """Process all benchmark data in a directory"""
        input_dir = input_dir or self.base_dir
        output_dir = output_dir or input_dir
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all the raw data files
        time_file = os.path.join(input_dir, "time_output.txt")
        cycles_file = os.path.join(input_dir, "cycles_output.txt")
        
        # Find Wireshark stats files
        stats_files = glob.glob(os.path.join(input_dir, "*.txt"))
        stats_files = [f for f in stats_files if "udp" in os.path.basename(f) and 
                      "conv_stats" in os.path.basename(f)]
        
        # Find energy data files
        energy_files = glob.glob(os.path.join(input_dir, "energy_*.csv"))
        
        results = []
        
        # Process each stats file
        for stats_file in stats_files:
            base_name = os.path.basename(stats_file)
            base_name = os.path.splitext(base_name)[0]
            
            # Parse configuration from filename
            config = BenchmarkConfig.from_filename(base_name)
            
            # Find matching energy file if available
            matching_energy = None
            for energy_file in energy_files:
                energy_base = os.path.basename(energy_file)
                if base_name in energy_base:
                    matching_energy = energy_file
                    break
            
            # Generate output filename
            output_file = os.path.join(output_dir, f"{base_name}.csv")
            
            # Process the benchmark
            benchmark = self.process_benchmark(
                time_file=time_file,
                cycles_file=cycles_file,
                stats_file=stats_file,
                energy_file=matching_energy,
                config=config,
                output_file=output_file
            )
            
            if benchmark:
                results.append(benchmark)
                
        return results
    
    def merge_energy_data(self, energy_file: str, benchmark_file: str) -> bool:
        """Merge energy data into a benchmark results file"""
        try:
            # Check if files exist
            if not os.path.exists(energy_file):
                print(f"Error: Energy file {energy_file} not found")
                return False
            
            if not os.path.exists(benchmark_file):
                print(f"Error: Benchmark file {benchmark_file} not found")
                return False
            
            # Read energy data
            energy_df = pd.read_csv(energy_file, delimiter=';')
            
            # Find the mean and std rows
            mean_row_idx = len(energy_df) - 2
            std_row_idx = len(energy_df) - 1
            
            # Extract energy values
            power = float(energy_df.iloc[mean_row_idx].get("Power (W)") or 
                         energy_df.iloc[mean_row_idx].get("power"))
            
            max_power = float(energy_df.iloc[mean_row_idx].get("Max Power (W)") or 
                             energy_df.iloc[mean_row_idx].get("max_power"))
            
            energy = float(energy_df.iloc[mean_row_idx].get("Energy (Wh)") or 
                          energy_df.iloc[mean_row_idx].get("energy"))
            
            power_std = float(energy_df.iloc[std_row_idx].get("Power (W)") or 
                             energy_df.iloc[std_row_idx].get("power") or 0)
            
            energy_std = float(energy_df.iloc[std_row_idx].get("Energy (Wh)") or 
                              energy_df.iloc[std_row_idx].get("energy") or 0)
            
            # Read benchmark file
            benchmark_df = pd.read_csv(benchmark_file, delimiter=';')
            
            # Add energy columns to benchmark dataframe
            benchmark_df["Power (W)"] = None
            benchmark_df["Max Power (W)"] = None
            benchmark_df["Energy (Wh)"] = None
            
            # Find separator row in benchmark file
            separator_idx = None
            for i, row in benchmark_df.iterrows():
                # Check if this is a separator row (contains dashes)
                if isinstance(row.iloc[0], str) and '---' in row.iloc[0]:
                    separator_idx = i
                    break
            
            # If separator row was found, use it to identify mean and std rows
            if separator_idx is not None:
                mean_idx = separator_idx + 1
                std_idx = separator_idx + 2
                
                # Set energy values
                # Separator row
                benchmark_df.at[separator_idx, "Power (W)"] = '------------'
                benchmark_df.at[separator_idx, "Max Power (W)"] = '------------'
                benchmark_df.at[separator_idx, "Energy (Wh)"] = '------------'
                
                # Mean row
                benchmark_df.at[mean_idx, "Power (W)"] = power
                benchmark_df.at[mean_idx, "Max Power (W)"] = max_power
                benchmark_df.at[mean_idx, "Energy (Wh)"] = energy
                
                # Std row
                benchmark_df.at[std_idx, "Power (W)"] = power_std
                benchmark_df.at[std_idx, "Max Power (W)"] = 0
                benchmark_df.at[std_idx, "Energy (Wh)"] = energy_std
                
                # Fill remaining rows with mean values
                #for i in range(len(benchmark_df)):
                #    if i != separator_idx and i != mean_idx and i != std_idx:
                #        benchmark_df.at[i, "Power (W)"] = power
                #        benchmark_df.at[i, "Max Power (W)"] = max_power
                #        benchmark_df.at[i, "Energy (Wh)"] = energy
            
            # Write the updated benchmark file back
            benchmark_df.to_csv(benchmark_file, index=False, sep=';')
            
            print(f"Successfully merged energy data into {benchmark_file}")
            return True
            
        except Exception as e:
            print(f"Error merging energy data: {e}")
            return False
    
    def aggregate_iterations(self, session_id: str, iterations: int, 
                            data_dir: str = None, output_dir: str = None) -> bool:
        """Aggregate metrics across multiple iterations of the same benchmark"""
        base_data_dir = data_dir or os.getcwd()
        output_dir = output_dir or os.path.join(base_data_dir, f"bench-data-agg-{session_id}")
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # For each iteration, get list of result files
        all_files = {}
        
        for i in range(1, iterations + 1):
            iteration_dir = os.path.join(base_data_dir, f"bench-data-{session_id}-{i}")
            
            if not os.path.exists(iteration_dir):
                print(f"Warning: Iteration directory {iteration_dir} not found")
                continue
                
            # Find all CSV result files
            csv_files = glob.glob(os.path.join(iteration_dir, "*.csv"))
            
            for file_path in csv_files:
                file_name = os.path.basename(file_path)
                
                # Group by file name
                if file_name not in all_files:
                    all_files[file_name] = []
                    
                all_files[file_name].append(file_path)
        
        # Aggregate each set of files
        for file_name, file_paths in all_files.items():
            if len(file_paths) < 2:
                # Just copy the file if only one iteration
                if len(file_paths) == 1:
                    output_path = os.path.join(output_dir, file_name)
                    import shutil
                    shutil.copy(file_paths[0], output_path)
                continue
                
            # Aggregate across iterations
            dfs = []
            for path in file_paths:
                try:
                    df = pd.read_csv(path, sep=';')
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
            
            if not dfs:
                continue
                
            # Perform aggregation
            # For each file, extract the mean and standard deviation rows
            means = []
            stds = []
            
            for df in dfs:
                # Find separator row
                separator_idx = None
                for i, row in df.iterrows():
                    # Check for separator row
                    if isinstance(row.iloc[0], str) and '---' in row.iloc[0]:
                        separator_idx = i
                        break
                        
                if separator_idx is None:
                    continue
                    
                # Extract mean and std rows
                try:
                    mean_row = df.iloc[separator_idx + 1].copy()
                    std_row = df.iloc[separator_idx + 2].copy()
                    
                    # Convert to numeric values
                    mean_row = pd.to_numeric(mean_row, errors='coerce')
                    std_row = pd.to_numeric(std_row, errors='coerce')
                    
                    means.append(mean_row)
                    stds.append(std_row)
                except Exception as e:
                    print(f"Error extracting statistics from {path}: {e}")
            
            if not means:
                continue
                
            # Create aggregated result
            # Use the first dataframe as a template
            template_df = dfs[0].copy()
            
            # Get number of data rows (rows before separator)
            n = separator_idx
            
            # Convert means and stds to DataFrames
            means_df = pd.DataFrame(means)
            stds_df = pd.DataFrame(stds)
            
            # Calculate aggregated statistics
            # Number of iterations
            k = len(means_df)
            
            # Total number of samples
            N = n * k
            
            # Calculate overall mean
            overall_mean = means_df.mean(numeric_only=True)
            
            # Calculate average variance
            avg_variance = stds_df.pow(2).mean(numeric_only=True)
            
            # Calculate variance of means
            means_variance = means_df.var(ddof=1, numeric_only=True) if k > 1 else pd.Series(0, index=means_df.columns, dtype='float64')
            
            # Calculate total variance
            total_variance = pd.Series(0, index=avg_variance.index, dtype='float64')
            
            if N > 1:
                if n > 1:
                    for col in avg_variance.index:
                        if col in means_variance.index:
                            total_variance[col] = ((n-1) * avg_variance[col] + n * means_variance[col]) / (N-1)
                else:
                    total_variance = means_variance
            else:
                total_variance = avg_variance
                
            # Calculate overall standard deviation
            overall_std = np.sqrt(total_variance)
            
            # Create the aggregated DataFrame
            agg_df = template_df.iloc[:n].copy()
            
            # Add separator row
            separator_row = pd.Series("------------", index=agg_df.columns)
            agg_df = pd.concat([agg_df, separator_row.to_frame().T], ignore_index=True)
            
            # Add overall mean row
            mean_row = pd.Series(index=agg_df.columns)
            for col in agg_df.columns:
                if col in overall_mean.index:
                    mean_row[col] = overall_mean[col]
                else:
                    mean_row[col] = "N/A"
            agg_df = pd.concat([agg_df, mean_row.to_frame().T], ignore_index=True)
            
            # Add overall std dev row
            std_row = pd.Series(index=agg_df.columns)
            for col in agg_df.columns:
                if col in overall_std.index:
                    std_row[col] = overall_std[col]
                else:
                    std_row[col] = 0
            agg_df = pd.concat([agg_df, std_row.to_frame().T], ignore_index=True)
            
            # Save aggregated results
            output_path = os.path.join(output_dir, file_name)
            agg_df.to_csv(output_path, index=False, sep=';')
            print(f"Created aggregated file: {output_path}")
            
        return True


def process_command(args):
    """Process the command-line arguments and execute the appropriate action"""
    manager = BenchmarkDataManager(args.data_dir)
    
    if args.command == 'process':
        # Process a single benchmark or all benchmarks in a directory
        if args.input_dir:
            print(f"Processing all benchmarks in {args.input_dir}")
            manager.process_from_directory(args.input_dir, args.output_dir)
        else:
            # Process a single benchmark from specified files
            if not (args.time_file or args.cycles_file or args.stats_file):
                print("Error: No input files specified")
                return 1
            
            config = BenchmarkConfig(
                security_mode=args.security_mode,
                algorithm=args.algorithm,
                cert_type=args.cert_type,
                clients=args.clients,
                observer_time=args.observer_time,
                parallelization=args.parallelization,
                scenario=args.scenario,
                rasp=args.rasp,
                client_auth=args.client_auth
            )
            
            output_file = args.output_file or os.path.join(
                args.output_dir or ".", 
                f"{config.to_filename()}.csv"
            )
            
            manager.process_benchmark(
                time_file=args.time_file,
                cycles_file=args.cycles_file,
                stats_file=args.stats_file,
                energy_file=args.energy_file,
                config=config,
                output_file=output_file
            )
    
    elif args.command == 'merge':
        # Merge energy data into benchmark results
        if not args.energy_file or not args.benchmark_file:
            print("Error: Both --energy-file and --benchmark-file are required for merge command")
            return 1
        
        print(f"Merging energy data from {args.energy_file} into {args.benchmark_file}")
        manager.merge_energy_data(args.energy_file, args.benchmark_file)
    
    elif args.command == 'aggregate':
        # Aggregate metrics across iterations
        if not args.session_id or not args.iterations:
            print("Error: Both --session-id and --iterations are required for aggregate command")
            return 1
        
        print(f"Aggregating metrics for session {args.session_id} across {args.iterations} iterations")
        manager.aggregate_iterations(
            session_id=args.session_id,
            iterations=args.iterations,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    
    return 0


def main():
    """Main function to parse command-line arguments and execute the program"""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Data Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True
    
    # Common arguments for all commands
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--data-dir", help="Base directory for benchmark data")
    base_parser.add_argument("--output-dir", help="Directory for output files")
    base_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process", 
        help="Process benchmark data from raw files",
        parents=[base_parser]
    )
    
    # Input options for process command
    input_group = process_parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-dir", help="Directory containing input files")
    input_group.add_argument("--time-file", help="File with timing data")
    
    process_parser.add_argument("--cycles-file", help="File with CPU cycles data")
    process_parser.add_argument("--stats-file", help="File with network statistics")
    process_parser.add_argument("--energy-file", help="File with energy measurements")
    process_parser.add_argument("--output-file", help="Output file for processed data")
    
    # Configuration options for process command
    process_parser.add_argument("--security-mode", choices=["pki", "psk", "nosec"], help="Security mode")
    process_parser.add_argument("--algorithm", help="Key exchange algorithm (e.g., KYBER_LEVEL5)")
    process_parser.add_argument("--cert-type", help="Certificate type (e.g., DILITHIUM_LEVEL3)")
    process_parser.add_argument("--clients", type=int, help="Number of clients")
    process_parser.add_argument("--observer-time", type=int, help="Observer time in seconds")
    process_parser.add_argument("--parallelization", choices=["background", "parallel"], help="Parallelization mode")
    process_parser.add_argument("--scenario", choices=["A", "B", "C"], help="Benchmark scenario")
    process_parser.add_argument("--rasp", action="store_true", help="Using Raspberry Pi")
    process_parser.add_argument("--client-auth", action="store_true", help="Client authentication enabled")
    
    # Merge command
    merge_parser = subparsers.add_parser(
        "merge", 
        help="Merge energy data into benchmark results",
        parents=[base_parser]
    )
    merge_parser.add_argument("--energy-file", required=True, help="Energy data file")
    merge_parser.add_argument("--benchmark-file", required=True, help="Benchmark results file")
    
    # Aggregate command
    aggregate_parser = subparsers.add_parser(
        "aggregate", 
        help="Aggregate metrics across iterations",
        parents=[base_parser]
    )
    aggregate_parser.add_argument("--session-id", required=True, help="Benchmark session ID")
    aggregate_parser.add_argument("--iterations", type=int, required=True, help="Number of iterations")
    
    args = parser.parse_args()
    
    return process_command(args)


if __name__ == "__main__":
    sys.exit(main())   