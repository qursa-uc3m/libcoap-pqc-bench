#!/usr/bin/env python3
"""
CodeCarbon-based Energy Monitor for Local Benchmarking

This script provides energy monitoring using CodeCarbon library for local
benchmarking without requiring physical USB power meters. It's designed
to be a drop-in replacement for the FNIRSI-based energy_monitor.py.

Usage:
  python3 energy_monitor_codecarbon.py [options]

Options:
  --output FILE      Output file name (default: energy_data.csv)
  --duration SECONDS Duration to collect data in seconds (default: 0 = infinite)
  --identify         Check system compatibility and exit
  --verbose          Enable verbose output
  --start-pipe PATH  Named pipe for start synchronization
  --stop-pipe PATH   Named pipe for stop synchronization
  --merge ENERGY_FILE  Merge energy data into benchmark CSV
  --benchmark BENCH_FILE  Benchmark CSV file to merge energy data into
"""

import sys
import os
import time
import argparse
import csv
import signal
import logging
import numpy as np
from datetime import datetime
from typing import Optional

# Suppress CodeCarbon's verbose logging
logging.getLogger("codecarbon").setLevel(logging.ERROR)

try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not installed. Install with: pip install codecarbon", file=sys.stderr)

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: pandas not installed. Merge functionality will not be available.", file=sys.stderr)


class MeasurementState:
    """Tracks measurement state and accumulated values"""
    def __init__(self):
        self.energy_kwh = 0.0       # Energy in kWh (CodeCarbon native unit)
        self.start_time = None
        self.end_time = None
        self.samples_count = 0
        self.power_values = []       # Instantaneous power estimates
        self.emissions_kg = 0.0      # CO2 emissions in kg
        self.duration = 0.0


# Global state
quit_flag = False
prepare_to_quit = False


def signal_handler(sig, frame):
    """Signal handler for graceful termination"""
    global quit_flag, prepare_to_quit
    
    if sig == signal.SIGUSR1:
        print("\nReceived USR1 signal. Preparing to terminate...", file=sys.stderr)
        prepare_to_quit = True
    else:
        print("\nReceived termination signal. Shutting down...", file=sys.stderr)
        quit_flag = True


def check_system_compatibility():
    """Check if the system supports power measurement via CodeCarbon"""
    info = {
        "codecarbon_available": CODECARBON_AVAILABLE,
        "rapl_available": False,
        "nvidia_gpu": False,
        "cpu_info": "Unknown"
    }
    
    if not CODECARBON_AVAILABLE:
        return info
    
    # Check for Intel RAPL
    rapl_paths = [
        "/sys/class/powercap/intel-rapl",
        "/sys/class/powercap/intel-rapl:0"
    ]
    for path in rapl_paths:
        if os.path.exists(path):
            info["rapl_available"] = True
            break
    
    # Check for NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            info["nvidia_gpu"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Get CPU info
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_info"] = line.split(":")[1].strip()
                    break
    except:
        pass
    
    return info


def print_system_info():
    """Print system compatibility information"""
    info = check_system_compatibility()
    
    print("=" * 50)
    print("CodeCarbon Energy Monitor - System Check")
    print("=" * 50)
    print(f"CodeCarbon installed: {info['codecarbon_available']}")
    print(f"Intel RAPL available: {info['rapl_available']}")
    print(f"NVIDIA GPU detected:  {info['nvidia_gpu']}")
    print(f"CPU: {info['cpu_info']}")
    print("=" * 50)
    
    if not info['codecarbon_available']:
        print("\nERROR: CodeCarbon is not installed.")
        print("Install it with: pip install codecarbon")
        return False
    
    if not info['rapl_available']:
        print("\nWARNING: Intel RAPL not available.")
        print("Power measurements will be TDP-based estimates.")
        print("For more accurate measurements, ensure:")
        print("  - Running on Intel/AMD processor with RAPL support")
        print("  - Kernel has powercap module loaded")
        print("  - User has read access to /sys/class/powercap/")
    
    print("\nSystem is ready for energy monitoring.")
    return True


def create_tracker(output_dir: str = None, measure_power_secs: float = 0.5) -> Optional[object]:
    """Create and configure a CodeCarbon emissions tracker"""
    if not CODECARBON_AVAILABLE:
        return None
    
    try:
        # Use offline tracker to avoid network calls
        tracker = OfflineEmissionsTracker(
            country_iso_code="ESP",  # Default to Spain, adjust as needed
            output_dir=output_dir or ".",
            output_file="codecarbon_emissions.csv",
            measure_power_secs=measure_power_secs,
            log_level="error",
            save_to_file=False,  # We manage our own output
            save_to_api=False,
            save_to_logger=False,
        )
        return tracker
    except Exception as e:
        print(f"Error creating tracker: {e}", file=sys.stderr)
        return None


def run_measurement(duration: float, output_file: str, 
                   start_pipe: str = None, stop_pipe: str = None,
                   verbose: bool = False) -> MeasurementState:
    """
    Run energy measurement for the specified duration
    
    Args:
        duration: Measurement duration in seconds (0 = wait for signal)
        output_file: Base name for output files
        start_pipe: Named pipe for signaling readiness
        stop_pipe: Named pipe for signaling completion
        verbose: Enable verbose output
        
    Returns:
        MeasurementState with collected data
    """
    global quit_flag, prepare_to_quit
    
    state = MeasurementState()
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tracker
    tracker = create_tracker(output_dir, measure_power_secs=0.5)
    if tracker is None:
        print("ERROR: Could not create CodeCarbon tracker", file=sys.stderr)
        return state
    
    # Prepare raw data output file
    raw_output_file = output_file + "_raw.csv"
    
    try:
        with open(raw_output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                "sample", "timestamp", "elapsed", "energy_kwh_cumulative", "emissions_kg_cumulative"
            ])
            
            # Start tracking
            state.start_time = time.time()
            tracker.start()
            
            if verbose:
                print(f"Started energy tracking at {datetime.now().isoformat()}", file=sys.stderr)
            
            # Signal readiness via start pipe
            if start_pipe and os.path.exists(start_pipe):
                try:
                    with open(start_pipe, 'w') as f:
                        f.write("READY\n")
                    if verbose:
                        print(f"Sent READY signal via {start_pipe}", file=sys.stderr)
                except Exception as e:
                    print(f"Error signaling readiness: {e}", file=sys.stderr)
            
            # Calculate end time if duration specified
            end_time = state.start_time + duration if duration > 0 else None
            sample_interval = 0.5  # Sample every 500ms
            last_sample_time = state.start_time
            
            # Main measurement loop
            while not quit_flag:
                current_time = time.time()
                
                # Check if duration reached
                if end_time and current_time >= end_time:
                    if verbose:
                        print("Specified duration reached", file=sys.stderr)
                    break
                
                # Check if preparing to quit
                if prepare_to_quit:
                    if verbose:
                        print("Preparing to quit...", file=sys.stderr)
                    
                    # Signal completion via stop pipe
                    if stop_pipe and os.path.exists(stop_pipe):
                        try:
                            with open(stop_pipe, 'w') as f:
                                f.write("DONE\n")
                            if verbose:
                                print(f"Sent DONE signal via {stop_pipe}", file=sys.stderr)
                        except Exception as e:
                            print(f"Error sending DONE signal: {e}", file=sys.stderr)
                    break
                
                # Record sample at intervals
                if current_time - last_sample_time >= sample_interval:
                    elapsed = current_time - state.start_time
                    time_delta = current_time - last_sample_time
                    
                    # Get current energy from internal tracker state
                    try:
                        # Flush to update internal state
                        tracker.flush()
                        
                        # Get cumulative energy from internal state
                        prev_energy = state.energy_kwh
                        if hasattr(tracker, '_total_energy'):
                            state.energy_kwh = tracker._total_energy.kWh
                        if hasattr(tracker, '_total_emissions'):
                            state.emissions_kg = tracker._total_emissions.kgCO2
                        
                        # Calculate instantaneous power from energy delta
                        energy_delta_kwh = state.energy_kwh - prev_energy
                        if time_delta > 0 and energy_delta_kwh > 0:
                            instantaneous_power = (energy_delta_kwh * 1000) / (time_delta / 3600)  # W
                            state.power_values.append(instantaneous_power)
                    except Exception as e:
                        if verbose:
                            print(f"Error getting energy data: {e}", file=sys.stderr)
                    
                    state.samples_count += 1
                    
                    # Write sample to CSV
                    csv_writer.writerow([
                        state.samples_count,
                        f"{current_time:.6f}",
                        f"{elapsed:.6f}",
                        f"{state.energy_kwh:.9f}",
                        f"{state.emissions_kg:.9f}"
                    ])
                    
                    last_sample_time = current_time
                    csvfile.flush()
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.1)
            
            # Stop tracking
            tracker.stop()
            state.end_time = time.time()
            state.duration = state.end_time - state.start_time
            
            # Get final energy values from internal state
            if hasattr(tracker, '_total_energy'):
                state.energy_kwh = tracker._total_energy.kWh
            if hasattr(tracker, '_total_emissions'):
                state.emissions_kg = tracker._total_emissions.kgCO2
            
            if verbose:
                print(f"Stopped energy tracking at {datetime.now().isoformat()}", file=sys.stderr)
                print(f"Duration: {state.duration:.2f}s", file=sys.stderr)
                print(f"Energy: {state.energy_kwh * 1000:.6f} mWh", file=sys.stderr)
                
    except Exception as e:
        print(f"Error during measurement: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        # Attempt to stop tracker
        try:
            tracker.stop()
        except:
            pass
    
    return state


def save_energy_summary(state: MeasurementState, output_file: str) -> str:
    """
    Save energy data summary compatible with the benchmark system
    
    Creates a CSV file in the same format as the FNIRSI energy monitor
    for compatibility with bench-data-manager.py
    """
    if state.start_time is None or state.duration == 0:
        print("No measurement data to save", file=sys.stderr)
        return None
    
    # Convert units
    energy_wh = state.energy_kwh * 1000  # kWh to Wh
    
    # Calculate average power
    if state.duration > 0:
        avg_power = energy_wh / (state.duration / 3600)  # Wh / hours = W
    else:
        avg_power = 0
    
    # Get max power from samples (or use average if no samples)
    max_power = max(state.power_values) if state.power_values else avg_power
    power_std = np.std(state.power_values) if len(state.power_values) > 1 else 0
    
    # Energy uncertainty (rough estimate based on power variation)
    energy_std = power_std * (state.duration / 3600) if power_std > 0 else 0
    
    summary_file = output_file + ".csv"
    
    try:
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            
            # Header row
            writer.writerow([
                "timestamp", "voltage", "current", "power", "temperature",
                "Power (W)", "Max Power (W)", "Energy (Wh)"
            ])
            
            # Data row
            writer.writerow([
                time.time(),
                "0.000000",  # No voltage measurement with CodeCarbon
                "0.000000",  # No current measurement with CodeCarbon
                f"{avg_power:.6f}",
                "0.0",       # No temperature measurement
                f"{avg_power:.6f}",
                f"{max_power:.6f}",
                f"{energy_wh:.6f}"
            ])
            
            # Separator row
            writer.writerow(["-----------"] * 8)
            
            # Mean values row
            writer.writerow([
                time.time(),
                "0.000000",
                "0.000000",
                f"{avg_power:.6f}",
                "0.0",
                f"{avg_power:.6f}",
                f"{max_power:.6f}",
                f"{energy_wh:.6f}"
            ])
            
            # Standard deviation row
            writer.writerow([
                "0",
                "0",
                "0",
                f"{power_std:.6f}",
                "0",
                f"{power_std:.6f}",
                "0",
                f"{energy_std:.6f}"
            ])
        
        print(f"Energy summary saved to {summary_file}", file=sys.stderr)
        
        # Create completion marker
        with open(f"{summary_file}.done", 'w') as f:
            f.write(f"Completed at {datetime.now().isoformat()}\n")
        
        return summary_file
        
    except Exception as e:
        print(f"Error saving energy summary: {e}", file=sys.stderr)
        return None


def print_summary(state: MeasurementState):
    """Print a summary of the collected measurements"""
    if state.start_time is None:
        print("\nNo data was collected.", file=sys.stderr)
        return
    
    energy_wh = state.energy_kwh * 1000  # Convert to Wh
    avg_power = energy_wh / (state.duration / 3600) if state.duration > 0 else 0
    max_power = max(state.power_values) if state.power_values else avg_power
    power_std = np.std(state.power_values) if len(state.power_values) > 1 else 0
    
    print("\n---------- Measurement Summary ----------", file=sys.stderr)
    print(f"Duration: {state.duration:.2f} seconds", file=sys.stderr)
    print(f"Samples collected: {state.samples_count}", file=sys.stderr)
    print(f"Average power: {avg_power:.6f} W", file=sys.stderr)
    print(f"Maximum power: {max_power:.6f} W", file=sys.stderr)
    print(f"Power std deviation: {power_std:.6f} W", file=sys.stderr)
    print(f"Total energy: {energy_wh:.6f} Wh ({state.energy_kwh * 1e6:.6f} mWh)", file=sys.stderr)
    print(f"CO2 emissions: {state.emissions_kg * 1000:.3f} g", file=sys.stderr)
    print("-------------------------------------------", file=sys.stderr)
    print("Note: CodeCarbon estimates may differ from physical measurements", file=sys.stderr)


def merge_energy_data(energy_file: str, benchmark_file: str, verbose: bool = False) -> bool:
    """
    Merge energy data from energy CSV into benchmark CSV file
    
    This is the same merge function as in energy_monitor.py for compatibility.
    """
    if pd is None:
        print("Error: pandas is required for merge functionality", file=sys.stderr)
        return False
    
    try:
        if not os.path.exists(energy_file):
            print(f"Error: Energy file {energy_file} not found", file=sys.stderr)
            return False
        
        if not os.path.exists(benchmark_file):
            print(f"Error: Benchmark file {benchmark_file} not found", file=sys.stderr)
            return False
        
        if verbose:
            print(f"Merging energy data from {energy_file} into {benchmark_file}", file=sys.stderr)
        
        # Detect delimiters
        with open(energy_file, 'r') as f:
            energy_delimiter = ';' if ';' in f.readline() else ','
        with open(benchmark_file, 'r') as f:
            benchmark_delimiter = ';' if ';' in f.readline() else ','
        
        # Read files
        energy_df = pd.read_csv(energy_file, delimiter=energy_delimiter)
        benchmark_df = pd.read_csv(benchmark_file, delimiter=benchmark_delimiter)
        
        # Find mean and std rows in energy file
        mean_row_idx = len(energy_df) - 2
        std_row_idx = len(energy_df) - 1
        
        # Extract energy values
        mean_power = float(energy_df.iloc[mean_row_idx].get("Power (W)", 0))
        mean_max_power = float(energy_df.iloc[mean_row_idx].get("Max Power (W)", 0))
        mean_energy = float(energy_df.iloc[mean_row_idx].get("Energy (Wh)", 0))
        std_power = float(energy_df.iloc[std_row_idx].get("Power (W)", 0))
        std_energy = float(energy_df.iloc[std_row_idx].get("Energy (Wh)", 0))
        
        if verbose:
            print(f"Power: {mean_power} W, Max: {mean_max_power} W, Energy: {mean_energy} Wh", file=sys.stderr)
        
        # Add energy columns to benchmark
        benchmark_df["Power (W)"] = None
        benchmark_df["Max Power (W)"] = None
        benchmark_df["Energy (Wh)"] = None
        
        # Find separator in benchmark file
        separator_indices = []
        for i, row in benchmark_df.iterrows():
            for col in benchmark_df.columns:
                if isinstance(row[col], str) and '---' in str(row[col]):
                    separator_indices.append(i)
                    break
        
        # Fill energy values
        for i, row in benchmark_df.iterrows():
            is_separator = i in separator_indices
            if is_separator:
                benchmark_df.at[i, "Power (W)"] = '------------'
                benchmark_df.at[i, "Max Power (W)"] = '------------'
                benchmark_df.at[i, "Energy (Wh)"] = '------------'
            elif i == len(benchmark_df) - 1:
                # Std row
                benchmark_df.at[i, "Power (W)"] = f"{std_power:.6f}"
                benchmark_df.at[i, "Max Power (W)"] = "0.000000"
                benchmark_df.at[i, "Energy (Wh)"] = f"{std_energy:.6f}"
            else:
                benchmark_df.at[i, "Power (W)"] = f"{mean_power:.6f}"
                benchmark_df.at[i, "Max Power (W)"] = f"{mean_max_power:.6f}"
                benchmark_df.at[i, "Energy (Wh)"] = f"{mean_energy:.6f}"
        
        # Write back
        benchmark_df.to_csv(benchmark_file, index=False, sep=benchmark_delimiter)
        
        if verbose:
            print(f"Successfully merged energy data into {benchmark_file}", file=sys.stderr)
        
        return True
        
    except Exception as e:
        print(f"Error merging energy data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main program function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CodeCarbon-based Energy Monitor for Local Benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=str, default="energy_data",
                       help="Output file name (without extension)")
    parser.add_argument("--duration", type=float, default=0,
                       help="Duration to collect data in seconds (0 = infinite)")
    parser.add_argument("--identify", action="store_true",
                       help="Check system compatibility and exit")
    parser.add_argument("--list-devices", action="store_true",
                       help="Alias for --identify (for compatibility)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--force-reset", action="store_true",
                       help="Ignored (for compatibility)")
    parser.add_argument("--start-pipe", type=str,
                       help="Named pipe for start synchronization")
    parser.add_argument("--stop-pipe", type=str,
                       help="Named pipe for stop synchronization")
    parser.add_argument("--merge", metavar='ENERGY_FILE',
                       help="Merge energy data from ENERGY_FILE into benchmark CSV")
    parser.add_argument("--benchmark", metavar='BENCH_FILE',
                       help="Benchmark CSV file to merge energy data into")
    
    args = parser.parse_args()
    
    # Handle merge mode
    if args.merge:
        if not args.benchmark:
            print("Error: --benchmark is required when using --merge", file=sys.stderr)
            return 1
        result = merge_energy_data(args.merge, args.benchmark, args.verbose)
        return 0 if result else 1
    
    # Handle identify/list-devices
    if args.identify or args.list_devices:
        success = print_system_info()
        return 0 if success else 1
    
    # Check if CodeCarbon is available
    if not CODECARBON_AVAILABLE:
        print("Error: CodeCarbon is not installed", file=sys.stderr)
        print("Install with: pip install codecarbon", file=sys.stderr)
        return 1
    
    # Run measurement
    print(f"Starting CodeCarbon energy monitoring...", file=sys.stderr)
    if args.duration > 0:
        print(f"Duration: {args.duration} seconds", file=sys.stderr)
    else:
        print("Duration: Until terminated (Ctrl+C or signal)", file=sys.stderr)
    
    state = run_measurement(
        duration=args.duration,
        output_file=args.output,
        start_pipe=args.start_pipe,
        stop_pipe=args.stop_pipe,
        verbose=args.verbose
    )
    
    # Print summary
    print_summary(state)
    
    # Save summary file
    save_energy_summary(state, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
