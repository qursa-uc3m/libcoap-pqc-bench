#!/usr/bin/env python3
"""Shared utilities and base classes for energy monitoring."""

import os
import csv
import time
import signal
import numpy as np
from datetime import datetime
from typing import List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None


class MeasurementState:
    """Tracks measurement state and accumulated values."""
    def __init__(self):
        self.energy = 0.0
        self.capacity = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration = 0.0
        self.temp_ema: Optional[float] = None
        self.max_power = 0.0
        self.max_current = 0.0
        self.max_voltage = 0.0
        self.samples_count = 0
        self.power_values: List[float] = []
        self.last_elapsed_time = 0.0
        # CodeCarbon specific
        self.energy_kwh = 0.0
        self.emissions_kg = 0.0


# Global flags for signal handling
quit_flag = False
prepare_to_quit = False


def setup_signal_handlers():
    """Set up signal handlers for graceful termination."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)


def signal_handler(sig, frame):
    """Signal handler for graceful termination."""
    global quit_flag, prepare_to_quit
    if sig == signal.SIGUSR1:
        print("\nReceived USR1 signal. Preparing to terminate...", file=__import__('sys').stderr)
        prepare_to_quit = True
    else:
        print("\nReceived termination signal. Shutting down...", file=__import__('sys').stderr)
        quit_flag = True
    if os.path.exists("fnirsi_stop"):
        os.remove("fnirsi_stop")


def calculate_stddev(values: List[float]) -> float:
    """Calculate standard deviation of values."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def save_energy_summary(state: MeasurementState, output_file: str, is_codecarbon: bool = False) -> Optional[str]:
    """Save energy data summary compatible with the benchmark system."""
    if state.start_time is None:
        return None
    
    if is_codecarbon:
        energy_wh = state.energy_kwh * 1000
        duration = state.duration if state.duration > 0 else (time.time() - state.start_time)
        avg_power = energy_wh / (duration / 3600) if duration > 0 else 0
        max_power = max(state.power_values) if state.power_values else avg_power
        max_voltage = 0.0
        max_current = 0.0
        temp = 0.0
    else:
        duration = time.time() - state.start_time
        avg_power = float(np.mean(state.power_values)) if state.power_values else 0
        max_power = state.max_power
        max_voltage = state.max_voltage
        max_current = state.max_current
        temp = state.temp_ema if state.temp_ema is not None else 0
        energy_wh = state.energy / 3600

    power_std = calculate_stddev(state.power_values)
    energy_std = power_std * duration / 3600 if power_std > 0 else 0

    summary_file = os.path.splitext(output_file)[0] + ".csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["timestamp", "voltage", "current", "power", "temperature",
                        "Power (W)", "Max Power (W)", "Energy (Wh)"])
        writer.writerow([time.time(), f"{max_voltage:.6f}", f"{max_current:.6f}",
                        f"{avg_power:.6f}", f"{temp:.1f}", f"{avg_power:.6f}",
                        f"{max_power:.6f}", f"{energy_wh:.6f}"])
        writer.writerow(["-----------"] * 8)
        writer.writerow([time.time(), f"{max_voltage:.6f}", f"{max_current:.6f}",
                        f"{avg_power:.6f}", f"{temp:.1f}", f"{avg_power:.6f}",
                        f"{max_power:.6f}", f"{energy_wh:.6f}"])
        writer.writerow(["0", "0", "0", f"{power_std:.6f}", "0",
                        f"{power_std:.6f}", "0", f"{energy_std:.6f}"])

    print(f"Energy summary saved to {summary_file}", file=__import__('sys').stderr)
    with open(f"{summary_file}.done", 'w') as f:
        f.write(f"Completed at {datetime.now().isoformat()}\n")
    return summary_file


def print_summary(state: MeasurementState, duration: float = 0, is_codecarbon: bool = False):
    """Print a summary of the collected measurements."""
    import sys
    
    if state.start_time is None:
        print("\nNo data was collected.", file=sys.stderr)
        return

    if is_codecarbon:
        actual_duration = state.duration if state.duration > 0 else duration
        energy_wh = state.energy_kwh * 1000
        avg_power = energy_wh / (actual_duration / 3600) if actual_duration > 0 else 0
        max_power = max(state.power_values) if state.power_values else avg_power
    else:
        actual_duration = duration if duration > 0 else time.time() - state.start_time
        actual_elapsed = state.last_elapsed_time if state.samples_count > 0 else actual_duration
        avg_power = float(np.mean(state.power_values)) if state.power_values else 0
        max_power = state.max_power
        energy_wh = state.energy / 3600

    power_std = calculate_stddev(state.power_values)

    print("\n---------- Measurement Summary ----------", file=sys.stderr)
    print(f"Duration: {actual_duration:.2f} seconds", file=sys.stderr)
    if not is_codecarbon and state.samples_count > 0:
        print(f"Measurement elapsed time: {actual_elapsed:.2f} seconds", file=sys.stderr)
        print(f"Effective Sample Rate: {state.samples_count/actual_elapsed:.2f} sps", file=sys.stderr)
    print(f"Samples collected: {state.samples_count}", file=sys.stderr)
    print(f"Average power: {avg_power:.6f} W", file=sys.stderr)
    print(f"Maximum power: {max_power:.6f} W", file=sys.stderr)
    print(f"Power std deviation: {power_std:.6f} W", file=sys.stderr)
    if not is_codecarbon:
        print(f"Maximum current: {state.max_current:.6f} A", file=sys.stderr)
        print(f"Maximum voltage: {state.max_voltage:.6f} V", file=sys.stderr)
    print(f"Total energy: {energy_wh:.6f} Wh", file=sys.stderr)
    if is_codecarbon:
        print(f"CO2 emissions: {state.emissions_kg * 1000:.3f} g", file=sys.stderr)
    if state.temp_ema is not None and not is_codecarbon:
        print(f"Last temperature: {state.temp_ema:.3f} °C", file=sys.stderr)
    print("-------------------------------------------", file=sys.stderr)


def merge_energy_data(energy_file: str, benchmark_file: str, verbose: bool = False) -> bool:
    """Merge energy data from energy CSV into benchmark CSV file."""
    if pd is None:
        print("Error: pandas required for merge. Install with: pip install pandas", file=__import__('sys').stderr)
        return False
    
    try:
        if not os.path.exists(energy_file) or not os.path.exists(benchmark_file):
            print(f"Error: File not found", file=__import__('sys').stderr)
            return False

        with open(energy_file, 'r') as f:
            energy_delimiter = ';' if ';' in f.readline() else ','
        with open(benchmark_file, 'r') as f:
            benchmark_delimiter = ';' if ';' in f.readline() else ','

        energy_df = pd.read_csv(energy_file, delimiter=energy_delimiter)
        benchmark_df = pd.read_csv(benchmark_file, delimiter=benchmark_delimiter)

        # Find separator and extract mean/std rows
        separator_indices = []
        for i, row in energy_df.iterrows():
            for col in energy_df.columns:
                if isinstance(row[col], str) and '---' in row[col]:
                    separator_indices.append(i)
                    break

        mean_row_idx = separator_indices[0] + 1 if separator_indices else len(energy_df) - 2
        std_row_idx = mean_row_idx + 1

        mean_power = float(energy_df.iloc[mean_row_idx]["Power (W)"])
        mean_max_power = float(energy_df.iloc[mean_row_idx]["Max Power (W)"])
        mean_energy = float(energy_df.iloc[mean_row_idx]["Energy (Wh)"])
        std_power = float(energy_df.iloc[std_row_idx]["Power (W)"])
        std_energy = float(energy_df.iloc[std_row_idx]["Energy (Wh)"])

        benchmark_df["Power (W)"] = None
        benchmark_df["Max Power (W)"] = None
        benchmark_df["Energy (Wh)"] = None

        benchmark_separators = []
        for i, row in benchmark_df.iterrows():
            for col in benchmark_df.columns:
                if isinstance(row[col], str) and '---' in str(row[col]):
                    benchmark_separators.append(i)
                    break

        for i, row in benchmark_df.iterrows():
            if i in benchmark_separators:
                benchmark_df.at[i, "Power (W)"] = '------------'
                benchmark_df.at[i, "Max Power (W)"] = '------------'
                benchmark_df.at[i, "Energy (Wh)"] = '------------'
            elif i == len(benchmark_df) - 1:
                benchmark_df.at[i, "Power (W)"] = f"{std_power:.6f}"
                benchmark_df.at[i, "Max Power (W)"] = "0.000000"
                benchmark_df.at[i, "Energy (Wh)"] = f"{std_energy:.6f}"
            else:
                benchmark_df.at[i, "Power (W)"] = f"{mean_power:.6f}"
                benchmark_df.at[i, "Max Power (W)"] = f"{mean_max_power:.6f}"
                benchmark_df.at[i, "Energy (Wh)"] = f"{mean_energy:.6f}"

        benchmark_df.to_csv(benchmark_file, index=False, sep=benchmark_delimiter)
        if verbose:
            print(f"Merged energy data into {benchmark_file}", file=__import__('sys').stderr)
        return True

    except Exception as e:
        print(f"Error merging energy data: {e}", file=__import__('sys').stderr)
        return False
