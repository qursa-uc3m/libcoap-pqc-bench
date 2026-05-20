#!/usr/bin/env python3
"""CodeCarbon-based energy monitoring backend."""

import os
import sys
import time
import csv
import logging
from datetime import datetime
from typing import Optional

from .base import (MeasurementState, quit_flag, prepare_to_quit,
                   save_energy_summary, print_summary)

logging.getLogger("codecarbon").setLevel(logging.ERROR)

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


def check_availability() -> bool:
    return CODECARBON_AVAILABLE


def check_system_compatibility() -> dict:
    """Check system support for power measurement via CodeCarbon."""
    info = {
        "codecarbon_available": CODECARBON_AVAILABLE,
        "rapl_available": False,
        "nvidia_gpu": False,
        "cpu_info": "Unknown"
    }
    
    if not CODECARBON_AVAILABLE:
        return info
    
    for path in ["/sys/class/powercap/intel-rapl", "/sys/class/powercap/intel-rapl:0"]:
        if os.path.exists(path):
            info["rapl_available"] = True
            break
    
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            info["nvidia_gpu"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_info"] = line.split(":")[1].strip()
                    break
    except:
        pass
    
    return info


def print_system_info() -> bool:
    """Print system compatibility information."""
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
        print("Install with: pip install codecarbon")
        return False
    
    if not info['rapl_available']:
        print("\nWARNING: Intel RAPL not available.")
        print("Power measurements will be TDP-based estimates.")
    
    print("\nSystem is ready for energy monitoring.")
    return True


def create_tracker(output_dir: str = None, measure_power_secs: float = 0.5):
    """Create and configure a CodeCarbon emissions tracker."""
    if not CODECARBON_AVAILABLE:
        return None
    try:
        return OfflineEmissionsTracker(
            country_iso_code="ESP",
            output_dir=output_dir or ".",
            output_file="codecarbon_emissions.csv",
            measure_power_secs=measure_power_secs,
            log_level="error",
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
        )
    except Exception as e:
        print(f"Error creating tracker: {e}", file=sys.stderr)
        return None


def run_measurement(duration: float, output_file: str,
                   start_pipe: str = None, stop_pipe: str = None,
                   verbose: bool = False) -> MeasurementState:
    """Run energy measurement for the specified duration."""
    from .base import quit_flag, prepare_to_quit
    import energy.base as base
    
    state = MeasurementState()
    output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    tracker = create_tracker(output_dir, measure_power_secs=0.5)
    if tracker is None:
        print("ERROR: Could not create CodeCarbon tracker", file=sys.stderr)
        return state
    
    raw_output_file = output_file + "_raw.csv"
    
    try:
        with open(raw_output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["sample", "timestamp", "elapsed", "energy_kwh_cumulative", "emissions_kg_cumulative"])
            
            state.start_time = time.time()
            tracker.start()
            
            if verbose:
                print(f"Started energy tracking at {datetime.now().isoformat()}", file=sys.stderr)
            
            if start_pipe and os.path.exists(start_pipe):
                try:
                    with open(start_pipe, 'w') as f:
                        f.write("READY\n")
                    if verbose:
                        print(f"Sent READY signal via {start_pipe}", file=sys.stderr)
                except Exception as e:
                    print(f"Error signaling readiness: {e}", file=sys.stderr)
            
            end_time = state.start_time + duration if duration > 0 else None
            sample_interval = 0.5
            last_sample_time = state.start_time
            
            while not base.quit_flag:
                current_time = time.time()
                
                if end_time and current_time >= end_time:
                    if verbose:
                        print("Specified duration reached", file=sys.stderr)
                    break
                
                if base.prepare_to_quit:
                    if verbose:
                        print("Preparing to quit...", file=sys.stderr)
                    if stop_pipe and os.path.exists(stop_pipe):
                        try:
                            with open(stop_pipe, 'w') as f:
                                f.write("DONE\n")
                        except Exception as e:
                            print(f"Error sending DONE signal: {e}", file=sys.stderr)
                    break
                
                if current_time - last_sample_time >= sample_interval:
                    elapsed = current_time - state.start_time
                    time_delta = current_time - last_sample_time
                    
                    try:
                        tracker.flush()
                        prev_energy = state.energy_kwh
                        if hasattr(tracker, '_total_energy'):
                            state.energy_kwh = tracker._total_energy.kWh
                        if hasattr(tracker, '_total_emissions'):
                            state.emissions_kg = tracker._total_emissions.kgCO2
                        
                        energy_delta_kwh = state.energy_kwh - prev_energy
                        if time_delta > 0 and energy_delta_kwh > 0:
                            instantaneous_power = (energy_delta_kwh * 1000) / (time_delta / 3600)
                            state.power_values.append(instantaneous_power)
                    except Exception as e:
                        if verbose:
                            print(f"Error getting energy data: {e}", file=sys.stderr)
                    
                    state.samples_count += 1
                    csv_writer.writerow([state.samples_count, f"{current_time:.6f}", f"{elapsed:.6f}",
                                        f"{state.energy_kwh:.9f}", f"{state.emissions_kg:.9f}"])
                    last_sample_time = current_time
                    csvfile.flush()
                
                time.sleep(0.1)
            
            tracker.stop()
            state.end_time = time.time()
            state.duration = state.end_time - state.start_time
            
            if hasattr(tracker, '_total_energy'):
                state.energy_kwh = tracker._total_energy.kWh
            if hasattr(tracker, '_total_emissions'):
                state.emissions_kg = tracker._total_emissions.kgCO2
                
    except Exception as e:
        print(f"Error during measurement: {e}", file=sys.stderr)
        try:
            tracker.stop()
        except:
            pass
    
    return state


def run(args) -> int:
    """Run CodeCarbon energy monitoring."""
    if args.identify or getattr(args, 'list_devices', False):
        return 0 if print_system_info() else 1
    
    if not CODECARBON_AVAILABLE:
        print("Error: CodeCarbon is not installed", file=sys.stderr)
        print("Install with: pip install codecarbon", file=sys.stderr)
        return 1
    
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
    
    print_summary(state, args.duration, is_codecarbon=True)
    save_energy_summary(state, args.output + ".csv", is_codecarbon=True)
    return 0
