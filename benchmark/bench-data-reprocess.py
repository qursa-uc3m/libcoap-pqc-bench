#!/usr/bin/env python3
"""
Complete benchmark reprocessor - preserves raw durations, cpu_cycles,
energy mean/std via merge mode, handles aggregation sessions.
Enhanced with observer mode support to filter out noise durations.
"""
import os
import time
import pandas as pd
from pathlib import Path
import subprocess
import shutil
import argparse
import re

class MetricPreserver:
    def __init__(self, base_dir: str, data_manager_path: str):
        self.base_dir = Path(base_dir)
        self.data_manager = data_manager_path
        self.processed = 0
        self.skipped = 0
        self.errors = []
        self.aggregate_count = 0

    def extract_observer_time(self, filename: str) -> int:
        """
        Extract the observer subscription time from filename pattern _s#_
        Returns the time value or 0 if not an observer mode file.
        
        Example: 'udp_rasp_conv_stats_KYBER_LEVEL1_DILITHIUM_LEVEL2_n5_s60_parallel_pki_scenarioB.csv'
        Returns: 60
        """
        match = re.search(r'_s(\d+)_', filename)
        if match:
            return int(match.group(1))
        return 0

    def filter_observer_durations(self, durations: list, min_duration: float) -> list:
        """
        Filter durations to keep only those >= min_duration (observer conversations).
        This removes the noise from quick update operations (milliseconds).
        
        Example: [0.025, 0.025, 0.024, 0.027, 60.105, 60.104, 60.106, 60.105, 60.106]
        With min_duration=54 -> [60.105, 60.104, 60.106, 60.105, 60.106]
        """
        filtered = [d for d in durations if d >= min_duration]
        print(f"    Observer mode: filtered {len(durations)} -> {len(filtered)} durations (min: {min_duration}s)")
        if len(filtered) != len(durations):
            removed = [d for d in durations if d < min_duration]
            print(f"    Removed update noise: {removed[:5]}{'...' if len(removed) > 5 else ''} ({len(removed)} total)")
            print(f"    Kept observer durations: {filtered[:3]}{'...' if len(filtered) > 3 else ''}")
        return filtered

    def has_extended_stats(self, csv_file: Path) -> bool:
        """Check if CSV already has extended statistics (separator lines)."""
        try:
            with csv_file.open('r') as f:
                return any('======' in line for line in f)
        except:
            return False

    def extract_metrics(self, csv_file: Path) -> dict:
        """
        Extract raw durations, cpu_cycles and energy mean/std stats.
        For observer mode files, filter durations to remove update noise.
        
        Example raw data section:
        0.025
        0.025  
        0.024
        0.027    <- These are quick updates (milliseconds)
        60.105
        60.104   <- These are actual observers (~subscription time)
        60.106
        60.105
        60.106
        -----------  <- separator
        (statistics calculated from ALL above)
        
        We want to recalculate statistics using only the observer durations.
        """
        try:
            df = pd.read_csv(csv_file, sep=';')
            sep_idx = next((i for i,row in df.iterrows() if str(row.iloc[0]).startswith('---')), None)
            if sep_idx is None:
                print(f"    No separator in {csv_file.name}")
                return {}
            
            metrics = {}
            
            # Extract raw durations from the raw data section (before separator)
            if 'duration' in df.columns:
                raw_durations = []
                for i in range(sep_idx):
                    val = df.at[i, 'duration']
                    if pd.notna(val) and str(val).strip():
                        try:
                            raw_durations.append(float(val))
                        except ValueError:
                            continue
                
                if raw_durations:
                    # Check if this is an observer mode file
                    observer_time = self.extract_observer_time(csv_file.name)
                    
                    if observer_time > 0:
                        # Filter durations for observer mode
                        # Use a threshold slightly less than observer_time to account for timing variations
                        min_duration = observer_time * 0.9  # 90% threshold (e.g., 54s for 60s observers)
                        filtered_durations = self.filter_observer_durations(raw_durations, min_duration)
                        
                        if filtered_durations:
                            metrics['durations'] = filtered_durations
                            metrics['observer_mode'] = True
                            metrics['observer_time'] = observer_time
                        else:
                            print(f"    Warning: No observer durations found >= {min_duration}s")
                            metrics['durations'] = raw_durations  # Fallback to all durations
                            metrics['observer_mode'] = False
                    else:
                        # Regular mode - use all durations
                        metrics['durations'] = raw_durations
                        metrics['observer_mode'] = False
            
            # Extract cpu_cycles mean (from first stats row after separator)
            mean_idx = sep_idx + 1
            if 'cpu_cycles' in df.columns and mean_idx < len(df):
                val = df.at[mean_idx,'cpu_cycles']
                if pd.notna(val):
                    metrics['cpu_cycles'] = int(float(val))
            
            # Extract energy mean/std (from stats rows after separator)
            stats = ['mean','std']
            energy_cols = ['Power (W)','Max Power (W)','Energy (Wh)']
            for j, label in enumerate(stats, start=1):
                idx = sep_idx + j
                if idx >= len(df): 
                    break
                for col in energy_cols:
                    if col in df.columns:
                        val = df.at[idx, col]
                        if pd.notna(val) and str(val).strip():
                            metrics[f"{col}_{label}"] = float(val)
            
            return metrics
            
        except Exception as e:
            print(f"    Error extracting metrics: {e}")
            return {}

    def create_temp_files(self, metrics: dict, temp_dir: Path) -> dict:
        """Generate temp files: time, cycles, and energy_data.csv."""
        files = {}
        
        # Create durations file
        if 'durations' in metrics:
            tf = temp_dir / 'time_output.txt'
            tf.write_text("\n".join(map(str, metrics['durations'])))
            files['time'] = tf
            mode_info = "(observer mode)" if metrics.get('observer_mode', False) else ""
            print(f"    Created time file ({len(metrics['durations'])} values) {mode_info}")
        
        # Create cpu_cycles file
        if 'cpu_cycles' in metrics:
            cf = temp_dir / 'cycles_output.txt'
            cf.write_text(str(metrics['cpu_cycles']))
            files['cycles'] = cf
            print("    Created cycles file")
        
        # Create energy_data.csv with preserved mean/std
        energy_cols = ['Power (W)', 'Max Power (W)', 'Energy (Wh)']
        if any(f"{col}_mean" in metrics or f"{col}_std" in metrics for col in energy_cols):
            ef = temp_dir / 'energy_data.csv'
            hdr = ['timestamp', 'voltage', 'current', 'power', 'temperature'] + energy_cols
            lines = []
            
            # Header
            lines.append(';'.join(hdr))
            
            # Raw placeholder row: zeros for first fields, mean values for energy
            raw = ['0', '0', '0', '0', '0'] + [str(metrics.get(f"{col}_mean", '')) for col in energy_cols]
            lines.append(';'.join(raw))
            
            # Separator of dashes
            lines.append(';'.join(['-----------'] * len(hdr)))
            
            # Mean row
            mean_row = ['0', '0', '0', '0', '0'] + [str(metrics.get(f"{col}_mean", '')) for col in energy_cols]
            lines.append(';'.join(mean_row))
            
            # Std row
            std_row = ['0', '0', '0', '0', '0'] + [str(metrics.get(f"{col}_std", '')) for col in energy_cols]
            lines.append(';'.join(std_row))
            
            ef.write_text("\n".join(lines))
            files['energy'] = ef
            print("    Created energy file with mean & std")
        
        return files

    def process_file(self, stats_file: Path):
        """Process one stats file, then merge energy if present."""
        csv_file = stats_file.with_suffix('.csv')
        print(f"\nProcessing {stats_file.name}...")
        
        # Check if this is an observer mode file that needs reprocessing
        is_observer_mode = self.extract_observer_time(stats_file.name) > 0
        
        # Always reprocess observer mode files (they need duration filtering)
        if is_observer_mode:
            if csv_file.exists() and self.has_extended_stats(csv_file):
                print("    Observer mode file - reprocessing to filter durations")
            else:
                print("    Observer mode file detected")
        # Skip regular files if they already have extended stats
        elif csv_file.exists() and self.has_extended_stats(csv_file):
            print("    Skipping, already extended")
            self.skipped += 1
            return
        
        # Create temporary directory
        tmp = stats_file.parent / '.temp'
        tmp.mkdir(exist_ok=True)
        
        # Extract metrics (with observer mode filtering if applicable)
        metrics = self.extract_metrics(csv_file) if csv_file.exists() else {}
        if metrics:
            observer_info = f" [Observer: {metrics['observer_time']}s]" if metrics.get('observer_mode') else ""
            print(f"    Extracted: {sorted(metrics.keys())}{observer_info}")
        
        # Create temporary files
        files = self.create_temp_files(metrics, tmp)
        
        # Run the data manager process command
        cmd = ["python3", self.data_manager, "process",
               "--stats-file", str(stats_file),
               "--output-file", str(csv_file)]
        
        if 'time' in files:
            cmd += ["--time-file", str(files['time'])]
        if 'cycles' in files:
            cmd += ["--cycles-file", str(files['cycles'])]
        if 'energy' in files:
            cmd += ["--energy-file", str(files['energy'])]
        
        # Execute process command
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"    ✗ process failed: {res.stderr}")
            self.errors.append((str(stats_file), res.stderr))
            shutil.rmtree(tmp, ignore_errors=True)
            return
        
        print("    ✓ processed")
        self.processed += 1
        
        # Merge energy data to inject std row
        if 'energy' in files:
            merge_cmd = ["python3", self.data_manager, "merge",
                         "--energy-file", str(files['energy']),
                         "--benchmark-file", str(csv_file)]
            m = subprocess.run(merge_cmd, capture_output=True, text=True)
            if m.returncode == 0:
                print("    ✓ energy merged")
            else:
                print(f"    ✗ merge failed: {m.stderr}")
                self.errors.append((str(stats_file), m.stderr))
        
        # Clean up temporary files
        shutil.rmtree(tmp, ignore_errors=True)

    def run(self, backup: bool = True):
        """Run the complete reprocessing workflow."""
        print(f"Starting reprocess in {self.base_dir}")
        print("Enhanced with observer mode support for filtering noise durations\n")
        
        if backup:
            bkp = f"{self.base_dir}_backup_{int(time.time())}"
            shutil.copytree(self.base_dir, bkp)
            print(f"Backup created: {bkp}\n")
        
        # Process individual sessions
        for session in sorted(self.base_dir.glob('bench-data-*')):
            if session.name.startswith('bench-data-agg-'): 
                continue
            
            print(f"\n=== SESSION {session.name} ===")
            for stats_file in session.glob('udp_*.txt'):
                self.process_file(stats_file)
        
        # Process aggregated sessions
        for agg in sorted(self.base_dir.glob('bench-data-agg-*')):
            sid = agg.name.replace('bench-data-agg-', '')
            iters = agg / 'iterations'
            
            if not iters.exists(): 
                continue
            
            print(f"\n=== AGG SESSION {sid} ===")
            
            # Process iterations
            for iteration in sorted(iters.glob(f'bench-data-{sid}-*')):
                print(f"\n-- ITER {iteration.name} --")
                for stats_file in iteration.glob('udp_*.txt'):
                    self.process_file(stats_file)
            
            # Run aggregate command
            print(f"\nRunning aggregate...")
            agg_cmd = ["python3", self.data_manager, "aggregate",
                       "--session-id", sid,
                       "--iterations", str(len(list(iters.iterdir()))),
                       "--data-dir", str(self.base_dir),
                       "--output-dir", str(agg)]
            
            res = subprocess.run(agg_cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print("    ✓ aggregate done")
                self.aggregate_count += 1
            else:
                print(f"    ✗ aggregate failed: {res.stderr}")
                self.errors.append((sid, 'aggregate', res.stderr))
            
            # Reprocess aggregated CSVs
            print(f"\n-- REPROCESS AGG CSVs --")
            for stats_file in agg.glob('udp_*.txt'):
                self.process_file(stats_file)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Processed: {self.processed}")
        print(f"Skipped: {self.skipped}")
        print(f"Aggregated: {self.aggregate_count}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for err in self.errors[:10]:
                print(f"  {err}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Complete reprocessor with energy merge and observer mode support")
    parser.add_argument("base_dir", help="Base directory containing bench-data-* folders")
    parser.add_argument("--data-manager", default="./bench-data-manager.py",
                       help="Path to bench-data-manager.py script")
    parser.add_argument("--no-backup", action="store_true",
                       help="Skip creating backup before processing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_manager):
        print(f"Error: data-manager not found at {args.data_manager}")
        return 1
    
    if not os.path.isdir(args.base_dir):
        print(f"Error: base directory not found: {args.base_dir}")
        return 1
    
    # Create and run the processor
    processor = MetricPreserver(args.base_dir, args.data_manager)
    processor.run(backup=not args.no_backup)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())