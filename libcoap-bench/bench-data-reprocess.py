#!/usr/bin/env python3
"""
Complete benchmark reprocessor - preserves raw durations, cpu_cycles,
energy mean/std via merge mode, handles aggregation sessions.
"""
import os
import time
import pandas as pd
from pathlib import Path
import subprocess
import shutil
import argparse

class MetricPreserver:
    def __init__(self, base_dir: str, data_manager_path: str):
        self.base_dir = Path(base_dir)
        self.data_manager = data_manager_path
        self.processed = 0
        self.skipped = 0
        self.errors = []
        self.aggregate_count = 0

    def has_extended_stats(self, csv_file: Path) -> bool:
        try:
            with csv_file.open('r') as f:
                return any('======' in line for line in f)
        except:
            return False

    def extract_metrics(self, csv_file: Path) -> dict:
        """Extract raw durations, cpu_cycles and energy mean/std stats."""
        try:
            df = pd.read_csv(csv_file, sep=';')
            sep_idx = next((i for i,row in df.iterrows() if str(row.iloc[0]).startswith('---')), None)
            if sep_idx is None:
                print(f"    No separator in {csv_file.name}")
                return {}
            metrics = {}
            # raw durations
            if 'duration' in df.columns:
                raw = [float(v) for v in df.loc[:sep_idx-1,'duration'] if pd.notna(v)]
                if raw:
                    metrics['durations'] = raw
            # cpu_cycles mean
            mean_idx = sep_idx + 1
            if 'cpu_cycles' in df.columns and mean_idx < len(df):
                val = df.at[mean_idx,'cpu_cycles']
                if pd.notna(val):
                    metrics['cpu_cycles'] = int(float(val))
            # energy mean/std
            stats = ['mean','std']
            energy_cols = ['Power (W)','Max Power (W)','Energy (Wh)']
            for j,label in enumerate(stats, start=1):
                idx = sep_idx + j
                if idx >= len(df): break
                for col in energy_cols:
                    if col in df.columns:
                        val = df.at[idx,col]
                        if pd.notna(val) and str(val).strip():
                            metrics[f"{col}_{label}"] = float(val)
            return metrics
        except Exception as e:
            print(f"    Error extracting metrics: {e}")
            return {}

    def create_temp_files(self, metrics: dict, temp_dir: Path) -> dict:
        """Generate temp files: time, cycles, and energy_data.csv."""
        files = {}
        # durations
        if 'durations' in metrics:
            tf = temp_dir / 'time_output.txt'
            tf.write_text("\n".join(map(str, metrics['durations'])))
            files['time'] = tf
            print(f"    Created time file ({len(metrics['durations'])} values)")
        # cpu_cycles
        if 'cpu_cycles' in metrics:
            cf = temp_dir / 'cycles_output.txt'
            cf.write_text(str(metrics['cpu_cycles']))
            files['cycles'] = cf
            print("    Created cycles file")
        # energy_data.csv
        energy_cols = ['Power (W)','Max Power (W)','Energy (Wh)']
        if any(f"{col}_mean" in metrics or f"{col}_std" in metrics for col in energy_cols):
            ef = temp_dir / 'energy_data.csv'
            hdr = ['timestamp','voltage','current','power','temperature'] + energy_cols
            lines = []
            # header
            lines.append(';'.join(hdr))
            # raw placeholder row: zeros for first fields, mean values for energy
            raw = ['0','0','0','0','0'] + [str(metrics.get(f"{col}_mean", '')) for col in energy_cols]
            lines.append(';'.join(raw))
            # separator of dashes
            lines.append(';'.join(['-----------'] * len(hdr)))
            # mean row
            mean_row = ['0','0','0','0','0'] + [str(metrics.get(f"{col}_mean", '')) for col in energy_cols]
            lines.append(';'.join(mean_row))
            # std row
            std_row = ['0','0','0','0','0'] + [str(metrics.get(f"{col}_std", '')) for col in energy_cols]
            lines.append(';'.join(std_row))
            ef.write_text("\n".join(lines))
            files['energy'] = ef
            print("    Created energy file with mean & std")
        return files

    def process_file(self, stats_file: Path):
        """Process one stats file, then merge energy if present."""
        csv_file = stats_file.with_suffix('.csv')
        print(f"\nProcessing {stats_file.name}...")
        if csv_file.exists() and self.has_extended_stats(csv_file):
            print("    Skipping, already extended")
            self.skipped += 1
            return
        tmp = stats_file.parent / '.temp'
        tmp.mkdir(exist_ok=True)
        metrics = self.extract_metrics(csv_file) if csv_file.exists() else {}
        if metrics:
            print(f"    Extracted: {sorted(metrics.keys())}")
        files = self.create_temp_files(metrics, tmp)
        # first, run process to regenerate base CSV (with mean energy if loaded)
        cmd = ["python3", self.data_manager, "process",
               "--stats-file", str(stats_file),
               "--output-file", str(csv_file)]
        if 'time' in files:
            cmd += ["--time-file", str(files['time'])]
        if 'cycles' in files:
            cmd += ["--cycles-file", str(files['cycles'])]
        if 'energy' in files:
            cmd += ["--energy-file", str(files['energy'])]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"    ✗ process failed: {res.stderr}")
            self.errors.append((str(stats_file), res.stderr))
            shutil.rmtree(tmp, ignore_errors=True)
            return
        print("    ✓ processed")
        self.processed += 1
        # now merge energy to inject std row
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
        shutil.rmtree(tmp, ignore_errors=True)

    def run(self, backup: bool = True):
        print(f"Starting reprocess in {self.base_dir}\n")
        if backup:
            bkp = f"{self.base_dir}_backup_{int(time.time())}"
            shutil.copytree(self.base_dir, bkp)
            print(f"Backup created: {bkp}\n")
        # handle single sessions
        for session in sorted(self.base_dir.glob('bench-data-*')):
            if session.name.startswith('bench-data-agg-'): continue
            print(f"\n=== SESSION {session.name} ===")
            for st in session.glob('udp_*.txt'):
                self.process_file(st)
        # handle aggregated sessions
        for agg in sorted(self.base_dir.glob('bench-data-agg-*')):
            sid = agg.name.replace('bench-data-agg-','')
            iters = agg / 'iterations'
            if not iters.exists(): continue
            print(f"\n=== AGG SESSION {sid} ===")
            for it in sorted(iters.glob(f'bench-data-{sid}-*')):
                print(f"\n-- ITER {it.name} --")
                for st in it.glob('udp_*.txt'):
                    self.process_file(st)
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
                self.errors.append((sid,'aggregate',res.stderr))
            print(f"\n-- REPROCESS AGG CSVs --")
            for st in agg.glob('udp_*.txt'):
                self.process_file(st)
        # summary
        print("\n==== DONE ====")
        print(f"Processed: {self.processed}, Skipped: {self.skipped}, Aggregated: {self.aggregate_count}")
        if self.errors:
            print(f"Errors ({len(self.errors)}):")
            for err in self.errors[:10]: print(f"  {err}")


def main():
    parser = argparse.ArgumentParser(description="Complete reprocessor with energy merge")
    parser.add_argument("base_dir")
    parser.add_argument("--data-manager", default="./bench-data-manager.py")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.data_manager):
        print(f"Error: data-manager not found at {args.data_manager}")
        return 1
    pres = MetricPreserver(args.base_dir, args.data_manager)
    pres.run(backup=not args.no_backup)
    return 0

if __name__ == '__main__':
    import sys; sys.exit(main())
