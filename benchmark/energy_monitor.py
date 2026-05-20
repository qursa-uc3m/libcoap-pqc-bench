#!/usr/bin/env python3
"""
Unified Energy Monitor for CoAP Benchmarking

Supports multiple backends:
  - fnirsi: FNIRSI USB power meters (FNB48, FNB58, C1, FNB48S)
  - codecarbon: Software-based power estimation via CodeCarbon

Usage:
  python3 energy_monitor.py --backend fnirsi [options]
  python3 energy_monitor.py --backend codecarbon [options]
"""

import sys
import argparse

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from energy.base import setup_signal_handlers, merge_energy_data


def main():
    setup_signal_handlers()
    
    parser = argparse.ArgumentParser(
        description="Unified Energy Monitor for CoAP Benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Backend selection
    parser.add_argument("--backend", "-b", type=str, default="fnirsi",
                       choices=["fnirsi", "codecarbon"],
                       help="Energy monitoring backend")
    
    # Common options
    parser.add_argument("--output", type=str, default="energy_data",
                       help="Output file name (without extension)")
    parser.add_argument("--duration", type=float, default=0,
                       help="Duration in seconds (0 = infinite)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--identify", action="store_true",
                       help="Check device/system and exit")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available devices and exit")
    parser.add_argument("--start-pipe", type=str,
                       help="Named pipe for start synchronization")
    parser.add_argument("--stop-pipe", type=str,
                       help="Named pipe for stop synchronization")
    
    # Merge mode
    parser.add_argument("--merge", metavar='ENERGY_FILE',
                       help="Merge energy data into benchmark CSV")
    parser.add_argument("--benchmark", metavar='BENCH_FILE',
                       help="Benchmark CSV file for merge")
    
    # FNIRSI-specific options
    parser.add_argument("--crc", action="store_true",
                       help="Enable CRC checks (fnirsi only)")
    parser.add_argument("--alpha", type=float, default=0.9,
                       help="Temperature smoothing factor (fnirsi only)")
    parser.add_argument("--force-reset", action="store_true",
                       help="Force USB device reset (fnirsi only)")
    parser.add_argument("--retry", type=int, default=3,
                       help="Retry attempts for device operations (fnirsi only)")
    
    args = parser.parse_args()
    
    # Handle merge mode (backend-independent)
    if args.merge:
        if not args.benchmark:
            print("Error: --benchmark required with --merge", file=sys.stderr)
            return 1
        return 0 if merge_energy_data(args.merge, args.benchmark, args.verbose) else 1
    
    # Route to appropriate backend
    if args.backend == "codecarbon":
        from energy import codecarbon_backend as backend
    else:
        from energy import fnirsi_backend as backend
    
    if not backend.check_availability():
        print(f"Error: {args.backend} backend not available", file=sys.stderr)
        return 1
    
    return backend.run(args)


if __name__ == "__main__":
    sys.exit(main())
