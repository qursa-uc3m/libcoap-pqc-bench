# CoAP-PQC Benchmark Suite

Step-by-step guide for running post-quantum CoAP benchmarks.

## Prerequisites

- Python environment: `source .bench-env/bin/activate`
- libcoap built with WolfSSL
- PSK key generated and activated (see main README)

## Step 1: Setup Environment

```bash
cd /path/to/libcoap-pqc-bench
source .bench-env/bin/activate
```

## Step 2: Network Emulation (Optional)

Skip this section for local-only benchmarks.

### Start the Network Emulation VM

```bash
# Launch existing VM
sudo ./network_emulation/setup_vm.sh --name vm-network

# Wait for login prompt, then configure traffic routing
sudo ./network_emulation/udp_config.sh
```

### Apply Network Conditions

```bash
# Baseline (no emulation)
sudo ./network_emulation/net_config.sh set fiducial

# Smart Home network
sudo ./network_emulation/net_config.sh set smart-home

# Smart Factory network
sudo ./network_emulation/net_config.sh set smart-factory

# Public Transport network
sudo ./network_emulation/net_config.sh set public-transport

# Check current configuration
sudo ./network_emulation/net_config.sh show

# Reset to no emulation
sudo ./network_emulation/net_config.sh reset
```

See [network_emulation/README.md](../network_emulation/README.md) for detailed setup.

## Step 3: Run Benchmarks

From the repository root folder:

### Basic Examples

```bash
# Simple benchmark (25 clients, PSK mode)
./libcoap-bench/run_benchmarks.sh -n 25 -security psk -resources time -y

# Observer mode (60 seconds)
./libcoap-bench/run_benchmarks.sh -n 25 -s 60 -security psk -resources example_data -y

# Parallel execution with energy monitoring
./libcoap-bench/run_benchmarks.sh -n 25 -parallelization parallel -energy -security pki,psk,nosec -resources time -iterations 5 -y
```

### Complete Benchmark Suite

Run for each network condition:

```bash
# 1. FIDUCIAL NETWORK
sudo ./network_emulation/net_config.sh set fiducial
./libcoap-bench/run_benchmarks.sh -n 25 -algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" -parallelization parallel -security "pki,psk,nosec" -iterations 5 -energy -y

# 2. SMART HOME NETWORK
sudo ./network_emulation/net_config.sh set smart-home
./libcoap-bench/run_benchmarks.sh -n 25 -algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" -parallelization parallel -security "pki,psk,nosec" -iterations 5 -energy -y

# 3. SMART FACTORY NETWORK
sudo ./network_emulation/net_config.sh set smart-factory
./libcoap-bench/run_benchmarks.sh -n 25 -algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" -parallelization parallel -security "pki,psk,nosec" -iterations 5 -energy -y

# 4. PUBLIC TRANSPORT NETWORK
sudo ./network_emulation/net_config.sh set public-transport
./libcoap-bench/run_benchmarks.sh -n 25 -algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" -parallelization parallel -security "pki,psk,nosec" -iterations 5 -energy -y

# Reset network after each set
sudo ./network_emulation/net_config.sh reset
```

## Step 4: Process Results

### Aggregate Data from Iterations

```bash
cd libcoap-bench/data

# Aggregate specific session (replace SESSION_ID, e.g., local_1205_w7)
python3 ../bench-data-manager.py aggregate --data-dir . --session-id SESSION_ID --iterations 5
```

This creates `aggregated/<SESSION_ID>/` with aggregated metrics.

## Step 5: Generate Plots

```bash
cd libcoap-plots

# Single session scatter plot
python3 bench-data-plots.py "duration" 1 --scatter --scenarios A \
    --data-dir ../libcoap-bench/data --custom-suffix "SESSION_ID" --p "parallel"

# Bar plot comparing scenarios
python3 bench-data-plots.py "Energy (Wh)" 1 --barplot --scenarios A,C \
    --data-dir ../libcoap-bench/data --custom-suffix "SESSION_ID"

# Or use the wrapper script
./plots_wrapper.sh "duration,Energy (Wh)" scatter A --session SESSION_ID
```

See [libcoap-plots/README.md](../libcoap-plots/README.md) for all visualization options.

## Data Organization

```text
libcoap-bench/data/
├── raw/                      # Raw iteration data
│   ├── local_1205_w7-1/
│   ├── local_1205_w7-2/
│   └── ...
├── aggregated/               # Aggregated statistics
│   └── local_1205_w7/
│       ├── iterations/       # Original raw data moved here
│       └── *.csv
├── plots/                    # Generated plots
│   └── local_1205_w7/
├── summaries/                # Session summaries
│   └── summary_local_1205_w7.txt
└── sessions.txt              # Session tracking
```

## Troubleshooting

```bash
# Check server connectivity
ping 192.168.0.157

# Verify VM is running and configured
sudo ./network_emulation/net_config.sh test

# Kill zombie CoAP processes
pgrep -f 'libcoap' | xargs -r kill -9

# Check energy monitor
python3 libcoap-bench/energy_monitor.py --backend codecarbon --list-devices
```

## Command Reference

```
./libcoap-bench/run_benchmarks.sh -n NUM_CLIENTS [OPTIONS]

Required:
  -n NUM_CLIENTS        Number of clients

Options:
  -algorithms ALGOS     Comma-separated: KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,...
  -security MODES       Comma-separated: pki,psk,nosec
  -resources RES        time (scenario A/C), async (scenario B), example_data
  -s TIME               Observer mode duration (seconds)
  -iterations N         Repeat each test N times
  -parallelization MODE background or parallel
  -energy               Enable energy measurements
  -rasp                 Server runs on Raspberry Pi
  -cert-filter PATTERN  Only run matching cert configs (PKI mode)
  -client-auth yes|no   Enable client certificate auth (PKI mode)
  -y                    Skip confirmation prompts
  -v                    Verbose output
```

## Related Documentation

- [Energy monitoring](energy/README.md) - FNIRSI and CodeCarbon backends
- [Network emulation](../network_emulation/README.md) - VM setup and network scenarios
- [Visualization](../libcoap-plots/README.md) - Plot types and metrics
- [Certificates](../certs/README.md) - PKI certificate management
