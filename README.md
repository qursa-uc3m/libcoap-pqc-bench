# Benchmarking Post-Quantum Cryptography in libcoap

A benchmarking tool for post-quantum cryptographic algorithms in CoAP using liboqs, wolfSSL, and libcoap libraries. This tool enables performance testing across different security modes, algorithms and protocols with automated data collection and analysis.

## Table of Contents
- [Installation](#installation)
  - [PQC Dependencies](#pqc-dependencies)
  - [libcoap Installation](#libcoap-installation)
- [Certificate Management](#certificate-management)
  - [Available Certificate Types](#available-certificate-types)
  - [Managing Certificates](#managing-certificates)
  - [Generating New Certificates](#generating-new-certificates)
- [PSK Key Management](#psk-key-management)
  - [Generating Keys](#generating-keys)
  - [Managing Keys](#managing-keys)
  - [Deploying Keys](#deploying-keys)
- [Running Benchmarks](#running-benchmarks)
  - [Dependencies](#dependencies)
  - [Algorithm Selection](#algorithm-selection)
  - [Basic Benchmarks](#basic-benchmarks)
  - [Automated Benchmarks](#automated-benchmarks)
  - [Filename Convention](#filename-convention)
- [Energy Measurement](#energy-measurement)
  - [Hardware Setup](#hardware-setup)
  - [Software Configuration](#software-configuration)
  - [Running Measurements](#running-measurements)
- [Data Visualization](#data-visualization)
  - [Scatter Plots](#scatter-plots)
  - [Bar Plots](#bar-plots)
  - [Heat Maps](#heat-maps)
  - [Box Plots](#box-plots)
  - [Candlestick Plots](#candlestick-plots)
  - [Metrics](#metrics)
- [Data Processing and Analysis](#data-processing-and-analysis)
  - [Outlier Filtering](#outlier-filtering)
  - [Cross-Network Comparison](#cross-network-comparison)
- [Network Emulation](#network-emulation)
- [Utility Scripts](#utility-scripts)
- [Troubleshooting](#troubleshooting)

## Installation

### PQC Dependencies

If you want to use Post-Quantum Cryptography, first install the dependencies:

```bash
./scripts/install_liboqs_for_wolfssl.sh
```

Then build wolfssl:

```bash
./scripts/install_wolfssl.sh [--fork | --release [version]]
```

Options:
- `--fork`: Clone from dasobral/wolfssl-liboqs.git (default). This version fixes issues with DILITHIUM and FALCON certificates across different security levels.
- `--release [ver]`: Clone from wolfSSL/wolfssl.git with specified version (default: v5.7.6-stable)

### libcoap Installation

Install libcoap dependencies:

```bash
sudo apt-get install -y autoconf automake libtool make gcc
sudo apt-get install autoconf-archive libwolfssl-dev libcunit1-dev pkg-config
```

And run the installation script with the desired options:

```bash
./scripts/install_libcoap.sh [wolfssl] [--install-dir=PATH]
```

Options:
- `wolfssl`: Configure libcoap with WolfSSL as the underlying crypto library (otherwise uses OpenSSL)
- `--install-dir=PATH`: Specify a custom installation directory

## Certificate Management

The framework includes a certificate management system that simplifies the use of different certificate types for benchmarking. This is essential for the PKI security mode.

### Available Certificate Types

- **RSA**: Traditional RSA certificates (RSA_2048)
- **Elliptic Curve**: Traditional EC certificates (EC_P256, EC_ED25519)
- **Dilithium**: Post-quantum signatures at different security levels (DILITHIUM_LEVEL2, DILITHIUM_LEVEL3, DILITHIUM_LEVEL5)
- **Falcon**: Post-quantum signatures at different security levels (FALCON_LEVEL1, FALCON_LEVEL5)

### Generating New Certificates

If you need to generate new post-quantum certificates:

```bash
./certs/generate_certs.sh [--rasp] [--rpi-address ADDR] [--rpi-user USER]
```

Options:
- `--rasp`: Sync certificates to Raspberry Pi after generation
- `--rpi-address ADDR`: Specify Raspberry Pi IP address (default: 192.168.0.157)
- `--rpi-user USER`: Specify Raspberry Pi username (default: root)

This requires the [oqs-provider](https://github.com/open-quantum-safe/oqs-provider) for OpenSSL. Installation instructions are in the folder `certs/oqs_prov_install/`.

### Managing Certificates

Use the certificate configuration scripts to list, validate, and set up certificates:

```bash
# List available certificate configurations
./certs/config_certs.sh --list

# Validate a specific certificate configuration
./certs/config_certs.sh --validate DILITHIUM_LEVEL3

# Set up a certificate configuration for use
./certs/config_certs.sh --setup DILITHIUM_LEVEL3
```

## PSK Key Management

The framework includes a Pre-Shared Key (PSK) management system for handling cryptographic keys. This is essential for the PSK security mode.

### Generating Keys

Create new keys with different bit strengths:

```bash
# Generate a new 256-bit key (default)
./pskeys/psk_manager.sh generate

# Generate keys with specific bit lengths
./pskeys/psk_manager.sh generate 128
./pskeys/psk_manager.sh generate 256
./pskeys/psk_manager.sh generate 384
./pskeys/psk_manager.sh generate 512
```

### Managing Keys

List and activate keys for benchmarking:

```bash
# List all available PSK keys
./pskeys/psk_manager.sh list

# Show the currently active key
./pskeys/psk_manager.sh current

# Activate a specific key for use in benchmarks
./pskeys/psk_manager.sh activate psk_256_12345678.key
```

### Deploying Keys

Synchronize keys between test systems if server runs in a different instance or device (you might have to manually update the IP address in the script):

```bash
# Deploy keys to the Raspberry Pi
./pskeys/psk_manager.sh deploy
```

This ensures that both client and server use the same cryptographic material for PSK-based security.

## Running Benchmarks

### Dependencies

Install the required dependencies:

```bash
# Install perf and tshark
sudo apt install linux-tools-$(uname -r) tshark

# Install Python requirements
python3 -m venv .bench-env
source .bench-env/bin/activate
pip install --no-cache-dir -r ./libcoap-bench/requirements.txt
```

### Algorithm Selection


The framework now supports runtime algorithm selection without recompilation. This is a major improvement that allows testing different algorithms dynamically.

#### Supported Algorithms

**Key Exchange Algorithms:**
- `KYBER_LEVEL1`: NIST Level 1 security (~AES-128)
- `KYBER_LEVEL3`: NIST Level 3 security (~AES-192)
- `KYBER_LEVEL5`: NIST Level 5 security (~AES-256)

**Hybrid Algorithms (Classical + PQC):**
- `P256_KYBER_LEVEL1`: ECDH P-256 + KYBER Level 1
- `P384_KYBER_LEVEL3`: ECDH P-384 + KYBER Level 3
- `P521_KYBER_LEVEL5`: ECDH P-521 + KYBER Level 5

#### Manual Algorithm Testing

Test individual algorithms manually without recompilation:

```bash
# Start server (on Raspberry Pi or local machine)
coap-server -A 0.0.0.0 -k ./pskeys/psk_256_1744210857.key -u uc3m

# Test different algorithms from client
COAP_WOLFSSL_GROUPS=KYBER_LEVEL1 ./libcoap/build/bin/coap-client -k ./pskeys/psk_256_1744210857.key -u uc3m -m get coaps://[server-ip]/time
COAP_WOLFSSL_GROUPS=KYBER_LEVEL3 ./libcoap/build/bin/coap-client -k ./pskeys/psk_256_1744210857.key -u uc3m -m get coaps://[server-ip]/time
COAP_WOLFSSL_GROUPS=P256_KYBER_LEVEL1 ./libcoap/build/bin/coap-client -k ./pskeys/psk_256_1744210857.key -u uc3m -m get coaps://[server-ip]/time
```

### Basic Benchmarks

For manual benchmark execution, use the individual scripts:

#### Server Side
```bash
./libcoap-bench/coap_benchmark_server.sh -sec-mode <pki|psk|nosec> [-rasp] [-cert-config <CONFIG>] [-client-auth <yes|no>]
```

Options:
- `-sec-mode`: Security mode (pki, psk, or nosec)
- `-rasp`: Indicates whether the server is running on a Raspberry Pi
- `-cert-config`: Certificate configuration to use (for PKI mode)
- `-client-auth`: Enable/disable client certificate authentication (default: no)
- `-list-certs`: Lists available certificate configurations

#### Client Side
```bash
./libcoap-bench/coap_benchmark.sh -n <positive_integer> -sec-mode <pki|psk|nosec> -r <time|async> [-confirm <con|non>] [-s <integer>=1] [-rasp] [-parallelization <background|parallel>] [-cert-config <CONFIG>] [-client-auth <yes|no>]
```

Options:
- `-n`: Number of clients making requests to the server
- `-sec-mode`: Security mode (pki, psk, or nosec)
- `-r`: Resource type (time for scenario A/C, async for scenario B)
- `-confirm`: Whether messages are confirmable (con) or non-confirmable (non)
- `-s`: Sets the clients in observer mode with the specified number of seconds
- `-rasp`: Indicates whether the server is running on a Raspberry Pi
- `-parallelization`: How clients run (background or parallel)
- `-cert-config`: Certificate configuration to use (for PKI mode)
- `-client-auth`: Enable/disable client certificate authentication

### Automated Benchmarks

For comprehensive automated testing, use the benchmark runner:

```bash
./libcoap-bench/run_benchmarks.sh -n NUM_CLIENTS [OPTIONS]
```

Required arguments:
- `-n NUM_CLIENTS`: Number of clients for benchmarking

Optional arguments:
- `-algorithms ALGOS`: **NEW** Comma-separated list of algorithms to test (default: KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5)
- `-s TIME`: Time for observer mode in seconds
- `-parallelization MODE`: Parallelization mode (background/parallel)
- `-client-auth MODE`: Client authentication mode (yes/no)
- `-pause SECONDS`: Seconds to pause between benchmark runs
- `-energy`: Enable energy measurements (requires of a suitable USB meter and a dedicated script for parsing the data to a suitable format. We provide our own [here](./libcoap-bench))
- `-cert-filter PATTERN`: Only run certificate configs matching pattern
- `-security MODES`: Security modes to test (comma-separated: pki,psk,nosec)
- `-resources RES`: Resources to test (time,async or async?N where N is delay seconds)
- `-async-delay SECONDS`: Set delay for async resource
- `-iterations N`: Run each test configuration N times
- `-y`: Skip confirmation prompts
- `-v`: Verbose output

#### Basic Usage Examples:

```bash
# Test default algorithms (KYBER_LEVEL1, KYBER_LEVEL3, KYBER_LEVEL5)
./libcoap-bench/run_benchmarks.sh -n 25

# Test specific algorithms
./libcoap-bench/run_benchmarks.sh -n 25 -algorithms "KYBER_LEVEL1,P256_KYBER_LEVEL1"

# Test with multiple security modes
./libcoap-bench/run_benchmarks.sh -n 25 -security "pki,psk" -algorithms "KYBER_LEVEL1,KYBER_LEVEL3"
```

#### Advanced Usage Examples:

```bash
# Full hybrid algorithm testing
./libcoap-bench/run_benchmarks.sh -n 25 \
  -algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,P256_KYBER_LEVEL1,P384_KYBER_LEVEL3,P521_KYBER_LEVEL5" \
  -security "pki,psk" \
  -iterations 5 \
  -energy

# Observer mode testing with parallelization
./libcoap-bench/run_benchmarks.sh -n 50 \
  -s 30 \
  -parallelization parallel \
  -algorithms "KYBER_LEVEL1,KYBER_LEVEL3" \
  -resources "async?5"

# Certificate-specific PKI testing
./libcoap-bench/run_benchmarks.sh -n 25 \
  -algorithms "KYBER_LEVEL1,KYBER_LEVEL3" \
  -security pki \
  -cert-filter "DILITHIUM_LEVEL2,FALCON_LEVEL1" \
  -client-auth yes

# Complete benchmark with energy monitoring
./libcoap-bench/run_benchmarks.sh -n 50 -s 30 -parallelization parallel -client-auth yes -energy -iterations 3 -resources time,async -security pki,psk -algorithms "KYBER_LEVEL1,KYBER_LEVEL3,P256_KYBER_LEVEL1"
```

### Filename Convention

The benchmark creates CSV files with a naming pattern that reflects the test parameters:

```
udp[_rasp]_conv_stats_[ALGORITHM]_[<CERT_CONFIG>]_n<N>[_s<S>][_<P>]_<SEC_MODE>[_client-auth]_scenario<SCENARIO>
```

Where:
- `_rasp`: Present if the `-rasp` flag was used
- `ALGORITHM`: The KEM algorithm used (e.g., KYBER_LEVEL5, P256_KYBER_LEVEL1) for PKI/PSK modes
- `N`: Number of clients
- `_s<S>`: Present if the `-s` parameter was used
- `_<P>`: Parallelization mode (background or parallel)
- `<SEC_MODE>`: Security mode (pki, psk, or nosec)
- `_<CERT_CONFIG>`: Present for PKI mode, indicating the certificate type
- `_client-auth`: Present if client authentication was enabled
- `_scenario<SCENARIO>`: Indicates the scenario (A, B, or C)

Example:
```
udp_rasp_conv_stats_KYBER_LEVEL1_DILITHIUM_LEVEL3_n10_s30_parallel_pki_client-auth_scenarioA.csv
```

## Energy Measurement

The framework supports energy measurement using the FNIRSI FNB58 USB Fast Charge Tester or compatible devices.

### Hardware Setup

1. Connect the FNIRSI FNB58 to your computer via USB
2. Set up USB permissions:
```bash
sudo bash -c 'echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"2e3c\", ATTRS{idProduct}==\"5558\", MODE=\"0666\"" > /etc/udev/rules.d/99-fnirsi.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Software Configuration

The energy monitoring utility is already included in the repository:

```bash
# Check if the device is properly detected
python3 libcoap-bench/energy_monitor.py --list-devices
python3 libcoap-bench/energy_monitor.py --identify
```

### Running Measurements

Use the automated benchmark runner with the `-energy` flag to enable energy measurements:

```bash
./libcoap-bench/run_benchmarks.sh -n 10 -energy -security pki -resources time -cert-filter DILITHIUM_LEVEL3 -algorithms "KYBER_LEVEL1,KYBER_LEVEL3"
```

For manual control:

```bash
# Start a measurement for 30 seconds
python3 libcoap-bench/energy_monitor.py --duration 30 --output ./bench-data/my_test

# Merge energy data with benchmark results
python3 libcoap-bench/energy_monitor.py --merge ./bench-data/energy_data.csv --benchmark ./bench-data/benchmark_results.csv
```

## Data Visualization

The benchmark framework includes comprehensive tools for visualizing benchmark results with support for multiple plot types and metrics.

### Scatter Plots

For detailed analysis of a single scenario with data points connected by lines:

```bash
python3 libcoap-plots/bench-data-plots.py <metric> --algorithms <algorithms_list> --cert-types <cert_types_list> <n> --scatter --scenarios <scenario> [options]
```

Example:
```bash
python3 libcoap-plots/bench-data-plots.py duration --algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" --cert-types "DILITHIUM_LEVEL2" 50 --scatter --scenarios A --rasp
```

### Bar Plots

For comparing multiple scenarios, algorithms, and certificate types:

```bash
python3 libcoap-plots/bench-data-plots.py <metric> --algorithms <algorithms_list> --cert-types <cert_types_list> <n> --barplot --scenarios <scenario_list> [options]
```

Example:
```bash
python3 libcoap-plots/bench-data-plots.py "Energy (mWh)" --algorithms "KYBER_LEVEL1,KYBER_LEVEL3" --cert-types "DILITHIUM_LEVEL2" 20 --barplot --scenarios A,C --rasp
```

### Heat Maps

For visualizing performance across algorithm-certificate combinations:

```bash
python3 libcoap-plots/bench-data-plots.py <metric> --algorithms <algorithms_list> --cert-types <cert_types_list> <n> --heatmap --scenarios <scenario> [options]
```

Example:
```bash
python3 libcoap-plots/bench-data-plots.py duration --algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" --cert-types "RSA_2048,DILITHIUM_LEVEL2,FALCON_LEVEL1" 25 --heatmap --scenarios A --rasp
```

### Box Plots

For analyzing performance variability across configurations:

```bash
python3 libcoap-plots/bench-data-plots.py <metric> --algorithms <algorithms_list> --cert-types <cert_types_list> <n> --boxplot --scenarios <scenario> [options]
```

Example:
```bash
python3 libcoap-plots/bench-data-plots.py duration --algorithms "KYBER_LEVEL1,KYBER_LEVEL3" --cert-types "DILITHIUM_LEVEL2,FALCON_LEVEL1" 25 --boxplot --scenarios A --rasp
```

### Candlestick Plots

For discrete metrics showing min-max ranges with mode values:

```bash
python3 libcoap-plots/bench-data-plots.py <metric> --algorithms <algorithms_list> --cert-types <cert_types_list> <n> --candlestick --scenarios <scenario> [options]
```

Example:
```bash
python3 libcoap-plots/bench-data-plots.py total_frames --algorithms "KYBER_LEVEL1,KYBER_LEVEL3" --cert-types "DILITHIUM_LEVEL2" 25 --candlestick --scenarios A --rasp
```

### Metrics

The visualization tools support various metrics:

**Continuous Metrics:**
- `duration`: Time taken for the benchmark (seconds)
- `duration ms`: Duration in milliseconds
- `cpu_cycles`: CPU cycle count on the server
- `Power (W)`: Average power consumption
- `Max Power (W)`: Maximum power consumption
- `Energy (Wh)`: Total energy consumed
- `Energy (mWh)`: Total energy consumed in milliwatt-hours

**Discrete Metrics:**
- `total_frames`: Total number of CoAP frames
- `total_bytes`: Total number of bytes transferred
- `frames_sent`: Frames sent by client
- `frames_received`: Frames received by client
- `bytes_sent`: Bytes sent by client
- `bytes_received`: Bytes received by client

**Common Options:**
- `--rasp`: Use Raspberry Pi dataset
- `--s <value>`: Include observer mode data
- `--p <mode>`: Include parallelization mode data
- `--filtered`: Use filtered dataset (outliers removed)
- `--custom-suffix <suffix>`: Use custom data directory suffix
- `--data-dir <dir>`: Specify data directory

## Data Processing and Analysis

### Outlier Filtering

Remove timeout-affected iterations for cleaner statistical analysis based upon a simple threshold on the coefficient of variation (CV):

```bash
# Filter outliers using CV-based detection
python3 libcoap-plots/bench-data-filter.py <input_file_or_directory> [--cv-threshold 3.0] [--file-pattern "*.csv"]
```

This creates `*_filtered.csv` files with outliers removed and statistics recalculated. You can adjust the CV threshold to keep more or less outliers.

Example:
```bash
# Filter all CSV files in bench-data directory
python3 libcoap-plots/bench-data-filter.py ./libcoap-bench/bench-data --cv-threshold 3.0

# Filter a specific file
python3 libcoap-plots/bench-data-filter.py ./libcoap-bench/bench-data/udp_rasp_conv_stats_KYBER_LEVEL1_n25_psk_scenarioA.csv
```

### Cross-Network Comparison

Compare performance across different network conditions:

```bash
# Network impact analysis
python3 libcoap-plots/bench-data-compare.py
```

This tool supports:
- **Tradeoff plots**: Performance vs. energy consumption across networks
- **Spider plots**: Multi-metric network impact visualization
- **Statistical difference analysis**: Network condition impact quantification
- **Algorithm scaling analysis**: Performance scaling with algorithm complexity

This script spects a folder structure within libcoap-plots like bench-data-*/bench-data-#, where * can be any string identifying the experiment and # a string with the network type name. You might have to modify this names at the beginning of the script. 

## Network Emulation

For testing in simulated network conditions:

```bash
# Set up a new VM for emulation
sudo ./network_emulation/setup_vm.sh --install --name <vm_name>

# Launch an existing VM
sudo ./network_emulation/setup_vm.sh --name <vm_name>

# Configure traffic redirection through the VM
sudo ./network_emulation/udp_config.sh
```

Inside the VM, you can apply network conditions using NetEm:

```bash
# Add 100ms delay
sudo tc qdisc add dev <vm_interface> root netem delay 100ms

# Add packet loss
sudo tc qdisc add dev <vm_interface> root netem loss 10%

# Remove network conditions
sudo tc qdisc del dev <vm_interface> root
```

For detailed instructions, see [network_emulation/README.md](network_emulation/README.md).

## Utility Scripts

The repository includes several utility scripts:

### Data Management

For processing, merging, and aggregating benchmark data:

```bash
# Process raw benchmark data
python3 libcoap-bench/bench-data-manager.py process --input-dir <dir>

# Merge energy data with benchmark results
python3 libcoap-bench/bench-data-manager.py merge --energy-file <file> --benchmark-file <file>

# Aggregate data from multiple iterations
python3 libcoap-bench/bench-data-manager.py aggregate --session-id <id> --iterations <N>
```

### Plotting Wrapper Scripts

Batch generate plots for multiple networks:

```bash
# Generate plots for all networks and metrics
./libcoap-plots/plots_wrapper.sh "duration,Energy (Wh)" barplot A --filtered
```

## Troubleshooting

### Cleaning Zombie Processes

You can check if there are any zombie processes with:

```bash
sudo netstat -tulnp | grep -E '5683|5684'
```

You can remove them with:

```bash
sudo pgrep -f 'libcoap' | while read pid; do sudo kill -9 $pid; done
```

### Algorithm Selection Issues

If algorithm selection is not working:

```bash
# Verify libcoap was built with runtime algorithm support
./libcoap/build/bin/coap-client --help | grep -i wolfssl

# Check algorithm.txt file is being written
cat algorithm.txt

# Test manual algorithm selection
COAP_WOLFSSL_GROUPS=KYBER_LEVEL1 ./libcoap/build/bin/coap-client -k ./pskeys/active_psk.txt -u uc3m -m get coaps://[server-ip]/time
```

### Energy Monitoring Issues

If energy monitoring is not working:

```bash
# Check device detection
python3 libcoap-bench/energy_monitor.py --list-devices

# Verify USB permissions
lsusb | grep -i fnirsi
ls -la /dev/ttyACM*

# Test device communication
python3 libcoap-bench/energy_monitor.py --identify
```

### Data Analysis Issues

If plots are not generating correctly:

```bash
# Check data file structure
head -20 libcoap-bench/bench-data/udp_rasp_conv_stats_*.csv

# Verify Python dependencies
pip list | grep -E "pandas|matplotlib|numpy|scipy"

# Test with minimal dataset
python3 libcoap-plots/bench-data-plots.py duration --algorithms "KYBER_LEVEL1" --cert-types "RSA_2048" 25 --scatter --scenarios A
```

### Analyzing Traffic with Wireshark

See [OQS-wireshark](https://github.com/open-quantum-safe/oqs-demos/blob/main/wireshark/USAGE.md) for details.

Setup:
```bash
xhost +si:localuser:root  # If running as root
sudo docker run --net=host --privileged --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" openquantumsafe/wireshark
```

You can just run our helper script with sudo privileges:
```bash
sudo ./oqs_wireshark.sh
```

Filter by:
```
udp.port==5684 || udp.port==5683
```