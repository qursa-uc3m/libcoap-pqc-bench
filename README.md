# Benchmarking Post-Quantum Cryptography in libcoap

A comprehensive framework for benchmarking post-quantum cryptographic algorithms in CoAP using liboqs, wolfSSL, and libcoap libraries. This framework supports performance testing across different security modes, algorithms, and protocols with automated data collection and analysis.

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
  - [Metrics](#metrics)
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
- `--fork`: Clone from dasobral/wolfssl-liboqs.git (default)
- `--release [ver]`: Clone from wolfSSL/wolfssl.git with specified version (default: v5.7.6-stable)

You can also specify algorithm preferences with:

```bash
./scripts/install_wolfssl.sh --groups-spec=KYBER_LEVEL5
```

### libcoap Installation

Install libcoap dependencies:

```bash
sudo apt-get install -y autoconf automake libtool make gcc
sudo apt-get install autoconf-archive libwolfssl-dev libcunit1-dev pkg-config
```

And run the installation script with the desired options:

```bash
./scripts/install_libcoap.sh [wolfssl] [--groups-spec=ALGORITHM] [--install-dir=PATH]
```

Options:
- `wolfssl`: Configure libcoap with WolfSSL as the underlying crypto library (otherwise uses OpenSSL)
- `--groups-spec=ALGORITHM`: Set specific cryptographic groups during configuration (e.g., KYBER_LEVEL5)
- `--install-dir=PATH`: Specify a custom installation directory

## Certificate Management

The framework includes a certificate management system that simplifies the use of different certificate types for benchmarking. This is essential for the PKI security mode.

### Available Certificate Types

- **RSA**: Traditional RSA certificates (RSA_2048)
- **Dilithium**: Post-quantum signatures at different security levels (DILITHIUM_LEVEL2, DILITHIUM_LEVEL3, DILITHIUM_LEVEL5)
- **Falcon**: Post-quantum signatures at different security levels (FALCON_LEVEL1, FALCON_LEVEL5)
- **Elliptic Curve**: Traditional EC certificates (EC_P256, EC_ED25519)

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

Synchronize keys between test systems:

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

For Raspberry Pi CPU cycle counting (if not using perf):
```bash
cd enable_ccr_2024
make
sudo insmod enable_ccr.ko
dmesg | tail
gcc -Wall -O3 cycles.c -o cycles
time taskset 0x1 ./cycles
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
./libcoap-bench/coap_benchmark_client.sh -n <positive_integer> -sec-mode <pki|psk|nosec> -r <time|async> [-confirm <con|non>] [-s <integer>=1] [-rasp] [-parallelization <background|parallel>] [-cert-config <CONFIG>] [-client-auth <yes|no>]
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
- `-s TIME`: Time for observer mode in seconds
- `-parallelization MODE`: Parallelization mode (background/parallel)
- `-client-auth MODE`: Client authentication mode (yes/no)
- `-pause SECONDS`: Seconds to pause between benchmark runs
- `-energy`: Enable energy measurements
- `-cert-filter PATTERN`: Only run certificate configs matching pattern
- `-security MODES`: Security modes to test (comma-separated: pki,psk,nosec)
- `-resources RES`: Resources to test (time,async or async?N where N is delay seconds)
- `-async-delay SECONDS`: Set delay for async resource
- `-iterations N`: Run each test configuration N times
- `-y`: Skip confirmation prompts
- `-v`: Verbose output

Example:
```bash
./libcoap-bench/run_benchmarks.sh -n 50 -s 30 -parallelization parallel -client-auth yes -energy -iterations 3 -resources time,async -security pki,psk
```

### Filename Convention

The benchmark creates CSV files with a naming pattern that reflects the test parameters:

```
udp[_rasp]_conv_stats_[ALGORITHM]_[<CERT_CONFIG>]_n<N>[_s<S>][_<P>]_<SEC_MODE>[_client-auth]_scenario<SCENARIO>
```

Where:
- `_rasp`: Present if the `-rasp` flag was used
- `ALGORITHM`: The KEM algorithm used (e.g., KYBER_LEVEL5) for PKI/PSK modes
- `N`: Number of clients
- `_s<S>`: Present if the `-s` parameter was used
- `_<P>`: Parallelization mode (background or parallel)
- `<SEC_MODE>`: Security mode (pki, psk, or nosec)
- `_<CERT_CONFIG>`: Present for PKI mode, indicating the certificate type
- `_client-auth`: Present if client authentication was enabled
- `_scenario<SCENARIO>`: Indicates the scenario (A, B, or C)

Example:
```
udp_rasp_conv_stats_KYBER_LEVEL5_DILITHIUM_LEVEL3_n10_s30_parallel_pki_client-auth_scenarioA.csv
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
./libcoap-bench/run_benchmarks.sh -n 10 -energy -security pki -resources time -cert-filter DILITHIUM_LEVEL3
```

For manual control:

```bash
# Start a measurement for 30 seconds
python3 libcoap-bench/energy_monitor.py --duration 30 --output ./bench-data/my_test

# Merge energy data with benchmark results
python3 libcoap-bench/energy_monitor.py --merge ./bench-data/energy_data.csv --benchmark ./bench-data/benchmark_results.csv
```

## Data Visualization

The benchmark framework includes tools for visualizing benchmark results.

### Scatter Plots

For detailed analysis of a single scenario with data points connected by lines:

```bash
python3 libcoap-bench/coap_benchmark_plots.py --scatter <metric> <algorithms_list> <cert_types_list> <n> --scenarios <scenario> [options]
```

Example:
```bash
python3 libcoap-bench/coap_benchmark_plots.py --scatter "duration" "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" "DILITHIUM_LEVEL2" 50 --scenarios A --rasp
```

### Bar Plots

For comparing multiple scenarios, algorithms, and certificate types:

```bash
python3 libcoap-bench/coap_benchmark_plots.py --barplot <metric> <algorithms_list> <cert_types_list> <n> --scenarios <scenario_list> [options]
```

Example:
```bash
python3 libcoap-bench/coap_benchmark_plots.py --barplot "Energy (Wh)" "KYBER_LEVEL1,KYBER_LEVEL3" "DILITHIUM_LEVEL2" 20 --scenarios A,C --rasp
```

### Metrics

The visualization tools support various metrics:
- `duration`: Time taken for the benchmark (seconds)
- `CPU cycles`: CPU cycle count on the server
- `Power (W)`: Average power consumption
- `Max Power (W)`: Maximum power consumption
- `Energy (Wh)`: Total energy consumed

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

### Analyzing Traffic with Wireshark

See [OQS-wireshark](https://github.com/open-quantum-safe/oqs-demos/blob/main/wireshark/USAGE.md) for details.

Setup:
```bash
xhost +si:localuser:root  # If running as root
sudo docker run --net=host --privileged --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" openquantumsafe/wireshark
```

Filter by:
```
udp.port==5684 || udp.port==5683
```