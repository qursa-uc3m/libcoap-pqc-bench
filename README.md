# Benchmarking Post-Quantum Cryptography in libcoap

Benchmarking post-quantum cryptographic algorithms in CoAP using liboqs, wolfSSL, and libcoap libraries.

## Installation

### PQC Dependencies

If you want to use Post-Quantum Cryptography, first install the dependencies:

```bash
./scripts/install_liboqs_for_wolfssl.sh
```

Then build wolfssl:

```bash
./scripts/install_wolfssl.sh
```

### libcoap Installation

Install libcoap dependencies:

```bash
sudo apt-get install -y autoconf automake libtool make gcc
```

You may also need:

```bash
sudo apt-get install autoconf-archive libwolfssl-dev libcunit1-dev pkg-config
```

And run the installation script with the desired options:

```bash
./scripts/install_libcoap.sh [wolfssl] [--groups-spec]
```

Flags:

- `wolfssl`: This option indicates that you want to configure libcoap with WolfSSL as the underlying cryptographic library. If not provided, the script will configure libcoap with OpenSSL.
- `--groups-spec`: When provided, this option will set specific cryptographic groups during the configuration phase. Indicate the desired groups in the script. If not provided, the script will configure libcoap with the default groups.

## Certificate Management

The framework includes a certificate management system that simplifies the use of different certificate types (RSA, Dilithium, Falcon) for benchmarking.

### Available Certificate Types

- **RSA**: Traditional RSA certificates (RSA_2048)
- **Dilithium**: Post-quantum signatures at different security levels (DILITHIUM_LEVEL2, DILITHIUM_LEVEL3, DILITHIUM_LEVEL5)
- **Falcon**: Post-quantum signatures at different security levels (FALCON_LEVEL1, FALCON_LEVEL5)

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
./certs/generate_certs.sh
```

This requires the [oqs-provider](https://github.com/open-quantum-safe/oqs-provider) for OpenSSL. Installation instructions are in the folder `certs/oqs_prov_install/`.

## Running Benchmarks

### Dependencies Installation

Install `perf` and `tshark`:

```bash
sudo apt install linux-tools-$(uname -r) tshark
```

Install the Python requirements:

```bash
# Using conda
conda create -n libcoap-bench python=3.10
conda activate libcoap-bench
pip install --no-cache-dir -r ./libcoap-bench/requirements.txt

# Using venv (matches your system's python version)
python3 -m venv .bench-env
source .bench-env/bin/activate
pip install --no-cache-dir -r ./libcoap-bench/requirements.txt
```

*Remark*: To perform the CPU cycles count in Raspberry Pi without relying on perf (not recommended at this point), the instructions described in [this site](https://matthewarcus.wordpress.com/2018/01/27/using-the-cycle-counter-registers-on-the-raspberry-pi-3/) must be followed. Note that some modifications may be necessary due to particularities of the Raspberry Pi being used. In the Raspberry Pi used for our experiments the following commands must be run every time it is rebooted:

```bash
cd enable_ccr_2024
make
sudo insmod enable_ccr.ko
dmesg | tail
gcc -Wall -O3 cycles.c -o cycles
time taskset 0x1 ./cycles
```

### Running the Benchmark

In one terminal, start the server:

```bash
./libcoap-bench/coap_benchmark_server.sh -sec-mode <pki|psk|nosec> [-rasp] [-cert-config <CONFIG>]
```

Where:

- `-sec-mode`: Security mode (pki, psk, or nosec).
- `-rasp`: Indicates whether the server is running on a Raspberry Pi.
- `-cert-config`: Certificate configuration to use (for PKI mode, e.g., DILITHIUM_LEVEL3).
- `-list-certs`: Lists available certificate configurations.

In another terminal, run the client (don't forget to activate the python environment):

```bash
./libcoap-bench/coap_benchmark_client.sh -n <positive_integer> -sec-mode <pki|psk|nosec> -r <time|async> [-confirm <con|non>] [-s <integer>=1] [-rasp] [-parallelization <background|parallel>] [-cert-config <CONFIG>] [-client-auth <yes|no>]
```

Where:

- `-n`: Number of clients making requests to the server.
- `-sec-mode`: Security mode (pki, psk, or nosec).
- `-r`: Resource that the client asks for. The resource "time" corresponds to scenario A (or C if `-confirm` is set to non). The resource "async" corresponds to scenario B.
- `-confirm`: Whether messages between client and server are confirmable. Mandatory if `-r` is set to time.
- `-s`: Sets the clients in observer mode with the specified number of seconds.
- `-rasp`: Indicates whether the server is running on a Raspberry Pi.
- `-parallelization`: Only needed when the `-s` parameter is provided. Indicates whether the clients run in the same core (background, default) or different cores (parallel).
- `-cert-config`: Certificate configuration to use (for PKI mode).
- `-client-auth`: Enable/disable client certificate authentication. Default is 'yes' (mutual authentication).
- `-list-certs`: Lists available certificate configurations.

### Output File Naming Convention

The benchmark creates CSV files with a naming pattern that reflects the test parameters:

```
udp[_rasp]_conv_stats_[ALGORITHM]_[<CERT_CONFIG>]_n<N>[_s<S>][_<P>]_<SEC_MODE>[_client-auth]_scenario<SCENARIO>
```

Where:
- `_rasp`: Present if the `-rasp` flag was used
- `ALGORITHM`: The KEM algorithm used (e.g., KYBER_LEVEL5) for PKI/PSK modes
- `N`: Number of clients specified with `-n`
- `_s<S>`: Present if the `-s` parameter was used
- `_<P>`: Parallelization mode (background or parallel) if specified
- `<SEC_MODE>`: Security mode (pki, psk, or nosec)
- `_<CERT_CONFIG>`: Present for PKI mode, indicates the certificate type used
- `_client-auth`: Present if client authentication was enabled in PKI mode
- `_scenario<SCENARIO>`: Indicates the scenario (A, B, or C)

Example:
```
udp_rasp_conv_stats_KYBER_LEVEL5_DILITHIUM_LEVEL3_n10_s30_parallel_pki_client-auth_scenarioA.csv
```

### Energy Measurement

To include energy consumption measurements in the CSV:

1. Install the GitHub repository [rd-usb](https://github.com/kolinger/rd-usb).
2. Connect the energy tester to your PC via Bluetooth:

```bash
sudo modprobe btusb
sudo systemctl restart bluetooth

# Use this if first time
sudo rfcomm bind 0 00:15:A6:01:AA:21
# Or this otherwise
sudo rfcomm connect hci0 00:15:A6:01:AA:21
```

3. Run `python3 web.py` in the `rd-usb` directory to open the energy tester interface.
4. Run the server and client as explained before.
5. Export the energy CSV from the `rd-usb` interface.
6. Run the `energy_analysis.sh` script:

```bash
./libcoap-bench/energy_analysis.sh ~/Downloads/2024-06-24.csv ./libcoap-bench/bench-data/udp_rasp_conv_stats_KYBER_LEVEL5_n10_s30_parallel_pki_scenarioA.csv
```

### Generating Plots

Once you have created CSV files with your benchmark results, you can generate plots to visualize the data:

#### Scatter Plots for Single Scenario Analysis

For detailed analysis of a single scenario (A, B, or C) with data points connected by lines:

```bash
python3 libcoap-bench/coap_benchmark_plots.py <metric> <algorithms_list> <cert_types_list> <n> <scenario> <rasp> [s] [p]
```

Where:

- `metric`: The metric to plot (e.g., 'duration', 'CPU cycles', 'Power (W)', 'Energy (Wh)')
- `algorithms_list`: Comma-separated list of KEM algorithms (e.g., 'KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5')
- `cert_types_list`: Comma-separated list of certificate types/signature algorithms (e.g., 'DILITHIUM_LEVEL2,RSA_2048')
- `n`: Number of clients used in the benchmark
- `scenario`: Single scenario to plot (A, B, or C)
- `rasp`: Whether the server was running on Raspberry Pi (True or False)
- `s`: Optional observer duration parameter
- `p`: Optional parallelization mode

Examples:

```bash
# Compare different KEM algorithms with DILITHIUM_LEVEL2 certificates
python3 libcoap-bench/coap_benchmark_plots.py "duration" "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" "DILITHIUM_LEVEL2" 50 A True

# Compare two certificate types for the same algorithms
python3 libcoap-bench/coap_benchmark_plots.py "duration" "KYBER_LEVEL1,KYBER_LEVEL3" "DILITHIUM_LEVEL2,RSA_2048" 50 A True

# Plot energy consumption data
python3 libcoap-bench/coap_benchmark_plots.py "Energy (Wh)" "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" "DILITHIUM_LEVEL2" 50 B True
```

#### Bar Plots for Multi-Scenario Comparison

For comparing multiple scenarios, algorithms, and certificate types in a single visualization:

```bash
python3 libcoap-bench/coap_benchmark_barplots.py <metric> <algorithms_list> <cert_types_list> <n> <rasp> <scenarios_list> [s] [p]
```

Where:

- `metric`: The metric to plot (e.g., 'duration', 'CPU cycles', 'Power (W)', 'Energy (Wh)')
- `algorithms_list`: Comma-separated list of KEM algorithms (e.g., 'KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5')
- `cert_types_list`: Comma-separated list of certificate types/signature algorithms (e.g., 'DILITHIUM_LEVEL2,RSA_2048')
- `n`: Number of clients used in the benchmark
- `rasp`: Whether the server was running on Raspberry Pi (True or False)
- `scenarios_list`: Comma-separated list of scenarios to compare (e.g., 'A,B,C')
- `s`: Optional observer duration parameter
- `p`: Optional parallelization mode

Examples:

```bash
# Compare all scenarios with different algorithms and certificate types
python3 libcoap-bench/coap_benchmark_barplots.py "duration" "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" "DILITHIUM_LEVEL2,RSA_2048" 50 True "A,B,C"

# Focus on energy metrics for selected scenarios
python3 libcoap-bench/coap_benchmark_barplots.py "Energy (Wh)" "KYBER_LEVEL1,KYBER_LEVEL3" "DILITHIUM_LEVEL2" 20 True "A,C"

# Observer mode with parallelization
python3 libcoap-bench/coap_benchmark_barplots.py "duration" "KYBER_LEVEL1,KYBER_LEVEL3" "DILITHIUM_LEVEL2" 20 True "A,B" 30 parallel
```

#### Understanding Metrics

The scripts support various metrics available in the CSV files:

- `duration`: Time taken for the benchmark (seconds)
- `CPU cycles`: CPU cycle count on the server
- `Power (W)`: Average power consumption
- `Max Power (W)`: Maximum power consumption
- `Energy (Wh)`: Total energy consumed

#### Output Files

All plots are saved to the `./bench-plots/` directory with filenames that reflect the parameters used, making it easy to identify specific visualizations. The format for filenames is:

- Scatter plots: `[rasp_]<metric>_n<N>[_s<S>][_<P>]_<ALGORITHMS>_<CERT_TYPES>_scenario<SCENARIO>.png`
- Bar plots: `barplot_[rasp_]<metric>_n<N>[_s<S>][_<P>]_<ALGORITHMS>_<CERT_TYPES>_<SCENARIOS>.png`
```

## Network Emulation

For information on how to set up and run network emulation with a Kernel-based Virtual Machine (KVM) and NetEm, see the [network_emulation/README.md](network_emulation/README.md).

## Cleaning Zombie Processes

You can check if there are any zombie processes with the following command:

```bash
sudo netstat -tulnp | grep -E '5683|5684'
```

You can remove them with the following command (note: this is for development, you have to ensure that you are not killing any important processes):

```bash
sudo pgrep -f 'libcoap' | while read pid; do sudo kill -9 $pid; done
```

## Analyzing the Traffic with Wireshark

See [OQS-wireshark](https://github.com/open-quantum-safe/oqs-demos/blob/main/wireshark/USAGE.md) for more details. Perhaps you need to run:

```console
xhost +si:localuser:root
```

instead of:

```console
xhost +si:localuser:$USER
```

if your user is not in the **docker** group. In that case:

```console
sudo docker run --net=host --privileged --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" openquantumsafe/wireshark
```

Then you can filter by `udp.port==5684` for DTLS or `udp.port==5683` for CoAP (or `udp.port==5684 || udp.port==5683` for both).