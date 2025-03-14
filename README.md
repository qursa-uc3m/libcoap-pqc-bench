# Benchmarking Post-Quantum Cryptography in libcoap

Benchmarking post-quantum cryptographic algorithms in CoAP using liboqs, OpenSSL with DTLS 1.3 support, and libcoap libraries.

# CoAP with DTLS 1.3 and Post-Quantum Cryptography

## Installation

### System Dependencies

Ensure you have the required system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake
sudo apt-get install -y autoconf automake libtool make gcc
sudo apt-get install -y autoconf-archive pkg-config libcunit1-dev
```

### OpenSSL with DTLS 1.3 Support
First, install OpenSSL with DTLS 1.3 support:

```bash
./scripts/install_ossl-dtls13.sh -p /opt/openssl_dtls13
```

Options:
- `-p <install_dir>`: Optional. Set the installation directory. Default is `/opt/openssl_dtls13`.
- `-d <debug>`: Optional. Enable debug mode (1) or disable (0). Default is 0.

### Post-Quantum Cryptography Support
Next, install liboqs and the oqs-provider to enable PQC support:

```bash
./scripts/install_liboqs_for_ossl-dtsl13.sh -p /opt/openssl_dtls13 -v 0.11.0
```

Options:
- `-p <install_dir>`: The OpenSSL installation directory. Should match the one used in previous step.
- `-v <version>`: Optional. The liboqs version to install. Default is liboqs 0.11.0 and OQS provider 0.7.0. See this page for version [alignments](https://openquantumsafe.org/applications/tls.html).

### Environment Configuration
Set up the environment to use the DTLS 1.3 OpenSSL installation:

```bash
source ./openssl-env.sh dtls13
```

You should run this before using any OpenSSL commands or building libcoap.

### Certificate Generation
Generate the certificates needed for testing:

```bash
./certs/generate_scripts.sh
```

This script creates certificates for various post-quantum signature algorithms.

### libcoap Installation
Install libcoap with OpenSSL DTLS 1.3 support:

```bash
./scripts/install_libcoap.sh --openssl-dir=$OPENSSL_DIR --algorithm="kyber768" --groups-spec
```

Options:
- `--openssl-dir=<dir>`: Path to the OpenSSL installation directory. Use `$OPENSSL_DIR` when environment is configured.
- `--algorithm=<algs>`: Specify the supported groups/algorithms. Default is "kyber768".
- `--groups-spec`: Enable group specification.
- `--install-dir=<dir>`: Set a custom installation directory for libcoap.

## Testing DTLS 1.3 with PQC

The test script allows you to verify your DTLS 1.3 and PQC setup with different signature algorithms:

```bash
# Start the server in one terminal
./scripts/test_dtls13_pqc.sh -m server -a dilithium5

# In another terminal, run the client
./scripts/test_dtls13_pqc.sh -m client -a dilithium5
```

Options:
- `-m <mode>`: Specify "server", "client", or "both" (default). "Both" displays instructions.
- `-a <algorithm>`: Select the signature algorithm to use. Options include:
  - dilithium2, dilithium3, dilithium5
  - falcon512, falcon1024
  - sphincssha2128fsimple
- `-b <build_dir>`: Specify the libcoap build directory if not using the default.

## Troubleshooting

If you encounter issues:

1. Verify environment variables are set correctly:
   ```bash
   source ./setup-openssl-env.sh dtls13
   echo $OPENSSL_DIR
   echo $OPENSSL_CONF
   ```

2. Check if providers are loaded correctly:
   ```bash
   $OPENSSL_BIN_DIR/openssl list -providers
   ```

3. Ensure certificates were generated properly:
   ```bash
   $OPENSSL_BIN_DIR/openssl verify -provider oqsprovider -provider default \
       -CAfile ./certs/dilithium/dilithium5_root_cert.pem \
       ./certs/dilithium/dilithium5_entity_cert.pem
   ```

4. Verify libcoap was built with the correct OpenSSL path:
   ```bash
   ldd ./libcoap/build/bin/coap-client | grep ssl
   ```

## Certificate Generation with OQS Provider

The OpenSSL DTLS 1.3 installation includes the OQS provider and can be used for PQC certificate generation. Just source the environment script and use the dedicated script:

```bash
source ./openssl_env.sh dtls13
./certs/generate_certs.sh
```

The certificates will be generated in the `./certs` folder, organized according to their type. We currently support (in development): 

- **RSA**: Traditional RSA certificates (RSA_2048)
- **Dilithium**: Post-quantum signatures at different security levels (DILITHIUM_LEVEL2, DILITHIUM_LEVEL3, DILITHIUM_LEVEL5)
- **Falcon**: Post-quantum signatures at different security levels (FALCON_LEVEL1, FALCON_LEVEL5)



## Certificate Management

The framework includes a certificate management system that simplifies the use of different certificate types (RSA, Dilithium, Falcon) for benchmarking.

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

## Running Benchmarks

### Dependencies Installation

Install `perf` and `tshark`:

```bash
sudo apt install linux-tools-$(uname -r) tshark
```

Install the Python requirements:

```bash
conda create -n libcoap-bench python=3.10
conda activate libcoap-bench
pip install --no-cache-dir -r ./requirements_installation/requirements.txt
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
udp[_rasp]_conv_stats_[ALGORITHM]_n<N>[_s<S>][_<P>]_<SEC_MODE>[_<CERT_CONFIG>][_client-auth]_scenario<SCENARIO>
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

Once you have created CSV files with your benchmark results, you can generate plots:

```bash
cd libcoap-bench
python3 coap_benchmark_barplots.py <metric> <algorithms_list> <n> <rasp> <scenarios_list> [s] [p]
```

Where:

- `metric`: The metric to plot (e.g., 'duration', 'Wh').
- `algorithms_list`: Comma-separated list of algorithms.
- `n`: Number of clients used in the CSV files.
- `rasp`: Whether the server was running on Raspberry Pi (True or False).
- `scenarios_list`: Comma-separated list of scenarios (A, B, C).
- `s`: Optional observer duration.
- `p`: Optional parallelization mode.

Examples:

```bash
# Plot duration metric for multiple algorithms across different scenarios
python3 ./coap_benchmark_barplots.py 'duration' "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" 500 True A,B,C

# Plot energy consumption for specific test configuration
python3 ./coap_benchmark_barplots.py 'Wh' "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" 20 True A,C 30 "background"
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