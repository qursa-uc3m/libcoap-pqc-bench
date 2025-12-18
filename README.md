# Benchmarking Post-Quantum Cryptography in libcoap

This repository provides a tool for benchmarking Post-Quantum Cryptography (PQC) algorithms within the CoAP protocol. It integrates **liboqs** (for PQC algorithms), **wolfSSL** (for SSL/TLS support), and **libcoap** (for CoAP implementation) to enable performance testing across different security modes, algorithms, and network conditions.

The goal is to evaluate the impact of PQC on constrained IoT environments, measuring metrics like handshake duration, energy consumption, and data overhead.

## Benchmark Architecture

The tool runs in two main configurations. **Local Mode** runs the Client and Server on the same machine, which is useful for development. **Remote Mode** runs the Client on a PC and the Server on a constrained device (e.g., Raspberry Pi), optionally with a Network Emulation VM in between to simulate real-world conditions.

- The **`libcoap-bench/run_benchmarks.sh`** script is the main orchestrator. It manages the entire benchmark session, iterating through algorithms, security modes, and repetition counts, launching the client and server processes automatically.

- The **`libcoap-bench/coap_benchmark.sh`** script wraps the client execution, while **`libcoap-bench/coap_benchmark_server.sh`** wraps the server execution, ensuring the correct security configurations are applied on the target device.

## Quick Start (Local Mode)

Follow these steps to get a benchmark running on your local machine.

### 1. Install System Dependencies

```bash
sudo apt install tshark parallel linux-tools-$(uname -r)
```

### 2. Setup Python Environment

```bash
python3 -m venv .bench-env
source .bench-env/bin/activate
pip install -r ./libcoap-bench/requirements.txt
```

### 3. Install Libraries

First, install the PQC dependencies.

```bash
./scripts/install_liboqs_for_wolfssl.sh
```

Then build wolfssl.

```bash
./scripts/install_wolfssl.sh [--fork | --release [version]]
```

Options:

- `--fork`: Clone from dasobral/wolfssl-liboqs.git (default). This version fixes issues with DILITHIUM and FALCON certificates across different security levels.
- `--release [ver]`: Clone from wolfSSL/wolfssl.git with specified version (default: v5.7.6-stable)

Finally, install libcoap dependencies and build the library.

```bash
sudo apt-get install -y autoconf automake libtool make gcc
sudo apt-get install autoconf-archive libwolfssl-dev libcunit1-dev pkg-config
```

Run the installation script with the desired options:

```bash
./scripts/install_libcoap.sh [wolfssl] [--install-dir=PATH]
```

Options:

- `wolfssl`: Configure libcoap with WolfSSL as the underlying crypto library (otherwise uses OpenSSL)
- `--install-dir=PATH`: Specify a custom installation directory

### 4. Prepare Security Material (PSK)

Generate a Pre-Shared Key for the simplest security mode.

```bash
./pskeys/psk_manager.sh generate 256
./pskeys/psk_manager.sh activate $(ls pskeys/psk_256_*.key | head -1 | xargs basename)
```

### 5. Run a Benchmark

Run a simple test with 5 clients using PSK security and recommended PQC scenarios.

```bash
./libcoap-bench/run_benchmarks.sh -n 5 -security psk -scenarios A,C -y
```

**Note**: The `-scenarios A,C` flag runs only Scenarios A and C, which are recommended for PQC evaluation as they focus on cryptographic performance. Scenario B has artificial delays that mask PQC overhead.

**→ For detailed benchmark instructions, see [libcoap-bench/README.md](./libcoap-bench/README.md)**

---

## Security Modes & Setup

The framework supports three security modes.

### 1. Pre-Shared Keys (PSK)

This is the simplest secure mode, ideal for constrained devices. Use the `pskeys/` tools to generate and manage keys.

Command: `./pskeys/psk_manager.sh generate <bits>`

See [pskeys/README.md](./pskeys/README.md) for details.

### 2. Public Key Infrastructure (PKI)

This mode uses X.509 certificates and is where PQC algorithms (Dilithium, Falcon) are most critical. You must generate PQC certificates before running PKI benchmarks.

Setup:

```bash
# Generate all certificate chains (RSA, EC, Dilithium, Falcon)
./certs/generate_certs.sh
```

Use `./certs/config_certs.sh` to switch between active certificate types (e.g., `DILITHIUM_LEVEL3`). See [certs/README.md](./certs/README.md) for configuration details.

### 3. No Security (NoSec)

Baseline CoAP over UDP without encryption. No setup is required.

## Supported Algorithms

You can benchmark the following algorithms (controlled via `-groups` and `-signatures` flags in `run_benchmarks.sh`):

**Key Exchange (KEM) - via `-groups` flag:**

- Post-Quantum: `KYBER_LEVEL1`, `KYBER_LEVEL3`, `KYBER_LEVEL5`
- Hybrid PQ: `P256_KYBER_LEVEL1`, `P384_KYBER_LEVEL3`, `P521_KYBER_LEVEL5`
- Traditional: `P256`, `P384`, `P521`, `X25519`

**Signatures (PKI Certificates) - via `-signatures` flag:**

- Classical: `RSA_2048`, `EC_P256`, `EC_ED25519`
- Post-Quantum: `DILITHIUM_LEVEL2`, `DILITHIUM_LEVEL3`, `DILITHIUM_LEVEL5`, `FALCON_LEVEL1`, `FALCON_LEVEL5`

**Defaults:**

- KEM group: `KYBER_LEVEL3`
- Signature: `DILITHIUM_LEVEL3`
- Use `all` to test all available algorithms in each category

## Advanced Features

### Energy Monitoring

The framework can measure energy consumption of the handshake and data transfer. It supports FNIRSI USB meters (FNB48/58) for physical measurement and CodeCarbon (Intel RAPL) for local estimation.

See [libcoap-bench/energy/README.md](./libcoap-bench/energy/README.md) for setup instructions.

### Network Emulation

You can simulate real-world networks (Smart Home, Factory, etc.) using a VM bridge with QEMU/KVM and NetEm.

See [network_emulation/README.md](./network_emulation/README.md) for details.

## System Setup Details

### RAPL Permissions (Energy Monitoring)

CodeCarbon uses Intel RAPL to measure CPU energy. By default, this is root-only. Grant permissions to run benchmarks as a standard user:

```bash
# Temporary (resets on reboot)
sudo chmod -R a+r /sys/devices/virtual/powercap/

# Permanent fix (udev rule)
sudo bash -c 'cat > /etc/udev/rules.d/99-rapl.rules << EOF
SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod -R a+r /sys/devices/virtual/powercap/"
EOF'
sudo udevadm control --reload-rules
```

### Perf Tool (CPU Cycles)

Used for precise CPU cycle counting.

```bash
sudo apt install linux-tools-generic linux-tools-$(uname -r)
```

**Ubuntu 24.04 with kernel 6.14 bug**: The `linux-tools-6.14.x` packages are missing the `perf` binary ([Bug #2117159](https://bugs.launchpad.net/ubuntu/+source/linux-hwe-6.14/+bug/2117159)). Apply this workaround:

```bash
# Find a working perf and symlink it
WORKING_PERF=$(ls /usr/lib/linux-tools-*/perf 2>/dev/null | head -1)
sudo ln -sf "$WORKING_PERF" /usr/lib/linux-tools/$(uname -r)/perf
 
# Verify it works
perf --version
```

The perf command can be configured in `config.env`:

- `PERF_CMD`: Command for local mode (default: `perf`)
- `PERF_CMD_RPI`: Command for Raspberry Pi (default: `perf_5.10`)

### Packet Capture

Allow non-root packet capture for `tshark`:

```bash
sudo usermod -aG wireshark $USER
# Log out and back in to apply
```

## Troubleshooting

If a benchmark crashes, you can cleanup old processes with:

```bash
pgrep -f 'libcoap' | xargs -r kill -9
```

To analyze PQC traffic using the OQS-enabled Wireshark docker image:

```bash
sudo ./scripts/oqs_wireshark.sh
```

and filter by

```text
udp.port==5684 || udp.port==5683
```
