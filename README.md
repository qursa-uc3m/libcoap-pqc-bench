# Benchmarking Post-Quantum Cryptography in IoT Protocols

This repository provides a tool for benchmarking Post-Quantum Cryptography (PQC) algorithms within IoT protocols. It supports **CoAP** and **MQTT-SN** protocols, integrating **liboqs** (for PQC algorithms), **wolfSSL** (for TLS/DTLS support), **libcoap** (for CoAP), and **paho-mqtt-sn-gateway** + **wolfMQTT** (for MQTT-SN).

The goal is to evaluate the impact of PQC on constrained IoT environments, measuring metrics like handshake duration, energy consumption, and data overhead.

## Supported Protocols

| Protocol | Transport | Implementation | Security Modes |
|----------|-----------|----------------|----------------|
| **CoAP** | DTLS 1.3 | libcoap + wolfSSL | PKI, PSK, nosec |
| **MQTT-SN** | DTLS 1.3 | paho-gateway + wolfMQTT | PKI, nosec |

## Benchmark Architecture

The tool runs in two main configurations. **Local Mode** runs the Client and Server on the same machine, which is useful for development. **Remote Mode** runs the Client on a PC and the Server on a constrained device (e.g., Raspberry Pi), optionally with a Network Emulation VM in between to simulate real-world conditions.

- The **`benchmark/run_benchmarks.sh`** script is the main orchestrator. It manages the entire benchmark session, iterating through protocols, algorithms, security modes, and repetition counts.

- For **CoAP**: `benchmark/coap_benchmark.sh` and `benchmark/coap_benchmark_server.sh` wrap client/server execution.

- For **MQTT-SN**: `benchmark/mqttsn_benchmark.sh` and `benchmark/mqttsn_benchmark_server.sh` wrap client/gateway execution.

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
pip install -r ./benchmark/requirements.txt
```

### 3. Install Libraries (CoAP or MQTT-SN)

First, install the PQC dependencies.

```bash
./scripts/install_liboqs_for_wolfssl.sh
```

Then build wolfSSL.

```bash
./scripts/install_wolfssl.sh [--fork | --release [version]]
```

Options:

- `--fork`: Clone from dasobral/wolfssl-liboqs.git (default). This version fixes issues with DILITHIUM and FALCON certificates across different security levels.
- `--release [ver]`: Clone from wolfSSL/wolfssl.git with specified version (default: v5.7.6-stable)

#### For CoAP Protocol

Install libcoap dependencies and build the library.

```bash
sudo apt-get install -y autoconf automake libtool make gcc
sudo apt-get install autoconf-archive libwolfssl-dev libcunit1-dev pkg-config
./scripts/install_libcoap.sh [wolfssl] [--install-dir=PATH]
```

Options:

- `wolfssl`: Configure libcoap with WolfSSL as the underlying crypto library (otherwise uses OpenSSL)
- `--install-dir=PATH`: Specify a custom installation directory

#### For MQTT-SN Protocol

Install wolfMQTT, Mosquitto broker, paho-mqttsn-gateway, and MQTT-SN clients:

```bash
./scripts/install_wolfmqtt.sh
./scripts/install_mosquitto.sh
./scripts/install_paho_mqttsn_gateway.sh
./scripts/install_mqttsn_clients.sh
```

### 4. Prepare Security Material

#### For PSK Mode (CoAP only)

Generate a Pre-Shared Key for the simplest security mode.

```bash
./pskeys/psk_manager.sh generate 256
./pskeys/psk_manager.sh activate $(ls pskeys/psk_256_*.key | head -1 | xargs basename)
```

#### For PKI Mode (CoAP and MQTT-SN)

Generate certificates (required for MQTT-SN PKI mode):

```bash
cd certs
./generate_certs.sh all
cd ..
```

### 5. Run a Benchmark

#### CoAP Benchmark

Run a simple CoAP test with 5 clients using PSK security and recommended PQC scenarios.

```bash
./benchmark/run_benchmarks.sh -protocol coap -n 5 -security psk -scenarios A,C -y
```

#### MQTT-SN Benchmark

Run a simple MQTT-SN test with 5 clients using PKI security.

```bash
./benchmark/run_benchmarks.sh -protocol mqttsn -n 5 -security pki -scenarios A,C -y
```

**Note**: The `-scenarios A,C` flag runs only Scenarios A and C, which are recommended for PQC evaluation as they focus on cryptographic performance. Scenario B has artificial delays that mask PQC overhead.

**→ For detailed benchmark instructions, see [benchmark/README.md](./benchmark/README.md)**

---

## Security Modes & Setup

The framework supports three security modes. Note that MQTT-SN only supports PKI and NoSec modes.

| Security Mode | CoAP | MQTT-SN | Description |
|---------------|------|---------|-------------|
| **PSK** | ✅ | ❌ | Pre-Shared Keys - simplest secure mode |
| **PKI** | ✅ | ✅ | X.509 certificates - PQC algorithms |
| **NoSec** | ✅ | ✅ | No encryption - baseline |

### 1. Pre-Shared Keys (PSK) - CoAP Only

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

Baseline CoAP/MQTT-SN over UDP without encryption. No setup is required.

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

See [benchmark/energy/README.md](./benchmark/energy/README.md) for setup instructions.

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
# For CoAP
pgrep -f 'libcoap' | xargs -r kill -9

# For MQTT-SN
pgrep -f 'MQTT-SNGateway|mqttsn_client' | xargs -r kill -9
pgrep -f 'mosquitto' | xargs -r kill -9
```

To analyze PQC traffic using the OQS-enabled Wireshark docker image:

```bash
sudo ./scripts/oqs_wireshark.sh
```
and filter by

```text
udp.port==5684 || udp.port==5683
```
