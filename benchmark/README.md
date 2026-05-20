# PQC Protocol Benchmark Suite

Step-by-step guide for running post-quantum cryptography benchmarks for **CoAP** and **MQTT-SN** protocols.

## Supported Protocols

| Protocol | Transport | Implementation | Security Modes |
|----------|-----------|----------------|----------------|
| **CoAP** | DTLS 1.3 | libcoap + wolfSSL | PKI, PSK, nosec |
| **MQTT-SN** | DTLS 1.3 | paho-gateway + wolfMQTT | PKI, nosec |

## Prerequisites

- Python environment: `source .bench-env/bin/activate`
- For CoAP: libcoap built with WolfSSL
- For MQTT-SN: Gateway and clients built with wolfSSL/wolfMQTT
- PSK key generated and activated (CoAP only - see main README)

## Step 1: Setup Environment

```bash
cd /path/to/libcoap-pqc-bench
source .bench-env/bin/activate
```

### Install Protocol Dependencies

**For CoAP:**
```bash
./scripts/install_liboqs_for_wolfssl.sh
./scripts/install_wolfssl.sh
./scripts/install_libcoap.sh wolfssl
```

**For MQTT-SN:**
```bash
./scripts/install_liboqs_for_wolfssl.sh
./scripts/install_wolfssl.sh
./scripts/install_wolfmqtt.sh
./scripts/install_mosquitto.sh
./scripts/install_paho_mqttsn_gateway.sh dtls
./scripts/install_mqttsn_clients.sh
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

### CoAP Benchmarks

```bash
# Simple benchmark (25 clients, PSK mode, scenarios A and C only - recommended for PQC)
./benchmark/run_benchmarks.sh -protocol coap -n 25 -security psk -scenarios A,C -y

# Observer mode (60 seconds)
./benchmark/run_benchmarks.sh -protocol coap -n 25 -s 60 -security psk -resources example_data -y

# Parallel execution with energy monitoring (PQC-focused scenarios)
./benchmark/run_benchmarks.sh -protocol coap -n 25 -parallelization parallel -energy -security pki,psk,nosec -scenarios A,C -iterations 5 -y
```

### MQTT-SN Benchmarks

> **Scenario mapping**: MQTT-SN uses two scenarios that mirror the CoAP
> connection-establishment workload (CoAP scenario A): `pub` measures a
> publisher's full session (DTLS handshake + CONNECT + REGISTER + PUBLISH
> + DISCONNECT); `sub` measures a subscriber's full session (DTLS handshake
> + CONNECT + SUBSCRIBE + DISCONNECT). Async-reply and observe variants
> (CoAP B/C) are CoAP-specific and have no MQTT-SN equivalent.
>
> **Gateway lifecycle**: the gateway is started/stopped automatically by
> `mqttsn_benchmark.sh` per run, mirroring how `coap_benchmark.sh` manages
> the CoAP server. A local Mosquitto MQTT broker must already be running
> on `127.0.0.1:1883` (started by `scripts/install_mosquitto.sh` or by
> hand). Override the broker with `MQTTSN_BROKER_HOST` if remote.

```bash
# Simple benchmark (25 clients, PKI mode, both pub and sub)
./benchmark/run_benchmarks.sh -protocol mqttsn -n 25 -security pki -scenarios pub,sub -y

# With specific KEM and signature algorithms (default scenarios = pub,sub)
./benchmark/run_benchmarks.sh -protocol mqttsn -n 25 -groups KYBER_LEVEL3 -signatures DILITHIUM_LEVEL3 -security pki -y

# Parallel execution with energy monitoring
./benchmark/run_benchmarks.sh -protocol mqttsn -n 25 -parallelization parallel -energy -security pki,nosec -scenarios pub,sub -iterations 5 -y
```

### Complete Benchmark Suite

Run for each network condition (using recommended PQC scenarios):

```bash
# 1. FIDUCIAL NETWORK - CoAP
sudo ./network_emulation/net_config.sh set fiducial
./benchmark/run_benchmarks.sh -protocol coap -n 25 -groups all -signatures all -parallelization parallel -security "pki,psk,nosec" -scenarios A,C -iterations 5 -energy -y

# 1. FIDUCIAL NETWORK - MQTT-SN
./benchmark/run_benchmarks.sh -protocol mqttsn -n 25 -groups all -signatures all -parallelization parallel -security "pki,nosec" -scenarios pub,sub -iterations 5 -energy -y

# 2. SMART HOME NETWORK - CoAP
sudo ./network_emulation/net_config.sh set smart-home
./benchmark/run_benchmarks.sh -protocol coap -n 25 -groups all -signatures all -parallelization parallel -security "pki,psk,nosec" -scenarios A,C -iterations 5 -energy -y

# 2. SMART HOME NETWORK - MQTT-SN
./benchmark/run_benchmarks.sh -protocol mqttsn -n 25 -groups all -signatures all -parallelization parallel -security "pki,nosec" -scenarios pub,sub -iterations 5 -energy -y

# Reset network after each set
sudo ./network_emulation/net_config.sh reset
```

## Step 4: Process Results

### Aggregate Data from Iterations

```bash
cd benchmark/data

# Aggregate specific session (auto-detects iterations)
python3 ../bench-data-manager.py aggregate --data-dir . --session-id local_1219_fiducial_x7

# Or specify iterations explicitly
python3 ../bench-data-manager.py aggregate --data-dir . --session-id local_1219_fiducial_x7 --iterations 5
```

This creates `aggregated/<SESSION_ID>/` with aggregated metrics.

## Step 5: Generate Plots

```bash
cd libcoap-plots

# Single session scatter plot
python3 bench-data-plots.py "duration" 1 --scatter --scenarios A \
    --data-dir ../benchmark/data --custom-suffix "local_1219_fiducial_x7" --p "parallel"

# Bar plot comparing scenarios
python3 bench-data-plots.py "Energy (Wh)" 1 --barplot --scenarios A,C \
    --data-dir ../benchmark/data --custom-suffix "local_1219_fiducial_x7"

# Or use the wrapper script
./plots_wrapper.sh "duration,Energy (Wh)" scatter A --session local_1219_fiducial_x7
```

See [libcoap-plots/README.md](../libcoap-plots/README.md) for all visualization options.

## Data Organization

The benchmark creates a hierarchical folder structure with clear session identification:

```text
benchmark/data/
├── current/                  # Temporary working directory
├── raw/                      # Raw iteration data (organized by session)
│   └── local_1219_fiducial_x7/     # Session folder (local_MMDD_NETWORK_RANDOM)
│       ├── session_metadata.txt    # Session metadata (parameters, times, etc.)
│       ├── iter_1/                 # Iteration 1 data
│       │   ├── *.csv               # Benchmark results
│       │   └── energy-data/        # Energy measurements (if enabled)
│       ├── iter_2/                 # Iteration 2 data
│       └── ...
├── aggregated/               # Aggregated statistics (one folder per session)
│   └── local_1219_fiducial_x7/
│       └── *.csv             # Aggregated metrics across iterations
├── plots/                    # Generated plots
│   └── local_1219_fiducial_x7/
├── summaries/                # Session summaries
│   └── summary_local_1219_fiducial_x7.txt
└── sessions.txt              # Session tracking log
```

### Session ID Format

The session ID follows the pattern: `{prefix}_{MMDD}_{network}_{random}`

- **prefix**: `local` or `rasp` (based on server mode)
- **MMDD**: Month and day of the benchmark
- **network**: Network condition (fiducial, smart-home, smart-factory, public-transport)
- **random**: 2-character random string for uniqueness

Example: `local_1219_fiducial_x7` = Local server, December 19, fiducial network

## Troubleshooting

```bash
# Check server connectivity
ping 192.168.0.157

# Verify VM is running and configured
sudo ./network_emulation/net_config.sh test

# Kill zombie CoAP processes
pgrep -f 'libcoap' | xargs -r kill -9

# Check energy monitor
python3 benchmark/energy_monitor.py --backend codecarbon --list-devices
```

## Command Reference

```bash
./benchmark/run_benchmarks.sh -n NUM_CLIENTS [OPTIONS]
```

```text
Required:
  -n NUM_CLIENTS        Number of clients

Options:
  -groups GROUPS        Comma-separated KEM groups or 'all' (default: KYBER_LEVEL3)
                        Available: KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,
                                   P256_KYBER_LEVEL1,P384_KYBER_LEVEL3,P521_KYBER_LEVEL5,
                                   P256,P384,P521,X25519
  -signatures SIGS      Comma-separated signatures or 'all' (default: DILITHIUM_LEVEL3)
                        Available: RSA_2048,EC_P256,EC_ED25519,DILITHIUM_LEVEL2,
                                   DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,FALCON_LEVEL1,FALCON_LEVEL5
  -security MODES       Comma-separated: pki,psk,nosec
  -resources RES        time (scenario A/C), async (scenario B), example_data
  -s TIME               Observer mode duration (seconds)
  -iterations N         Repeat each test N times
  -parallelization MODE background or parallel
  -energy               Enable energy measurements
  -rasp                 Server runs on Raspberry Pi
  -network CONDITION    Network condition label (auto-detected from net_config.sh)
                        Values: fiducial, smart-home, smart-factory, public-transport
  -cert-filter PATTERN  [DEPRECATED] Use -signatures instead
  -client-auth yes|no   Enable client certificate auth (PKI mode)
  -y                    Skip confirmation prompts
  -v                    Verbose output
```

## Benchmark Scenarios

Both protocols use the `-scenarios` flag to select which tests to run, but with protocol-specific options.

### CoAP Scenarios (A, B, C)

#### Scenario A: Synchronous Request-Response (Confirmable)

- **Resource**: `time`
- **Message Type**: Confirmable (CON)
- **Pattern**: Client sends GET request → Server responds immediately
- **Use Case**: Reliable communication with acknowledgments
- **Recommended for**: **PQC handshake overhead measurement**

#### Scenario B: Asynchronous/Observer Mode

- **Resource**: `async` or `example_data`
- **Message Type**: Confirmable (CON)
- **Pattern**:
  - `async`: Server-side delayed response (default **4 seconds** delay, configurable via query e.g., `async?2`)
  - `example_data`: CoAP Observe pattern (client subscribes, server pushes updates)
- **Observer Flag**: `-s TIME` sets observation duration in seconds
- **Use Case**: Testing delayed responses and publish-subscribe patterns

⚠️ **Important Note for PQC Benchmarking**: The `async` resource has a built-in server-side delay (default 4 seconds) and limited thread pool (3 threads), which causes significant queueing with multiple clients. This delay dominates the measurements and masks PQC cryptographic overhead. **For PQC evaluation, use Scenarios A and C instead.** Scenario B is primarily useful for testing server load and async protocol behavior, not for comparing cryptographic performance.

#### Scenario C: Synchronous Request-Response (Non-Confirmable)

- **Resource**: `time`
- **Message Type**: Non-confirmable (NON)
- **Pattern**: Client sends GET request → Server responds immediately (no ACK)
- **Use Case**: Best-effort communication without acknowledgments
- **Recommended for**: **PQC session maintenance and throughput measurement**

---

### MQTT-SN Scenarios (pub, sub)

#### Scenario pub: Publisher

- **Client**: `sn-pub`
- **Pattern**: Client connects → Registers topic → Publishes message → Disconnects
- **Use Case**: Testing publish path including DTLS handshake with PQC
- **Recommended for**: **PQC handshake and publish overhead measurement**

#### Scenario sub: Subscriber

- **Client**: `sn-sub`
- **Pattern**: Client connects → Registers topic → Subscribes → Disconnects
- **Use Case**: Testing subscribe path including DTLS handshake with PQC
- **Recommended for**: **PQC handshake and subscribe overhead measurement**

---

### Selecting Scenarios

Use the `-scenarios` flag with protocol-appropriate values:

**CoAP Examples:**
```bash
# Run only Scenarios A and C (recommended for PQC evaluation)
./run_benchmarks.sh -protocol coap -n 10 -security pki -scenarios A,C

# Run only Scenario A
./run_benchmarks.sh -protocol coap -n 25 -security pki -scenarios A

# Run all CoAP scenarios (default)
./run_benchmarks.sh -protocol coap -n 10 -security pki -scenarios A,B,C
```

**MQTT-SN Examples:**
```bash
# Run publisher scenario only (default for MQTT-SN)
./run_benchmarks.sh -protocol mqttsn -n 25 -security pki -scenarios pub

# Run both publisher and subscriber scenarios
./run_benchmarks.sh -protocol mqttsn -n 25 -security pki -scenarios pub,sub

# Run subscriber scenario only
./run_benchmarks.sh -protocol mqttsn -n 25 -security pki -scenarios sub
```

**Recommendation for PQC Benchmarking**:
- **CoAP**: Use `-scenarios A,C` to focus on meaningful cryptographic performance metrics
- **MQTT-SN**: Use `-scenarios pub` for typical IoT publish workflows

## Related Documentation

- [Energy monitoring](energy/README.md) - FNIRSI and CodeCarbon backends
- [Network emulation](../network_emulation/README.md) - VM setup and network scenarios
- [Visualization](../libcoap-plots/README.md) - Plot types and metrics
- [Certificates](../certs/README.md) - PKI certificate management
