# MQTT-SN Gateway Configuration Templates

This directory contains configuration templates for the MQTT-SN gateway used in PQC benchmarking.

## Files

| File | Description |
|------|-------------|
| `gateway.conf` | Main gateway configuration with dynamic certificate placeholders |
| `clients.conf` | Client configuration template |

## Certificate Handling

The `gateway.conf` contains placeholder paths for DTLS certificates:

- `DtlsCertsKey` - Server certificate path
- `DtlsPrivKey` - Server private key path  
- `DtlsCACertFile` - CA certificate for client verification

These are **dynamically updated** by `mqttsn_benchmark_server.sh` at runtime based on the `-cert-config` parameter, mirroring how CoAP server handles certificates via `config_certs.sh`.

## Installation

The `scripts/install_paho_mqttsn_gateway.sh` copies these templates to the gateway's bin directory during installation.

## Usage

```bash
# Start gateway with specific certificate config
./benchmark/mqttsn_benchmark_server.sh -sec-mode pki -cert-config DILITHIUM_LEVEL3

# Start in nosec mode (no certificates)
./benchmark/mqttsn_benchmark_server.sh -sec-mode nosec
```
