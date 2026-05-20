# MQTT-SN Clients Patches

This directory contains patched source files for the [pq-mqtt-sn-clients](https://github.com/qursa-uc3m/pq-mqtt-sn-clients) repository, adding benchmark timing measurement hooks for integration with the PQC benchmark pipeline.

## Patched Files

### [sn-pub.c](sn-pub.c)
Publisher client with timing measurement hooks:
- `get_current_time_ns()` - captures nanosecond-precision timestamps
- `append_time_to_file()` - writes timing data to `$BENCH_DATA_DIR/time_output.txt`
- Timing starts at `sn_test()` entry, stops at exit

### [sn-sub.c](sn-sub.c)
Subscriber client with identical timing hooks for consistent measurement across roles.

## Usage

Copy these files to replace the originals in `pq-mqtt-sn-clients/src/` before building:

```bash
# Copy patched files
cp benchmark/mqttsn-clients-patches/*.c pq-mqtt-sn-clients/src/

# Rebuild clients
cd pq-mqtt-sn-clients/build
make
```

Or run the install script which applies patches automatically:
```bash
./scripts/install_mqttsn_clients.sh
```

## Environment Variables

The timing hooks respect:
- `BENCH_DATA_DIR` - output directory for timing data (primary)
- `REPO_ROOT` - fallback path to `benchmark/data/current/`

## Comparison with CoAP Patches

These patches mirror the approach in `libcoap-patches/coap-client.c`:
- Same timing function signatures
- Same output format (seconds with 3 decimal places)
- Compatible with `bench-data-manager.py` processing
