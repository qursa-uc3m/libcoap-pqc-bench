# MQTT-SN Clients Patches

This directory contains patches to be applied to the [pq-mqtt-sn-clients](https://github.com/qursa-uc3m/pq-mqtt-sn-clients) repository for benchmark integration.

## Structure

Similar to `libcoap-patches/`, patches here modify the MQTT-SN clients to add:
- Timing measurement hooks
- Benchmark-specific output formats
- Integration with the benchmark pipeline

## Applying Patches

Patches are automatically applied by `scripts/install_mqttsn_clients.sh` when building the clients.

To manually apply a patch:

```bash
cd pq-mqtt-sn-clients
git apply ../benchmark/mqttsn-clients-patches/your_patch.patch
```

## Creating New Patches

1. Make changes to the pq-mqtt-sn-clients code
2. Generate a patch:
   ```bash
   cd pq-mqtt-sn-clients
   git diff > ../benchmark/mqttsn-clients-patches/descriptive_name.patch
   ```

## Current Patches

No patches required yet - the base clients already support runtime KEM selection via `MQTT_WOLFSSL_GROUPS` environment variable.
