# Network Emulation Scenarios for CoAP Testing

## System Configuration

- **VM Interface**: `ens3` (Yours might be different)
- **Original Configuration**: fq_codel queue discipline

## Scenario Parameters Table

| Use Case       | Delay (ms) | Loss (%) | Jitter (ms) | Rate (Mbps) |
|----------------|------------|----------|-------------|-------------|
| Fiducial       | 0          | 0        | 0           | Unlimited   |
| Smart Factory  | 20         | 1.0      | 5           | 50          |
| Smart Home     | 5          | 0.1      | 1           | 10          |
| Pub. Transport | 50         | 2.0      | 10          | 5           |

## Scenario Implementation

### 0. Fiducial Scenario (No Network Emulation)

This is the baseline configuration without any network emulation.
Simply ensure no netem rules are applied and use the original fq_codel configuration.

```bash
# Clear any existing emulation
sudo tc qdisc del dev ens3 root

# Apply original fq_codel configuration
sudo tc qdisc add dev ens3 root fq_codel limit 10240 flows 1024 quantum 1514 target 5ms interval 100ms memory_limit 32Mb ecn drop_batch 64
```

### 1. Smart Factory Scenario

```bash
# Clear existing queue discipline
sudo tc qdisc del dev ens3 root

# Apply network emulation
sudo tc qdisc add dev ens3 root netem delay 20ms 5ms distribution normal loss 1% rate 50Mbit
```

### 2. Smart Home Scenario

```bash
# Clear existing queue discipline
sudo tc qdisc del dev ens3 root

# Apply network emulation
sudo tc qdisc add dev ens3 root netem delay 5ms 1ms distribution normal loss 0.1% rate 10Mbit
```

### 3. Public Transport Scenario

```bash
# Clear existing queue discipline
sudo tc qdisc del dev ens3 root

# Apply network emulation
sudo tc qdisc add dev ens3 root netem delay 50ms 10ms distribution normal loss 2.0% rate 5Mbit
```

## Verification

To check current network emulation settings:

```bash
sudo tc qdisc show dev ens3
```

## Reset to Original Configuration

To restore the system to its original state after any test:

```bash
# Step 1: Remove all network emulation
sudo tc qdisc del dev ens3 root

# Step 2: Restore original fq_codel configuration
sudo tc qdisc add dev ens3 root fq_codel limit 10240 flows 1024 quantum 1514 target 5ms interval 100ms memory_limit 32Mb ecn drop_batch 64
```

## Workflow Example

1. Start with fiducial scenario (baseline testing)
2. Apply Smart Factory scenario
   - Run CoAP benchmarks
   - Record results
3. Reset to original configuration
4. Apply Smart Home scenario
   - Run CoAP benchmarks
   - Record results
5. Reset to original configuration
6. Apply Public Transport scenario
   - Run CoAP benchmarks
   - Record results
7. Reset to original configuration

## Important Notes

- All commands require `sudo` privileges
- Always clear existing qdisc before applying new emulation
- The fiducial scenario uses the original fq_codel configuration
- Network emulation affects all traffic on the ens3 interface
- After each test scenario, reset to original configuration before applying the next
- Packet loss and delay significantly impact connection stability