# Network Emulation for libcoap PQC Benchmarking

This guide provides instructions for setting up and using network emulation capabilities to test CoAP with Post-Quantum Cryptography (PQC) under various network conditions.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [VM Setup](#vm-setup)
  - [Automated VM Setup](#automated-vm-setup)
  - [Manual VM Setup](#manual-vm-setup)
- [Network Configuration](#network-configuration)
- [Network Scenarios](#network-scenarios)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Overview

Network emulation allows testing applications under various network conditions such as latency, packet loss, and bandwidth limitations. This is essential for understanding how PQC algorithms behave in real-world networking scenarios.

The framework uses:
- **KVM/QEMU** for virtualization
- **NetEm** (Network Emulator) for comprehensive network condition simulation
- **Bridge networking** with TAP devices for traffic interception
- **Automated scripts** for easy setup and configuration

### Architecture

```
Client Host ──┐
              ├─── Bridge (br0) ───── VM (NetEm) ───── Server (RPi)
SSH Traffic ──┘                      │
                                   Network
                                 Emulation
```

The VM sits between your client and server, intercepting and modifying network traffic according to configured scenarios. SSH traffic bypasses the VM for direct access.

## Quick Start

For users who want to get started immediately:

1. **First-time setup** (installs new VM):
   ```bash
   sudo ./setup_vm.sh --install --name netem-vm
   ```

2. **Configure network routing**:
   ```bash
   sudo ./udp_config.sh
   ```

3. **Apply a network scenario**:
   ```bash
   sudo ./net_config.sh set smart-factory
   ```

4. **Check current configuration**:
   ```bash
   sudo ./net_config.sh show
   ```

5. **Reset when done**:
   ```bash
   sudo ./net_config.sh reset
   ```

## VM Setup

### Automated VM Setup

The `setup_vm.sh` script handles VM creation and management:

#### First-time Installation
```bash
# Create and install a new VM
sudo ./setup_vm.sh --install --name <vm_name>

# You'll be prompted for:
# - OS ISO file path (full path required)
# - Installation confirmation
```

#### Starting an Existing VM
```bash
# Launch previously created VM
sudo ./setup_vm.sh --name <vm_name>

# Or simply run without parameters and enter name when prompted
sudo ./setup_vm.sh
```

#### VM Specifications
- **Disk**: 10GB QCOW2 image
- **RAM**: 2GB
- **CPU**: 2 cores
- **Network**: Bridged networking with TAP device
- **Console**: Serial console (headless mode)

#### Important Installation Notes

When installing Ubuntu-based distributions, you'll need to modify GRUB parameters:

1. During boot, press `e` to edit boot parameters at the GRUB menu
2. Add `console=ttyS0 text` at the end of the linux line
3. Example: `linux /casper/vmlinuz ... console=ttyS0 text`

This ensures proper serial console functionality for headless operation.

### Manual VM Setup

For advanced users requiring customization, here are detailed manual setup instructions:

#### On Debian 11

1. **Install required packages**:
   ```bash
   sudo apt update
   sudo apt install qemu-kvm virtinst
   ```

2. **Check hardware virtualization support**:
   ```bash
   egrep -c '(vmx|svm)' /proc/cpuinfo
   sudo apt install -y cpu-checker
   kvm-ok
   ```

3. **Install KVM and related tools**:
   ```bash
   sudo apt install libvirt-daemon-system libvirt-clients bridge-utils
   sudo systemctl start libvirtd
   sudo systemctl enable libvirtd
   ```

4. **Set up networking**:
   ```bash
   sudo virsh net-start default
   sudo virsh net-autostart default
   ip a show virbr0  # Verify bridge exists
   ```

5. **Create a VM**:
   ```bash
   sudo virt-install \
     --name netem-vm \
     --memory 2048 \
     --vcpus 2 \
     --disk path=/var/lib/libvirt/images/netem-vm.img,size=10 \
     --os-type linux \
     --os-variant debian10 \
     --network bridge=virbr0 \
     --graphics none \
     --console pty,target_type=serial \
     --location 'http://deb.debian.org/debian/dists/bullseye/main/installer-amd64/' \
     --extra-args 'console=ttyS0,115200n8 serial'
   ```

#### On Ubuntu 20.04

1. **Install required packages**:
   ```bash
   sudo apt update
   sudo apt install uml-utilities qemu-utils bridge-utils
   ```

2. **Create a disk image**:
   ```bash
   qemu-img create -f qcow2 /var/lib/libvirt/images/netem-vm.qcow2 10G
   ```

3. **Set up TAP device and bridge**:
   ```bash
   sudo tunctl -t tap0 -u `whoami`
   sudo ip link set tap0 up
   sudo brctl addif virbr0 tap0
   ```

4. **Launch VM with QEMU/KVM**:
   ```bash
   sudo qemu-system-x86_64 -enable-kvm \
     -m 2048 \
     -smp cpus=2 \
     -hda /var/lib/libvirt/images/netem-vm.qcow2 \
     -cdrom ~/Downloads/debian-10.13.0-amd64-netinst.iso \
     -net nic,model=virtio -net tap,ifname=tap0,script=no,downscript=no
   ```

## Network Configuration

### Automated Configuration

The `udp_config.sh` script configures network routing automatically:

```bash
sudo ./udp_config.sh
```

This script:
- Configures the VM for packet forwarding
- Sets up the client host to route traffic through the VM
- Configures direct SSH routing (bypasses VM)
- Applies necessary iptables rules for UDP ports 5683 and 5684

#### Network Parameters
The script uses these default IP addresses (modify in the script if different):
- **Client Host**: 192.168.0.228
- **VM**: 192.168.0.172
- **Server (RPi)**: 192.168.0.157

### Manual Configuration

For custom setups or when you need more control over the configuration process:

#### VM Configuration

1. **Inside the VM, enable IP forwarding**:
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   sudo sh -c 'echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf'
   ```

2. **Configure iptables for forwarding**:
   ```bash
   sudo iptables -t nat -A POSTROUTING -o ens3 -j MASQUERADE
   sudo iptables -A FORWARD -i ens3 -o ens3 -j ACCEPT
   ```

#### Client Host Configuration

1. **Enable IP forwarding on the host**:
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   ```

2. **Route traffic through VM**:
   ```bash
   # Make the server reachable through the VM
   sudo ip route add <server_ip>/32 via <vm_ip> dev br0
   
   # Configure iptables for the target ports
   sudo iptables -t nat -A PREROUTING -p udp --dport 5683 -j DNAT --to-destination <vm_ip>
   sudo iptables -t nat -A PREROUTING -p udp --dport 5684 -j DNAT --to-destination <vm_ip>
   ```

3. **For CoAP with DTLS (port 5684), filter ICMP packets**:
   ```bash
   # Filter out ICMP packets appearing at the end of the DTLS handshake
   sudo iptables -A OUTPUT -p icmp --icmp-type port-unreachable -j DROP
   ```

4. **Configure direct SSH routing (optional but recommended)**:
   ```bash
   # Create SSH routing table if it doesn't exist
   if ! grep -q "ssh-route" /etc/iproute2/rt_tables; then
       echo "200 ssh-route" >> /etc/iproute2/rt_tables
   fi
   
   # Add rule and route for direct SSH access
   sudo ip rule add to <server_ip> dport 22 table ssh-route
   sudo ip route add <server_ip> dev br0 table ssh-route
   ```

#### Example with Actual IPs

Using the default configuration from `udp_config.sh`:

```bash
# VM configuration (run inside VM)
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -o ens3 -j MASQUERADE
sudo iptables -A FORWARD -i ens3 -o ens3 -j ACCEPT

# Client host configuration
sudo sysctl -w net.ipv4.ip_forward=1
sudo ip route add 192.168.0.157/32 via 192.168.0.172 dev br0
sudo iptables -t nat -A PREROUTING -p udp --dport 5683 -j DNAT --to-destination 192.168.0.172
sudo iptables -t nat -A PREROUTING -p udp --dport 5684 -j DNAT --to-destination 192.168.0.172
sudo iptables -A OUTPUT -p icmp --icmp-type port-unreachable -j DROP

# SSH direct routing
echo "200 ssh-route" >> /etc/iproute2/rt_tables
sudo ip rule add to 192.168.0.157 dport 22 table ssh-route
sudo ip route add 192.168.0.157 dev br0 table ssh-route
```

## Network Scenarios

### Available Scenarios

The framework provides four predefined scenarios based on real-world use cases:

| Scenario         | Delay    | Loss  | Jitter | Rate     | Use Case Description |
|------------------|----------|-------|--------|----------|---------------------|
| Fiducial         | 0ms      | 0%    | 0ms    | Unlimited| Baseline (no emulation) |
| Smart Factory    | 20ms     | 1.0%  | 5ms    | 50 Mbps  | Industrial IoT environment |
| Smart Home       | 5ms      | 0.1%  | 1ms    | 10 Mbps  | Residential IoT network |
| Public Transport | 50ms     | 2.0%  | 10ms   | 5 Mbps   | Mobile/cellular network |

### Scenario Management

Use the `net_config.sh` script for scenario management:

#### Apply a Scenario
```bash
sudo ./net_config.sh set <scenario_name>

# Examples:
sudo ./net_config.sh set smart-factory
sudo ./net_config.sh set smart-home
sudo ./net_config.sh set public-transport
sudo ./net_config.sh set fiducial
```

#### Check Current Configuration
```bash
sudo ./net_config.sh show
```

#### Reset to Baseline
```bash
sudo ./net_config.sh reset
```

#### Test VM Connection
```bash
sudo ./net_config.sh test
```

### Advanced Configuration Options

The script supports customization:

```bash
# Custom VM connection
sudo ./net_config.sh --user myuser --host 192.168.1.100 set smart-factory

# Verbose output
sudo ./net_config.sh --verbose show

# Custom network interface
sudo ./net_config.sh --interface eth0 set smart-home
```

## Usage Examples

### Complete Testing Workflow

```bash
# 1. Initial setup (first time only)
sudo ./setup_vm.sh --install --name netem-vm
sudo ./udp_config.sh

# 2. Baseline testing
sudo ./net_config.sh set fiducial
# Run your CoAP tests here
# Record baseline results

# 3. Smart Factory testing
sudo ./net_config.sh set smart-factory
sudo ./net_config.sh show  # Verify configuration
# Run your CoAP tests here
# Record results

# 4. Reset and continue with other scenarios
sudo ./net_config.sh reset
sudo ./net_config.sh set smart-home
# ... continue testing

# 5. Final cleanup
sudo ./net_config.sh reset
```

### Daily Usage (After Initial Setup)

```bash
# Start VM
sudo ./setup_vm.sh --name netem-vm

# Apply desired scenario
sudo ./net_config.sh set public-transport

# Run tests...

# Clean up
sudo ./net_config.sh reset
```

## Troubleshooting

### Common Issues

#### SSH Connection Failed
```bash
# Test connectivity
sudo ./net_config.sh test

# Check if VM is running
sudo ./setup_vm.sh --name netem-vm
```

#### Wrong Network Interface
```bash
# Inside the VM, find the correct interface
ip a

# Use the correct interface name
sudo ./net_config.sh --interface ens3 show
```

#### Traffic Not Being Intercepted
```bash
# Verify routing configuration
ip route show
sudo iptables -t nat -L

# Check if traffic flows through VM
sudo tcpdump -i br0 udp port 5683
```

#### VM Won't Start
```bash
# Check if disk image exists
ls -la /var/lib/libvirt/images/

# Verify bridge configuration
ip link show br0
```

### Verification Commands

#### Check Traffic Flow
```bash
# On the VM
sudo tcpdump -i ens3 udp port 5683 or udp port 5684
```

#### Monitor Network Conditions
```bash
# Check current queue discipline
sudo ./net_config.sh show

# Detailed traffic control information
ssh user@vm_ip "sudo tc -s qdisc show dev ens3"
```

#### Reset Everything
```bash
# Reset network emulation
sudo ./net_config.sh reset

# Reset routing (if needed)
sudo ip route flush table ssh-route
sudo iptables -t nat -F
```

### Log Files and Debugging

The scripts provide verbose output when things go wrong. For additional debugging:

```bash
# Enable verbose mode
sudo ./net_config.sh --verbose show

# Check system logs
journalctl -u libvirtd
```

### Getting Help

For additional options and detailed usage:

```bash
sudo ./net_config.sh --help
```

## Script Reference

- **`setup_vm.sh`**: VM creation and management
- **`udp_config.sh`**: Network routing configuration  
- **`net_config.sh`**: Network scenario management
- **`scenarios.md`**: Detailed scenario parameters and manual commands

All scripts require sudo privileges and should be run from the `network_emulation` directory.