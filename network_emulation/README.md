# Network Emulation for libcoap PQC Benchmarking

This guide provides instructions for setting up and using network emulation capabilities to test CoAP with PQC under various network conditions.

## Table of Contents
- [Overview](#overview)
- [Setup Options](#setup-options)
- [VM Setup with KVM](#vm-setup-with-kvm)
  - [Automated VM Setup](#automated-vm-setup)
  - [Manual VM Setup](#manual-vm-setup)
    - [Debian 11](#on-debian-11)
    - [Ubuntu 20.04](#on-ubuntu-2004)
- [Network Configuration](#network-configuration)
  - [Using the Automated Script](#using-the-automated-script)
  - [Manual Configuration](#manual-configuration)
- [Applying Network Conditions](#applying-network-conditions)
  - [Delay](#adding-delay)
  - [Packet Loss](#adding-packet-loss)
  - [Other Network Conditions](#other-network-conditions)
- [Troubleshooting](#troubleshooting)

## Overview

Network emulation allows testing applications under various network conditions such as latency, packet loss, and bandwidth limitations. This is essential for understanding how PQC algorithms behave in real-world networking scenarios.

### NetEm vs. Other Tools

NetEm (Network Emulator) is a Linux kernel component that allows for precise control over network characteristics:

- **NetEm**: Provides comprehensive network emulation including:
  - Delay
  - Packet loss
  - Packet duplication
  - Packet corruption
  - Packet reordering
  - Bandwidth rate limitations

- **Pumba**: An alternative that offers:
  - Delay
  - Packet loss

NetEm is the recommended approach for our benchmarking framework due to its comprehensive features.

## Setup Options

There are two main approaches to set up network emulation:

1. **Automated Setup**: Using the provided scripts (recommended)
2. **Manual Setup**: Following step-by-step instructions

For most users, the automated setup should be sufficient. Manual setup instructions are provided for advanced customization.

## VM Setup with KVM

Network emulation is performed through a virtual machine (VM) running between your client and server. This allows for traffic interception and modification.

### Automated VM Setup

The repository includes a script for automated VM setup:

```bash
# First-time installation of a new VM
sudo ./setup_vm.sh --install --name <vm_name>

# For starting an existing VM
sudo ./setup_vm.sh --name <vm_name>
```

During the first-time installation:
1. You'll be prompted for an ISO file path
2. The script will create a 10GB disk image
3. It will set up a VM with 2 cores and 2GB RAM
4. Network will be configured using a TAP device and bridge

When using Ubuntu distributions, you might need to modify GRUB parameters:
- Press 'e' to edit boot parameters at the GRUB menu
- Add 'console=ttyS0 text' at the end of the linux line
- Example: `linux /casper/vmlinuz ... console=ttyS0 text`

### Manual VM Setup

For users who need more customization, here are the manual setup instructions.

#### On Debian 11

1. Install required packages:
   ```bash
   sudo apt update
   sudo apt install qemu-kvm virtinst
   ```

2. Check hardware virtualization support:
   ```bash
   egrep -c '(vmx|svm)' /proc/cpuinfo
   sudo apt install -y cpu-checker
   kvm-ok
   ```

3. Install KVM and related tools:
   ```bash
   sudo apt install libvirt-daemon-system libvirt-clients bridge-utils
   sudo systemctl start libvirt
   sudo systemctl enable libvirtd
   ```

4. Set up networking:
   ```bash
   sudo virsh net-start default
   sudo virsh net-autostart default
   ip a show virbr0  # Verify bridge exists
   ```

5. Create a VM:
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

1. Install required packages:
   ```bash
   sudo apt update
   sudo apt install uml-utilities qemu-utils bridge-utils
   ```

2. Create a disk image:
   ```bash
   qemu-img create -f qcow2 /var/lib/libvirt/images/netem-vm.qcow2 10G
   ```

3. Set up TAP device and bridge:
   ```bash
   sudo tunctl -t tap0 -u `whoami`
   sudo ip link set tap0 up
   sudo brctl addif virbr0 tap0
   ```

4. Launch VM with QEMU/KVM:
   ```bash
   sudo qemu-system-x86_64 -enable-kvm \
     -m 2048 \
     -smp cpus=2 \
     -hda /var/lib/libvirt/images/netem-vm.qcow2 \
     -cdrom ~/Downloads/debian-10.13.0-amd64-netinst.iso \
     -net nic,model=virtio -net tap,ifname=tap0,script=no,downscript=no
   ```

## Network Configuration

After setting up the VM, you need to configure it to intercept and process network traffic between your client and server.

### Using the Automated Script

The repository includes a script for automated network configuration:

```bash
# Configure traffic redirection through the VM
sudo ./udp_config.sh
```

This script:
1. Configures the client host to route traffic through the VM
2. Sets up the VM to forward traffic between client and server
3. Configures SSH direct routing to bypass VM for SSH traffic
4. Applies necessary iptables rules

### Manual Configuration

For manual configuration:

1. Inside the VM, enable IP forwarding:
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   sudo sh -c 'echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf'
   ```

2. Configure iptables for forwarding:
   ```bash
   sudo iptables -t nat -A POSTROUTING -o ens3 -j MASQUERADE
   sudo iptables -A FORWARD -i ens3 -o ens3 -j ACCEPT
   ```

3. On the client host, route traffic through VM:
   ```bash
   # Enable IP forwarding
   sudo sysctl -w net.ipv4.ip_forward=1
   
   # Make the server reachable through the VM
   sudo ip route add <server_ip>/32 via <vm_ip> dev br0
   
   # Configure iptables for the target port
   sudo iptables -t nat -A PREROUTING -p udp --dport <target_port> -j DNAT --to-destination <vm_ip>
   ```

4. For CoAP with DTLS (port 5684):
   ```bash
   # Filter out ICMP packets appearing at the end of the DTLS handshake
   sudo iptables -A OUTPUT -p icmp --icmp-type port-unreachable -j DROP
   ```

## Applying Network Conditions

Once the VM is set up and network traffic is flowing through it, you can apply various network conditions using NetEm.

### Adding Delay

To simulate network latency:

```bash
# Add 100ms delay to all packets
sudo tc qdisc add dev <vm_interface> root netem delay 100ms

# Add variable delay (100ms ±20ms)
sudo tc qdisc add dev <vm_interface> root netem delay 100ms 20ms

# Add correlated delay (25% correlation)
sudo tc qdisc add dev <vm_interface> root netem delay 100ms 20ms 25%
```

### Adding Packet Loss

To simulate unreliable networks:

```bash
# Add 10% packet loss
sudo tc qdisc add dev <vm_interface> root netem loss 10%

# Add burst packet loss (10% with 25% correlation)
sudo tc qdisc add dev <vm_interface> root netem loss 10% 25%
```

### Other Network Conditions

NetEm supports various other network conditions:

```bash
# Add packet corruption (2%)
sudo tc qdisc add dev <vm_interface> root netem corrupt 2%

# Add packet duplication (1%)
sudo tc qdisc add dev <vm_interface> root netem duplicate 1%

# Add packet reordering (25% of packets with 10ms delay)
sudo tc qdisc add dev <vm_interface> root netem delay 10ms reorder 25%

# Limit bandwidth to 1Mbit/s
sudo tc qdisc add dev <vm_interface> root tbf rate 1mbit burst 32kbit latency 400ms
```

To remove all network conditions:

```bash
sudo tc qdisc del dev <vm_interface> root
```

## Troubleshooting

### Finding the VM Interface Name

If you're not sure what network interface to use in the VM:

```bash
# List all interfaces
ip a

# The interface will typically be named ens3, enp1s0, or eth0
```

### Checking Traffic Flow

To verify traffic is flowing through the VM:

```bash
# Install tcpdump
sudo apt install tcpdump

# Monitor traffic on the interface
sudo tcpdump -i <vm_interface> udp port 5683 or udp port 5684
```

### Resetting Network Configuration

If you need to reset the network configuration:

```bash
# On the VM
sudo iptables -t nat -F
sudo iptables -F FORWARD

# On the host
sudo ip route del <server_ip>/32
sudo iptables -t nat -F PREROUTING
sudo iptables -D OUTPUT -p icmp --icmp-type port-unreachable -j DROP
```