#!/bin/bash
# Device configurations
COMPUTER="192.168.0.145"
VM="192.168.0.172"
RPI="192.168.0.157"
VM_USER="dasobral"
VM_NETWORK="192.168.0.0/24"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# SSH control socket paths
VM_SOCKET="/tmp/vm_ssh_control"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root${NC}"
    exit 1
fi

# Initialize SSH control sockets
ssh -M -S $VM_SOCKET -fN $VM_USER@$VM

# Function to configure VM networking
configure_vm() {
    local name=$1
    local socket=$2
    
    echo "Configuring VM ($name)..."
    
    ssh -S $socket $VM_USER@$VM -t "sudo bash -c '
        # Enable IP forwarding
        sysctl -w net.ipv4.ip_forward=1
        
        # Configure iptables
        iptables -t nat -F
        iptables -F FORWARD

        # Configure iptables for UDP
        sudo iptables -t nat -A POSTROUTING -o ens3 -j MASQUERADE
        sudo iptables -A FORWARD -i ens3 -o ens3 -j ACCEPT
    '"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ VM configuration successful${NC}"
    else
        echo -e "${RED}✗ VM configuration failed${NC}"
        exit 1
    fi
}

# Function to configure Computer networking
configure_computer() {
    local name=$1
    
    echo "Configuring Computer ($name)..."
    
    # Enable IP forwarding
    sysctl -w net.ipv4.ip_forward=1
    
    # Configure iptables for UDP
    sudo ip route add $RPI/32 via $VM dev br0
    sudo iptables -t nat -A PREROUTING -p udp --dport 5683 -j DNAT --to-destination $VM
    sudo iptables -t nat -A PREROUTING -p udp --dport 5684 -j DNAT --to-destination $VM
    sudo iptables -A OUTPUT -p icmp --icmp-type port-unreachable -j DROP
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Computer configuration successful${NC}"
    else
        echo -e "${RED}✗ Computer configuration failed${NC}"
        exit 1
    fi
}

# Main configuration process
echo "=== Starting Network Configuration ==="

# Configure each device
configure_vm "VM" $VM_SOCKET

configure_computer "Computer"

# Clean up control sockets
ssh -S $VM_SOCKET -O exit $VM_USER@$VM

echo "=== Network Configuration Complete ==="
