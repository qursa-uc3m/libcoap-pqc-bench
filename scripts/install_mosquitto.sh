#!/bin/bash

# ==============================================
# install_mosquitto.sh
# Installs Mosquitto MQTT broker for MQTT-SN Gateway
# ==============================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}==============================================\${NC}"
echo -e "${GREEN}Installing Mosquitto MQTT Broker${NC}"
echo -e "${GREEN}==============================================\${NC}"

# Check OS type
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo -e "${RED}Cannot detect OS type${NC}"
    exit 1
fi

case $OS in
    ubuntu|debian)
        echo "Installing Mosquitto on Debian/Ubuntu..."
        sudo apt-get update
        sudo apt-get install -y mosquitto mosquitto-clients
        ;;
    fedora|centos|rhel)
        echo "Installing Mosquitto on Fedora/CentOS/RHEL..."
        sudo dnf install -y mosquitto
        ;;
    arch)
        echo "Installing Mosquitto on Arch Linux..."
        sudo pacman -S mosquitto
        ;;
    *)
        echo -e "${YELLOW}Unknown OS: $OS${NC}"
        echo "Please install Mosquitto manually."
        exit 1
        ;;
esac

# Enable and start mosquitto service
echo "Enabling Mosquitto service..."
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Verify installation
if systemctl is-active --quiet mosquitto; then
    echo -e "${GREEN}Mosquitto is running.${NC}"
else
    echo -e "${YELLOW}Mosquitto installed but not running.${NC}"
    echo "Start manually with: sudo systemctl start mosquitto"
fi

echo ""
echo "Mosquitto MQTT broker installed successfully."
echo ""
echo "Default configuration:"
echo "  - Port: 1883 (unencrypted)"
echo "  - Config file: /etc/mosquitto/mosquitto.conf"
echo ""
echo "Test with:"
echo "  mosquitto_sub -t 'test/topic' &"
echo "  mosquitto_pub -t 'test/topic' -m 'Hello MQTT'"
