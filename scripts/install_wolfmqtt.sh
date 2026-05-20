#!/bin/bash

# ==============================================
# install_wolfmqtt.sh
# Installs wolfMQTT library for MQTT-SN clients
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

WOLFMQTT_VERSION="v1.19.0"
WOLFMQTT_DIR="${REPO_ROOT}/wolfMQTT"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}==============================================\${NC}"
echo -e "${GREEN}Installing wolfMQTT ${WOLFMQTT_VERSION}${NC}"
echo -e "${GREEN}==============================================\${NC}"

# Check if wolfSSL is installed
if ! pkg-config --exists wolfssl; then
    echo -e "${RED}Error: wolfSSL is not installed.${NC}"
    echo "Please run ./scripts/install_wolfssl.sh first."
    exit 1
fi

# Clone wolfMQTT if not present
if [ ! -d "$WOLFMQTT_DIR" ]; then
    echo "Cloning wolfMQTT..."
    git clone https://github.com/wolfSSL/wolfMQTT.git "$WOLFMQTT_DIR"
fi

cd "$WOLFMQTT_DIR"

# Checkout specific version
if [ "$WOLFMQTT_VERSION" != "master" ]; then
    git fetch --tags
    git checkout "$WOLFMQTT_VERSION"
fi

# Build wolfMQTT with MQTT-SN support
echo "Building wolfMQTT with MQTT-SN support..."
./autogen.sh
./configure --enable-sn
make -j$(nproc)

# Install
echo "Installing wolfMQTT..."
sudo make install
sudo ldconfig

echo -e "${GREEN}wolfMQTT ${WOLFMQTT_VERSION} installed successfully.${NC}"
