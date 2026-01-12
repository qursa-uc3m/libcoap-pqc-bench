#!/bin/bash

# ==============================================
# install_paho_mqttsn_gateway.sh
# Clones and builds the MQTT-SN Gateway with PQC support
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

GATEWAY_REPO="https://github.com/qursa-uc3m/paho.mqtt-sn.embedded-c.wolfssl-pq.git"
GATEWAY_DIR="${REPO_ROOT}/paho-mqttsn-gateway"
GATEWAY_BRANCH="master"

# Transport mode: dtls (default) or udp for nosec
TRANSPORT="${1:-dtls}"
SSL_LIB="wolfssl"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}==============================================\${NC}"
echo -e "${GREEN}Installing MQTT-SN Gateway with PQC support${NC}"
echo -e "${GREEN}Transport: ${TRANSPORT}, SSL: ${SSL_LIB}${NC}"
echo -e "${GREEN}==============================================\${NC}"

# Check dependencies for DTLS mode
if [ "$TRANSPORT" == "dtls" ]; then
    if ! pkg-config --exists wolfssl; then
        echo -e "${RED}Error: wolfSSL is not installed.${NC}"
        echo "Please run ./scripts/install_wolfssl.sh first."
        exit 1
    fi
fi

# Clone gateway if not present
if [ ! -d "$GATEWAY_DIR" ]; then
    echo "Cloning MQTT-SN Gateway..."
    git clone --branch "$GATEWAY_BRANCH" "$GATEWAY_REPO" "$GATEWAY_DIR"
else
    echo "Gateway directory exists, updating..."
    cd "$GATEWAY_DIR"
    git pull origin "$GATEWAY_BRANCH"
fi

cd "$GATEWAY_DIR"

# Build the gateway
echo "Building MQTT-SN Gateway with ${TRANSPORT} transport..."

# Clean previous builds
rm -rf build.gateway MQTTSNGateway/bin/* 2>/dev/null || true
mkdir -p MQTTSNGateway/bin
mkdir -p build.gateway
cd build.gateway

# Use cmake directly (the build.sh script has issues with parameter passing)
if [ "$TRANSPORT" == "dtls" ]; then
    cmake .. -DSENSORNET=dtls -DSSL_LIB=wolfssl
elif [ "$TRANSPORT" == "udp" ]; then
    cmake .. -DSENSORNET=udp
else
    echo -e "${RED}Unknown transport: ${TRANSPORT}${NC}"
    echo "Valid options: dtls, udp"
    exit 1
fi

# Build all targets
make -j$(nproc)

# Copy our custom config templates (with dynamic certificate placeholders)
CONFIG_TEMPLATES_DIR="${REPO_ROOT}/benchmark/mqttsn-gateway-config"
if [ -d "$CONFIG_TEMPLATES_DIR" ]; then
    echo "Copying custom gateway configuration templates..."
    cp "${CONFIG_TEMPLATES_DIR}/gateway.conf" "${GATEWAY_DIR}/MQTTSNGateway/bin/" 2>/dev/null || true
    cp "${CONFIG_TEMPLATES_DIR}/clients.conf" "${GATEWAY_DIR}/MQTTSNGateway/bin/" 2>/dev/null || true
else
    # Fallback to original config files
    cd "$GATEWAY_DIR/MQTTSNGateway"
    cp *.conf bin/ 2>/dev/null || true
fi

echo -e "${GREEN}MQTT-SN Gateway built successfully.${NC}"
echo "Binaries located in: ${GATEWAY_DIR}/MQTTSNGateway/bin/"
echo ""
echo "To run the gateway:"
echo "  cd ${GATEWAY_DIR}/MQTTSNGateway/bin"
echo "  ./MQTT-SNGateway"

