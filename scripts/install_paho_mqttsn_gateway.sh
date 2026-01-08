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
GATEWAY_BRANCH="main"

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

cd "$GATEWAY_DIR/MQTTSNGateway"

# Build the gateway
echo "Building MQTT-SN Gateway with ${TRANSPORT} transport..."

# Clean previous builds
rm -rf build.gateway bin/ 2>/dev/null || true
mkdir -p bin

# Use the gateway's build script
if [ "$TRANSPORT" == "dtls" ]; then
    ./build.sh dtls "" "" wolfssl
elif [ "$TRANSPORT" == "udp" ]; then
    ./build.sh udp
else
    echo -e "${RED}Unknown transport: ${TRANSPORT}${NC}"
    echo "Valid options: dtls, udp"
    exit 1
fi

# Copy binaries to bin directory
cp build.gateway/MQTT-SNGateway bin/ 2>/dev/null || true
cp build.gateway/MQTT-SNLogmonitor bin/ 2>/dev/null || true
cp *.conf bin/ 2>/dev/null || true

echo -e "${GREEN}MQTT-SN Gateway built successfully.${NC}"
echo "Binaries located in: ${GATEWAY_DIR}/MQTTSNGateway/bin/"
echo ""
echo "To run the gateway:"
echo "  cd ${GATEWAY_DIR}/MQTTSNGateway/bin"
echo "  ./MQTT-SNGateway"
