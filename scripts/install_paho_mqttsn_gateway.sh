#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

GATEWAY_REPO="https://github.com/qursa-uc3m/paho.mqtt-sn.embedded-c.wolfssl-pq.git"
GATEWAY_DIR="${REPO_ROOT}/paho-mqttsn-gateway"
GATEWAY_BRANCH="master"

# Transport mode: both (default), dtls, or udp
TRANSPORT="${1:-both}"
SSL_LIB="wolfssl"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}Installing MQTT-SN Gateway with PQC support${NC}"
echo -e "${GREEN}Transport: ${TRANSPORT}, SSL: ${SSL_LIB}${NC}"
echo -e "${GREEN}==============================================${NC}"

# Check dependencies for DTLS mode
if [ "$TRANSPORT" == "dtls" ] || [ "$TRANSPORT" == "both" ]; then
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

# Ensure gateway-relative cert paths (../../certs/...) resolve to repo certs/
ln -sfn "${REPO_ROOT}/certs" "${GATEWAY_DIR}/certs"

CONFIG_TEMPLATES_DIR="${REPO_ROOT}/benchmark/mqttsn-gateway-config"

copy_gateway_configs() {
    local output_dir="$1"
    if [ -d "$CONFIG_TEMPLATES_DIR" ]; then
        cp "${CONFIG_TEMPLATES_DIR}/gateway.conf" "$output_dir/" 2>/dev/null || true
        cp "${CONFIG_TEMPLATES_DIR}/clients.conf" "$output_dir/" 2>/dev/null || true
    else
        cp "${GATEWAY_DIR}/MQTTSNGateway"/*.conf "$output_dir/" 2>/dev/null || true
    fi
}

build_gateway() {
    local transport="$1"
    local build_dir="${GATEWAY_DIR}/build.gateway.${transport}"
    local output_dir="${GATEWAY_DIR}/MQTTSNGateway/bin-${transport}"

    echo "Building MQTT-SN Gateway with ${transport} transport..."
    rm -rf "$build_dir" "$output_dir" MQTTSNGateway/bin/* 2>/dev/null || true
    mkdir -p "$build_dir" "$output_dir" MQTTSNGateway/bin

    cd "$build_dir"
    if [ "$transport" == "dtls" ]; then
        cmake .. -DSENSORNET=dtls -DSSL_LIB=wolfssl
    elif [ "$transport" == "udp" ]; then
        mkdir -p "${GATEWAY_DIR}/MQTTSNGateway/src/linux/udp/openssl"
        ln -sfn ../SensorNetwork.cpp "${GATEWAY_DIR}/MQTTSNGateway/src/linux/udp/openssl/SensorNetwork.cpp"
        ln -sfn ../SensorNetwork.h "${GATEWAY_DIR}/MQTTSNGateway/src/linux/udp/openssl/SensorNetwork.h"
        cmake .. -DSENSORNET=udp
    else
        echo -e "${RED}Unknown transport: ${transport}${NC}"
        exit 1
    fi
    make -j$(nproc)

    cp "${GATEWAY_DIR}/MQTTSNGateway/bin/MQTT-SNGateway" "$output_dir/"
    copy_gateway_configs "$output_dir"
}

case "$TRANSPORT" in
    both)
        build_gateway dtls
        build_gateway udp
        cp -a "${GATEWAY_DIR}/MQTTSNGateway/bin-dtls/." "${GATEWAY_DIR}/MQTTSNGateway/bin/"
        ;;
    dtls|udp)
        build_gateway "$TRANSPORT"
        cp -a "${GATEWAY_DIR}/MQTTSNGateway/bin-${TRANSPORT}/." "${GATEWAY_DIR}/MQTTSNGateway/bin/"
        ;;
    *)
        echo -e "${RED}Unknown transport: ${TRANSPORT}${NC}"
        echo "Valid options: both, dtls, udp"
        exit 1
        ;;
esac

echo -e "${GREEN}MQTT-SN Gateway built successfully.${NC}"
echo "Binaries located in: ${GATEWAY_DIR}/MQTTSNGateway/bin-dtls/ and bin-udp/"
echo ""
echo "To run the gateway:"
echo "  cd ${GATEWAY_DIR}/MQTTSNGateway/bin"
echo "  ./MQTT-SNGateway"

