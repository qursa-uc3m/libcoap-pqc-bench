#!/bin/bash

# ==============================================
# install_mqttsn_clients.sh
# Clones and builds MQTT-SN clients with PQC support
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CLIENTS_REPO="https://github.com/qursa-uc3m/pq-mqtt-sn-clients.git"
CLIENTS_DIR="${REPO_ROOT}/pq-mqtt-sn-clients"
CLIENTS_BRANCH="main"
PATCHES_DIR="${REPO_ROOT}/benchmark/mqttsn-clients-patches"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}==============================================\${NC}"
echo -e "${GREEN}Installing MQTT-SN Clients with PQC support${NC}"
echo -e "${GREEN}==============================================\${NC}"

# Check dependencies
if ! pkg-config --exists wolfssl; then
    echo -e "${RED}Error: wolfSSL is not installed.${NC}"
    echo "Please run ./scripts/install_wolfssl.sh first."
    exit 1
fi

if ! pkg-config --exists wolfmqtt; then
    echo -e "${YELLOW}Warning: wolfMQTT is not installed.${NC}"
    echo "Running ./scripts/install_wolfmqtt.sh..."
    "${SCRIPT_DIR}/install_wolfmqtt.sh"
fi

# Clone clients if not present
if [ ! -d "$CLIENTS_DIR" ]; then
    echo "Cloning MQTT-SN clients..."
    git clone --branch "$CLIENTS_BRANCH" "$CLIENTS_REPO" "$CLIENTS_DIR"
else
    echo "Clients directory exists, updating..."
    cd "$CLIENTS_DIR"
    git fetch origin
    git checkout "$CLIENTS_BRANCH"
    git pull origin "$CLIENTS_BRANCH"
fi

cd "$CLIENTS_DIR"

# Apply patches if they exist
if [ -d "$PATCHES_DIR" ] && [ "$(ls -A "$PATCHES_DIR"/*.patch 2>/dev/null)" ]; then
    echo "Applying patches from ${PATCHES_DIR}..."
    for patch in "$PATCHES_DIR"/*.patch; do
        if [ -f "$patch" ]; then
            echo "Applying: $(basename "$patch")"
            git apply "$patch" || {
                echo -e "${YELLOW}Patch may already be applied: $(basename "$patch")${NC}"
            }
        fi
    done
fi

# Build clients
echo "Building MQTT-SN clients..."
rm -rf build 2>/dev/null || true
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo -e "${GREEN}MQTT-SN clients built successfully.${NC}"
echo ""
echo "Binaries located in: ${CLIENTS_DIR}/build/bin/"
echo "  - sn-client: Generic MQTT-SN client"
echo "  - sn-pub:    MQTT-SN publisher"
echo "  - sn-sub:    MQTT-SN subscriber"
echo ""
echo "Usage example:"
echo "  export MQTT_WOLFSSL_GROUPS=\"KYBER_LEVEL3\""
echo "  ./build/bin/sn-pub -h <gateway_host> -p <gateway_port> -t"
