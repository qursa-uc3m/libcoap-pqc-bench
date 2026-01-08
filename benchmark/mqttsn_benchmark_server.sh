#!/bin/bash

# ==============================================
# mqttsn_benchmark_server.sh
# Starts MQTT-SN Gateway for benchmarking
# Mirrors coap_benchmark_server.sh functionality
# ==============================================

# Import certificate configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
source "${REPO_ROOT}/certs/config_certs.sh"

# Load environment configuration
if [ -f "${REPO_ROOT}/config.local.env" ]; then
    source "${REPO_ROOT}/config.local.env"
elif [ -f "${REPO_ROOT}/config.env" ]; then
    source "${REPO_ROOT}/config.env"
fi

BENCH_DIR="${REPO_ROOT}/benchmark"
GATEWAY_DIR="${REPO_ROOT}/paho-mqttsn-gateway/MQTTSNGateway"
GATEWAY_BIN="${GATEWAY_DIR}/bin"

# Data directory for benchmark output
DATA_DIR="${BENCH_DATA_DIR:-${BENCH_DIR}/data/current}"

rasp_option=""
cert_config="DEFAULT"
SEC_MODE=""

# Perf configuration
PERF_CMD="${PERF_CMD:-perf}"
PERF_CMD_RPI="${PERF_CMD_RPI:-perf_5.10}"

sudo rm -f "${REPO_ROOT}/cycles_output.txt"

# Function to display usage/help
show_usage() {
    echo "Usage: ${0} -sec-mode <pki|nosec> [-rasp] [-cert-config <CONFIG>]"
    echo ""
    echo "Options:"
    echo "  -sec-mode <pki|nosec>        Security mode to use (required)"
    echo "                               pki: DTLS with certificates"
    echo "                               nosec: Plain UDP (no encryption)"
    echo "  -rasp                        Enable Raspberry Pi mode"
    echo "  -cert-config <CONFIG>        Certificate configuration to use (for PKI mode)"
    echo "  -list-certs                  List available certificate configurations"
    echo "  -h, --help                   Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -sec-mode)
            SEC_MODE="$2"
            shift 2
            ;;
        -[rR][aA][sS][pP])
            rasp_option="true"
            shift
            ;;
        -cert-config)
            cert_config="$2"
            shift 2
            ;;
        -list-certs)
            list_cert_configs
            exit 0
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate security mode
if [ -z "$SEC_MODE" ]; then
    echo "Please provide -sec-mode parameter."
    show_usage
fi

if [[ "$SEC_MODE" != "pki" && "$SEC_MODE" != "nosec" ]]; then
    echo "Error: MQTT-SN only supports 'pki' or 'nosec' modes."
    echo "PSK mode is not currently supported for MQTT-SN."
    show_usage
fi

# Validate certificate configuration for PKI mode
if [ "$SEC_MODE" == "pki" ]; then
    if ! validate_cert_files "$cert_config"; then
        echo "Certificate validation failed. Exiting."
        exit 1
    fi
    
    # Get certificate paths
    cert_paths=$(get_cert_paths "$cert_config")
    IFS=';' read -r cert_file key_file ca_file <<< "$cert_paths"
    
    echo "Using certificate configuration: $cert_config"
    echo "  Certificate: $cert_file"
    echo "  Key: $key_file"
    echo "  CA: $ca_file"
fi

# Check if gateway binary exists
if [ ! -f "${GATEWAY_BIN}/MQTT-SNGateway" ]; then
    echo "Error: MQTT-SN Gateway not found at ${GATEWAY_BIN}/MQTT-SNGateway"
    echo "Please run: ./scripts/install_paho_mqttsn_gateway.sh"
    exit 1
fi

# Check if Mosquitto broker is running
if ! pgrep -x "mosquitto" > /dev/null; then
    echo "Warning: Mosquitto broker does not appear to be running."
    echo "Starting Mosquitto..."
    sudo systemctl start mosquitto || {
        echo "Error: Failed to start Mosquitto. Please install with:"
        echo "  ./scripts/install_mosquitto.sh"
        exit 1
    }
fi

# Determine perf command based on mode
if [ "$rasp_option" == "true" ]; then
    PERF="$PERF_CMD_RPI"
else
    PERF="$PERF_CMD"
fi

# Configure gateway for security mode
configure_gateway() {
    local config_file="${GATEWAY_BIN}/gateway.conf"
    local template_file="${GATEWAY_DIR}/gateway.conf"
    
    # Copy template if needed
    if [ ! -f "$config_file" ]; then
        cp "$template_file" "$config_file"
    fi
    
    if [ "$SEC_MODE" == "pki" ]; then
        # Update DTLS certificate paths in gateway.conf
        sed -i "s|^DtlsCertsKey=.*|DtlsCertsKey=${cert_file}|" "$config_file"
        sed -i "s|^DtlsPrivKey=.*|DtlsPrivKey=${key_file}|" "$config_file"
        sed -i "s|^DtlsCACertFile=.*|DtlsCACertFile=${ca_file}|" "$config_file"
        echo "Gateway configured for DTLS with certificates."
    else
        echo "Gateway configured for plain UDP (nosec)."
    fi
}

# Configure gateway
configure_gateway

# Start the gateway with performance monitoring
echo "Starting MQTT-SN Gateway..."
echo "  Security mode: $SEC_MODE"
echo "  Binary: ${GATEWAY_BIN}/MQTT-SNGateway"

cd "${GATEWAY_BIN}"

# Run gateway with perf for CPU cycle measurement
if command -v "$PERF" &>/dev/null; then
    echo "Running with perf for CPU cycle measurement..."
    CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF stat -e cycles -o ${REPO_ROOT}/cycles_output.txt ./MQTT-SNGateway"
else
    echo "Warning: perf not available, running without CPU cycle measurement."
    CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./MQTT-SNGateway"
fi

echo "Executing: $CMD"
eval $CMD
