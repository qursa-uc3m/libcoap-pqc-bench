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

# Ensure gateway-relative cert paths (../../certs/...) resolve to our repo certs/
GATEWAY_CERTS_LINK="${REPO_ROOT}/paho-mqttsn-gateway/certs"
if [ ! -e "${GATEWAY_CERTS_LINK}" ]; then
    ln -s "${REPO_ROOT}/certs" "${GATEWAY_CERTS_LINK}" 2>/dev/null || true
fi

# Data directory for benchmark output
DATA_DIR="${BENCH_DATA_DIR:-${BENCH_DIR}/data/current}"

rasp_option=""
cert_config="DEFAULT"
SEC_MODE=""

# Broker configuration (local mosquitto by default; override via env if needed)
BROKER_HOST="${MQTTSN_BROKER_HOST:-127.0.0.1}"
BROKER_PORT="${MQTTSN_BROKER_PORT:-1883}"
BROKER_SECURE_PORT="${MQTTSN_BROKER_SECURE_PORT:-8883}"

# Perf configuration
PERF_CMD="${PERF_CMD:-perf}"
PERF_CMD_RPI="${PERF_CMD_RPI:-perf_5.10}"

# Allow running without sudo (useful in non-interactive environments)
# Set BENCH_SUDO_CMD="" to disable sudo.
# Note: use ${var-default} (no ':') so empty is respected.
BENCH_SUDO_CMD="${BENCH_SUDO_CMD-sudo -E}"

rm -f "${REPO_ROOT}/cycles_output.txt"

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

    # Prefer gateway-style relative paths: from MQTTSNGateway/bin -> ../../certs/...
    cert_file_rel="$cert_file"
    key_file_rel="$key_file"

    if [[ "$cert_file" == "${REPO_ROOT}/certs/"* ]]; then
        cert_subpath="${cert_file#${REPO_ROOT}/certs/}"
        cert_file_rel="../../certs/${cert_subpath}"
    fi

    if [[ "$key_file" == "${REPO_ROOT}/certs/"* ]]; then
        key_subpath="${key_file#${REPO_ROOT}/certs/}"
        key_file_rel="../../certs/${key_subpath}"
    fi

    # Save or update algorithm information (match coap_benchmark_server.sh behavior)
    if [ -f "${REPO_ROOT}/algorithm.txt" ]; then
        kem_algorithm=$(head -n 1 "${REPO_ROOT}/algorithm.txt")
        echo "${kem_algorithm}" > "${REPO_ROOT}/algorithm.txt"
        echo "${cert_config}" >> "${REPO_ROOT}/algorithm.txt"
    else
        echo "UNKNOWN_KEM" > "${REPO_ROOT}/algorithm.txt"
        echo "${cert_config}" >> "${REPO_ROOT}/algorithm.txt"
    fi
    
    echo "Using certificate configuration: $cert_config"
    echo "  Certificate: $cert_file"
    echo "  Key: $key_file"
    echo "  CA: $ca_file"
fi

echo "Creating benchmark data directory in ${DATA_DIR} ..."
mkdir -p "${DATA_DIR}"

# Check if gateway binary exists
if [ ! -f "${GATEWAY_BIN}/MQTT-SNGateway" ]; then
    echo "Error: MQTT-SN Gateway not found at ${GATEWAY_BIN}/MQTT-SNGateway"
    echo "Please run: ./scripts/install_paho_mqttsn_gateway.sh"
    exit 1
fi

# Determine perf command based on mode
if [ "$rasp_option" == "true" ]; then
    PERF="$PERF_CMD_RPI"
else
    PERF="$PERF_CMD"
fi

# Configure gateway for security mode
# Uses template from benchmark/mqttsn-gateway-config/ (dynamic certificate handling)
configure_gateway() {
    local template_dir="${BENCH_DIR}/mqttsn-gateway-config"
    local template_gateway="${template_dir}/gateway.conf"
    local template_clients="${template_dir}/clients.conf"

    local runtime_gateway_conf="${DATA_DIR}/gateway.conf"
    local runtime_clients_conf="${DATA_DIR}/clients.conf"

    # Render a runtime config (do not mutate installed bin/gateway.conf)
    if [ -f "${template_gateway}" ]; then
        cp "${template_gateway}" "${runtime_gateway_conf}"
    else
        # Fallback to upstream default
        cp "${GATEWAY_DIR}/gateway.conf" "${runtime_gateway_conf}" 2>/dev/null || true
    fi

    if [ -f "${template_clients}" ]; then
        cp "${template_clients}" "${runtime_clients_conf}"
    elif [ -f "${GATEWAY_DIR}/clients.conf" ]; then
        cp "${GATEWAY_DIR}/clients.conf" "${runtime_clients_conf}" 2>/dev/null || true
    fi

    # Always set broker parameters (so we can avoid local Mosquitto by default)
    sed -i "s|^BrokerName=.*|BrokerName=${BROKER_HOST}|" "${runtime_gateway_conf}" 2>/dev/null || true
    sed -i "s|^BrokerPortNo=.*|BrokerPortNo=${BROKER_PORT}|" "${runtime_gateway_conf}" 2>/dev/null || true
    sed -i "s|^BrokerSecurePortNo=.*|BrokerSecurePortNo=${BROKER_SECURE_PORT}|" "${runtime_gateway_conf}" 2>/dev/null || true

    # Ensure our clients.conf path is set (even if not required when ClientAuthentication=NO)
    if grep -q "^ClientsList=" "${runtime_gateway_conf}" 2>/dev/null; then
        sed -i "s|^ClientsList=.*|ClientsList=${runtime_clients_conf}|" "${runtime_gateway_conf}" 2>/dev/null || true
    else
        echo "ClientsList=${runtime_clients_conf}" >> "${runtime_gateway_conf}"
    fi

    if [ "$SEC_MODE" == "pki" ]; then
        sed -i "s|^DtlsCertsKey=.*|DtlsCertsKey=${cert_file_rel}|" "${runtime_gateway_conf}"
        sed -i "s|^DtlsPrivKey=.*|DtlsPrivKey=${key_file_rel}|" "${runtime_gateway_conf}"
        echo "Gateway configured for DTLS with certificates:"
        echo "  Server cert: ${cert_file_rel}"
        echo "  Server key:  ${key_file_rel}"
        echo "  Broker:      ${BROKER_HOST}:${BROKER_PORT}"
    else
        echo "Gateway configured for plain UDP (nosec)."
        echo "  Broker:      ${BROKER_HOST}:${BROKER_PORT}"
    fi

    echo "Using runtime gateway config: ${runtime_gateway_conf}"
    export MQTT_SN_GATEWAY_CONF="${runtime_gateway_conf}"
}

# Configure gateway
configure_gateway

# Start the gateway with performance monitoring
echo "Starting MQTT-SN Gateway..."
echo "  Security mode: $SEC_MODE"
echo "  Binary: ${GATEWAY_BIN}/MQTT-SNGateway"

cd "${GATEWAY_BIN}"

# Determine output directory for perf stats
PERF_OUTPUT_FILE="${DATA_DIR}/auxiliary_server.txt"

# Run gateway with perf for CPU cycle measurement
if command -v "$PERF" &>/dev/null; then
    echo "Running with perf for CPU cycle measurement..."
    echo "Perf output will be saved to: $PERF_OUTPUT_FILE"
    if [ -n "$BENCH_SUDO_CMD" ]; then
        CMD="$BENCH_SUDO_CMD env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF stat -e cycles -o ${PERF_OUTPUT_FILE} ./MQTT-SNGateway -f \"${MQTT_SN_GATEWAY_CONF}\""
    else
        CMD="env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PERF stat -e cycles -o ${PERF_OUTPUT_FILE} ./MQTT-SNGateway -f \"${MQTT_SN_GATEWAY_CONF}\""
    fi
else
    echo "Warning: perf not available, running without CPU cycle measurement."
    if [ -n "$BENCH_SUDO_CMD" ]; then
        CMD="$BENCH_SUDO_CMD env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./MQTT-SNGateway -f \"${MQTT_SN_GATEWAY_CONF}\""
    else
        CMD="env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./MQTT-SNGateway -f \"${MQTT_SN_GATEWAY_CONF}\""
    fi
fi

echo "Executing: $CMD"
eval $CMD
