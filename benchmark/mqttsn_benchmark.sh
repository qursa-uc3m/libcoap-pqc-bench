#!/bin/bash

# ==============================================
# mqttsn_benchmark.sh
# MQTT-SN client benchmark script
# Mirrors coap_benchmark.sh functionality
# ==============================================

# Import certificate configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
source "${REPO_ROOT}/certs/config_certs.sh"
export REPO_ROOT

# Load environment configuration
if [ -f "${REPO_ROOT}/config.local.env" ]; then
    source "${REPO_ROOT}/config.local.env"
elif [ -f "${REPO_ROOT}/config.env" ]; then
    source "${REPO_ROOT}/config.env"
fi

# Set defaults for local mode configuration
LOCAL_MODE="${LOCAL_MODE:-false}"
LOCAL_SERVER_ADDRESS="${LOCAL_SERVER_ADDRESS:-127.0.0.1}"
LOCAL_CAPTURE_INTERFACE="${LOCAL_CAPTURE_INTERFACE:-lo}"

# Timing defaults
TIMING_PORT_RELEASE_LOCAL="${TIMING_PORT_RELEASE_LOCAL:-0.2}"
TIMING_PORT_RELEASE_REMOTE="${TIMING_PORT_RELEASE_REMOTE:-1.0}"
TIMING_SERVER_START_LOCAL="${TIMING_SERVER_START_LOCAL:-1.0}"
TIMING_SERVER_START_REMOTE="${TIMING_SERVER_START_REMOTE:-3.0}"
TIMING_TSHARK_FLUSH_LOCAL="${TIMING_TSHARK_FLUSH_LOCAL:-1.0}"
TIMING_TSHARK_FLUSH_REMOTE="${TIMING_TSHARK_FLUSH_REMOTE:-2.0}"
TIMING_SERVER_STOP_LOCAL="${TIMING_SERVER_STOP_LOCAL:-0.5}"
TIMING_SERVER_STOP_REMOTE="${TIMING_SERVER_STOP_REMOTE:-3.0}"
TIMING_CLIENT_DELAY="${TIMING_CLIENT_DELAY:-0.5}"
TIMING_CLIENT_POLL="${TIMING_CLIENT_POLL:-0.1}"

BENCH_DIR="${REPO_ROOT}/benchmark"
CLIENTS_DIR="${REPO_ROOT}/pq-mqtt-sn-clients"
CLIENTS_BIN="${CLIENTS_DIR}/build/bin"

# Data directory for benchmark output
DATA_DIR="${BENCH_DATA_DIR:-${BENCH_DIR}/data/current}"
export BENCH_DATA_DIR="$DATA_DIR"

# MQTT-SN Gateway configuration
GATEWAY_HOST="${MQTTSN_GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${MQTTSN_GATEWAY_PORT:-10000}"

# Global variables
bridge_interface="${BRIDGE_INTERFACE:-br0}"
server_ip="${RASPBERRY_PI_IP}"
tshark_pid=""

# Default values
n=""
sec_mode=""
confirm_param=""
rasp_param=""
parallelization_mode=""
cert_config="DEFAULT"

# Function to display usage information
usage() {
    echo "Usage: $0 -n <positive_integer> -sec-mode <pki|nosec> [-confirm <con|non>] [-rasp] [-parallelization <background|parallel>] [-cert-config <CONFIG>]"
    echo ""
    echo "Required parameters:"
    echo "  -n <integer>                 Number of clients that will make requests"
    echo "  -sec-mode <pki|nosec>        Security mode to use"
    echo ""
    echo "Optional parameters:"
    echo "  -confirm <con|non>           Message confirmation mode (default: con)"
    echo "  -rasp                        Indicates gateway is running on Raspberry Pi"
    echo "  -parallelization <option>    How clients run:"
    echo "                               'background': clients run in the same core"
    echo "                               'parallel': clients run in different cores"
    echo "  -cert-config <CONFIG>        Certificate configuration to use (for PKI mode)"
    echo "  -list-certs                  List available certificate configurations"
    echo "  -h, --help                   Show this help message"
    exit 1
}

# Function to clean up on interruption
cleanup() {
    echo "Script interrupted. Cleaning up..."
    [ -n "$tshark_pid" ] && kill -9 "$tshark_pid" 2>/dev/null
    rm -f "${DATA_DIR}/udp_conversations.pcapng" 2>/dev/null
    exit 1
}

trap cleanup SIGINT SIGTERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -n)
            n="$2"
            shift 2
            ;;
        -sec-mode)
            sec_mode="$2"
            shift 2
            ;;
        -confirm)
            confirm_param="$2"
            shift 2
            ;;
        -[rR][aA][sS][pP])
            rasp_param="true"
            shift
            ;;
        -parallelization)
            parallelization_mode="$2"
            shift 2
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
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$n" ]; then
    echo "Error: -n parameter is required"
    usage
fi

if [ -z "$sec_mode" ]; then
    echo "Error: -sec-mode parameter is required"
    usage
fi

if [[ "$sec_mode" != "pki" && "$sec_mode" != "nosec" ]]; then
    echo "Error: MQTT-SN only supports 'pki' or 'nosec' modes."
    usage
fi

# Set default confirmation mode
confirm_param="${confirm_param:-con}"

# Check if client binaries exist
if [ ! -f "${CLIENTS_BIN}/sn-pub" ]; then
    echo "Error: MQTT-SN clients not found at ${CLIENTS_BIN}/"
    echo "Please run: ./scripts/install_mqttsn_clients.sh"
    exit 1
fi

# Configure for PKI mode
if [ "$sec_mode" == "pki" ]; then
    if ! validate_cert_files "$cert_config"; then
        echo "Certificate validation failed. Exiting."
        exit 1
    fi
    
    # Get certificate paths
    cert_paths=$(get_cert_paths "$cert_config")
    IFS=';' read -r cert_file key_file ca_file <<< "$cert_paths"
    
    echo "Using certificate configuration: $cert_config"
    echo "  CA: $ca_file"
fi

# Determine target address and interface
if [ "$rasp_param" == "true" ]; then
    target_host="$server_ip"
    capture_interface="$bridge_interface"
else
    target_host="$LOCAL_SERVER_ADDRESS"
    capture_interface="$LOCAL_CAPTURE_INTERFACE"
fi

# Read KEM algorithm from algorithm.txt
kem_algorithm=""
if [ -f "${REPO_ROOT}/algorithm.txt" ]; then
    kem_algorithm=$(head -n 1 "${REPO_ROOT}/algorithm.txt")
fi

echo "=============================================="
echo "MQTT-SN Benchmark Configuration"
echo "=============================================="
echo "Number of clients: $n"
echo "Security mode: $sec_mode"
echo "Gateway host: ${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "Parallelization: ${parallelization_mode:-none}"
echo "KEM algorithm: ${kem_algorithm:-default}"
echo "=============================================="

# Create data directory if needed
mkdir -p "$DATA_DIR"

# Start packet capture
echo "Starting packet capture on interface: $capture_interface"
tshark -i "$capture_interface" -w "${DATA_DIR}/udp_conversations.pcapng" \
    -f "udp port ${GATEWAY_PORT}" &
tshark_pid=$!
sleep 1

# Build client command
build_client_cmd() {
    local client_id=$1
    local cmd="${CLIENTS_BIN}/sn-pub"
    
    cmd="$cmd -h ${GATEWAY_HOST} -p ${GATEWAY_PORT}"
    cmd="$cmd -t"  # Enable timing output
    
    if [ "$sec_mode" == "pki" ]; then
        cmd="$cmd -A ${ca_file}"  # CA certificate
    fi
    
    echo "$cmd"
}

# Record start time
start_time=$(date +%s.%N)

# Run clients based on parallelization mode
echo "Starting $n MQTT-SN clients..."

case "$parallelization_mode" in
    "background")
        # Run clients in background (same core)
        for ((i=1; i<=n; i++)); do
            client_cmd=$(build_client_cmd $i)
            MQTT_WOLFSSL_GROUPS="$kem_algorithm" eval "$client_cmd" &
        done
        wait
        ;;
    "parallel")
        # Run clients in parallel using GNU parallel
        if command -v parallel &>/dev/null; then
            export -f build_client_cmd
            export CLIENTS_BIN GATEWAY_HOST GATEWAY_PORT sec_mode ca_file kem_algorithm
            seq 1 $n | parallel -j $n "
                client_cmd=\$(build_client_cmd {})
                MQTT_WOLFSSL_GROUPS=\"$kem_algorithm\" eval \"\$client_cmd\"
            "
        else
            echo "Warning: GNU parallel not found, falling back to background mode"
            for ((i=1; i<=n; i++)); do
                client_cmd=$(build_client_cmd $i)
                MQTT_WOLFSSL_GROUPS="$kem_algorithm" eval "$client_cmd" &
            done
            wait
        fi
        ;;
    *)
        # Sequential execution
        for ((i=1; i<=n; i++)); do
            client_cmd=$(build_client_cmd $i)
            MQTT_WOLFSSL_GROUPS="$kem_algorithm" eval "$client_cmd"
        done
        ;;
esac

# Record end time
end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)

echo "All clients completed."
echo "Total duration: ${duration}s"

# Stop packet capture
sleep 1  # Allow final packets to be captured
kill "$tshark_pid" 2>/dev/null
tshark_pid=""

# Save timing data
timing_file="${DATA_DIR}/timing_${sec_mode}_${cert_config}_${n}clients.txt"
echo "duration=${duration}" > "$timing_file"
echo "clients=$n" >> "$timing_file"
echo "sec_mode=$sec_mode" >> "$timing_file"
echo "kem_algorithm=$kem_algorithm" >> "$timing_file"
echo "cert_config=$cert_config" >> "$timing_file"

echo "Results saved to: $DATA_DIR"
echo "Timing file: $timing_file"
