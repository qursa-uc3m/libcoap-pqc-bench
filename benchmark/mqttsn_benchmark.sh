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
ENERGY_MONITOR_TYPE="${ENERGY_MONITOR_TYPE:-fnirsi}"
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
TIMING_ENERGY_SIGNAL="${TIMING_ENERGY_SIGNAL:-0.5}"
TIMING_ENERGY_FLUSH="${TIMING_ENERGY_FLUSH:-1.0}"
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
role="pub"  # default role: publisher

# Function to display usage information
usage() {
    echo "Usage: $0 -n <positive_integer> -sec-mode <pki|nosec> [-role <pub|sub>] [-confirm <con|non>] [-rasp] [-parallelization <background|parallel>] [-cert-config <CONFIG>]"
    echo ""
    echo "Required parameters:"
    echo "  -n <integer>                 Number of clients that will make requests"
    echo "  -sec-mode <pki|nosec>        Security mode to use"
    echo ""
    echo "Optional parameters:"
    echo "  -role <pub|sub>              Client role: 'pub' (publisher) or 'sub' (subscriber)"
    echo "                               Default: pub"
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
    
    # Stop energy monitoring if active
    if [ "${MEASURE_ENERGY:-false}" == "true" ] && [ -f "${BENCH_DIR}/.energy_monitor_pid" ]; then
        stop_energy_monitoring
    fi
    
    rm -f "${DATA_DIR}/udp_conversations.pcapng" 2>/dev/null
    exit 1
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# Energy Monitoring Functions (mirrored from coap_benchmark.sh)
# ============================================================================

# Function to start energy monitoring with guaranteed initialization using a named pipe
start_energy_monitoring() {
    energy_filename=$(generate_filenames "energy" false)
    energy_name="$energy_filename"
    start_sock="${BENCH_DIR}/.energy_monitor_start.sock"
    stop_sock="${BENCH_DIR}/.energy_monitor_stop.sock"
    
    echo "Starting energy monitoring..."
    
    rm -f "$start_sock" "$stop_sock"
    mkfifo "$start_sock" 2>/dev/null
    mkfifo "$stop_sock" 2>/dev/null
    
    local python_cmd="python3"
    local backend="fnirsi"
    
    if [ "$ENERGY_MONITOR_TYPE" = "codecarbon" ] || [ "$LOCAL_MODE" = "true" ]; then
        backend="codecarbon"
        echo "Using CodeCarbon energy monitor (software-based estimation)"
        if [ -f "${REPO_ROOT}/.bench-env/bin/python3" ]; then
            python_cmd="${REPO_ROOT}/.bench-env/bin/python3"
        fi
    else
        echo "Using FNIRSI energy monitor (USB power meter)"
    fi
    
    "$python_cmd" "${BENCH_DIR}/energy_monitor.py" --backend "$backend" --force-reset \
           --output "${DATA_DIR}/$energy_name" \
           --start-pipe "$start_sock" \
           --stop-pipe "$stop_sock" &
    
    ENERGY_PID=$!
    
    # Store the PID for later termination
    echo $ENERGY_PID > ${BENCH_DIR}/.energy_monitor_pid
    echo "Energy monitoring started with PID $ENERGY_PID"
    
    # Wait for the energy monitoring to signal readiness
    echo "Waiting for energy monitor to initialize..."
    
    # Read from the start pipe - this will block until energy monitor writes to it
    # Use a timeout to prevent indefinite hanging
    if read -t 30 status < "$start_sock"; then
        if [ "$status" = "READY" ]; then
            echo "Energy monitor signaled ready"
            echo "Energy monitor is ready and collecting data"
            # Remove the start pipe since we're done with it
            rm -f "$start_sock"
            return 0
        else
            echo "WARNING: Energy monitor sent unexpected status: $status"
        fi
    else
        echo "WARNING: Timed out waiting for energy monitor to signal ready"
        echo "Continuing anyway, but energy data may be incomplete or missing."
    fi
    
    # Clean up if something went wrong
    rm -f "$start_sock" "$stop_sock"
    return 1
}

# Function to stop energy monitoring and ensure completion
stop_energy_monitoring() {
    if [ ! -f ${BENCH_DIR}/.energy_monitor_pid ]; then
        echo "No energy monitoring process found"
        return
    fi
    
    ENERGY_PID=$(cat ${BENCH_DIR}/.energy_monitor_pid)
    stop_sock="${BENCH_DIR}/.energy_monitor_stop.sock"
    
    echo "Stopping energy monitoring (PID: $ENERGY_PID)..."
    
    # Signal the energy monitor to prepare for termination
    kill -USR1 $ENERGY_PID 2>/dev/null
    
    # Give it a moment to process the signal and open the pipe
    sleep $TIMING_ENERGY_SIGNAL
    
    # Wait for completion signal from energy monitor
    if [ -p "$stop_sock" ]; then
        if read -t 10 status < "$stop_sock"; then
            if [ "$status" = "DONE" ]; then
                echo "Energy monitor completed data processing"
            else
                echo "WARNING: Energy monitor sent unexpected status: $status"
            fi
        else
            echo "WARNING: Timed out waiting for energy monitor completion signal"
        fi
    else
        echo "WARNING: Stop pipe not found, energy monitor may not complete properly"
    fi
    
    # Now send the actual termination signal
    kill -2 $ENERGY_PID
    
    # Clean up
    rm -f "$stop_sock"
    rm -f ${BENCH_DIR}/.energy_monitor_pid
    
    # Wait a moment for energy data to be processed
    sleep $TIMING_ENERGY_FLUSH
}

# Filenames generation function
generate_filenames() {
    local prefix="$1"
    local add_udp_prefix="$2"
    
    # Define filename_add based on role
    local filename_add="_${role}"

    # Prepare base filename for results
    local filename=""
    if [ "$sec_mode" == "pki" ]; then
        # PKI mode: include cert config
        cert_indicator="_${cert_config}"
        
        if [ -n "$parallelization_mode" ]; then 
            filename="${prefix}${rasp_param:+_rasp}_mqttsn_stats${cert_indicator}_n${n}_${parallelization_mode}_${sec_mode}"
        else
            filename="${prefix}${rasp_param:+_rasp}_mqttsn_stats${cert_indicator}_n${n}_${sec_mode}"
        fi
    else
        # nosec mode: no algorithm in filename
        if [ -n "$parallelization_mode" ]; then 
            filename="${prefix}${rasp_param:+_rasp}_mqttsn_stats_n${n}_${parallelization_mode}_${sec_mode}"
        else
            filename="${prefix}${rasp_param:+_rasp}_mqttsn_stats_n${n}_${sec_mode}"
        fi
    fi
    
    echo "${filename}${filename_add}"
}

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
        -role)
            role="$2"
            if [[ "$role" != "pub" && "$role" != "sub" ]]; then
                echo "Error: -role must be 'pub' or 'sub'"
                usage
            fi
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

# Select client binary based on role
if [ "$role" == "pub" ]; then
    CLIENT_BIN="${CLIENTS_BIN}/sn-pub"
elif [ "$role" == "sub" ]; then
    CLIENT_BIN="${CLIENTS_BIN}/sn-sub"
fi

# Check if client binaries exist
if [ ! -f "$CLIENT_BIN" ]; then
    echo "Error: MQTT-SN client not found at ${CLIENT_BIN}"
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
echo "Client role: $role"
echo "Gateway host: ${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "Parallelization: ${parallelization_mode:-sequential}"
echo "KEM algorithm: ${kem_algorithm:-default}"
echo "Local mode: $LOCAL_MODE"
echo "Energy monitoring: ${MEASURE_ENERGY:-false}"
echo "=============================================="

# Create data directory if needed
mkdir -p "$DATA_DIR"

# Clear previous timing data
rm -f "${DATA_DIR}/time_output.txt"

# Start energy monitoring if enabled
if [ "${MEASURE_ENERGY:-false}" == "true" ]; then
    start_energy_monitoring
fi

# Start packet capture
echo "Starting packet capture on interface: $capture_interface"
tshark -i "$capture_interface" -w "${DATA_DIR}/udp_conversations.pcapng" \
    -f "udp port ${GATEWAY_PORT}" &
tshark_pid=$!
sleep 1

# Build client command
build_client_cmd() {
    local client_id=$1
    local cmd="$CLIENT_BIN"
    
    cmd="$cmd -h ${GATEWAY_HOST} -p ${GATEWAY_PORT}"
    
    if [ "$sec_mode" == "pki" ]; then
        cmd="$cmd -A ${ca_file}"  # CA certificate
    fi
    
    echo "$cmd"
}

# Record start time
start_time=$(date +%s.%N)

# Run clients based on parallelization mode
echo "Starting $n MQTT-SN $role clients..."

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
            export CLIENT_BIN GATEWAY_HOST GATEWAY_PORT sec_mode ca_file kem_algorithm
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

# Stop energy monitoring if it was started
if [ "${MEASURE_ENERGY:-false}" == "true" ]; then
    stop_energy_monitoring
fi

# Generate filename for results
filename=$(generate_filenames "udp" false)

# Extract per-conversation UDP stats from the pcap (mirrors coap_benchmark.sh)
# Filter for the gateway host so we only keep flows to/from the MQTT-SN gateway.
rm -f "${DATA_DIR}/${filename}.txt"
if [ -n "$rasp_param" ]; then
    tshark -r "${DATA_DIR}/udp_conversations.pcapng" -z conv,udp 2>/dev/null \
        | grep "<-> $server_ip" > "${DATA_DIR}/${filename}.txt" || true
else
    # Local case: match both IPv4 (127.0.0.1:GATEWAY_PORT) and IPv6 loopback (::1:GATEWAY_PORT)
    tshark -r "${DATA_DIR}/udp_conversations.pcapng" -z conv,udp 2>/dev/null \
        | grep -E "127\.0\.0\.1:${GATEWAY_PORT}|::1:${GATEWAY_PORT}" > "${DATA_DIR}/${filename}.txt" || true
fi

# Collect CPU cycles measured by perf on the gateway side
cpu_cycles=0
if [ -n "$rasp_param" ] && [ -n "$server_ip" ]; then
    cpu_cycles=$(ssh root@"$server_ip" "awk '/cycles/ {print \$1}' ~/libcoap-pqc-bench/benchmark/data/current/auxiliary_server.txt" 2>/dev/null || echo 0)
elif [ -f "${DATA_DIR}/auxiliary_server.txt" ]; then
    cpu_cycles=$(awk '/cycles/ {print $1}' "${DATA_DIR}/auxiliary_server.txt" 2>/dev/null || echo 0)
fi
# Normalize thousands separators
cpu_cycles=$(echo "$cpu_cycles" | tr -d ',' | sed 's/\.//g' | tr -d ' ')
[ -z "$cpu_cycles" ] && cpu_cycles=0
echo "$cpu_cycles" > "${DATA_DIR}/cycles_output.txt"

# Save timing data
timing_file="${DATA_DIR}/timing_${sec_mode}_${cert_config}_${n}clients_${role}.txt"
echo "duration=${duration}" > "$timing_file"
echo "clients=$n" >> "$timing_file"
echo "sec_mode=$sec_mode" >> "$timing_file"
echo "role=$role" >> "$timing_file"
echo "kem_algorithm=$kem_algorithm" >> "$timing_file"
echo "cert_config=$cert_config" >> "$timing_file"

# Process results with bench-data-manager.py if timing data is available
if [ -f "${DATA_DIR}/time_output.txt" ]; then
    # Process metrics with the unified benchmark data manager
    python3 "${BENCH_DIR}/bench-data-manager.py" process \
        --stats-file "${DATA_DIR}/${filename}.txt" \
        --time-file "${DATA_DIR}/time_output.txt" \
        --cycles-file "${DATA_DIR}/cycles_output.txt" \
        --output-file "${DATA_DIR}/${filename}.csv" 2>/dev/null || {
            echo "Note: bench-data-manager.py processing skipped (some files may be missing)"
        }

    # Add energy data to the CSV file if energy monitoring was enabled
    if [ "${MEASURE_ENERGY:-false}" == "true" ] && [ -e "${DATA_DIR}/${filename}.csv" ]; then
        # Find the energy measurements file
        energy_file="${DATA_DIR}/energy_${energy_filename}.csv"
        if [ -e "$energy_file" ]; then
            echo "Adding energy data from $energy_file to ${DATA_DIR}/${filename}.csv"
            python3 "${BENCH_DIR}/bench-data-manager.py" merge \
                --energy-file "$energy_file" \
                --benchmark-file "${DATA_DIR}/${filename}.csv" 2>/dev/null || true
        fi
    fi
fi

echo "Results saved to: $DATA_DIR"
echo "Timing file: $timing_file"
echo "MQTT-SN benchmark completed successfully."
