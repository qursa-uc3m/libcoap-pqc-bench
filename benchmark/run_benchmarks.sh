#!/bin/bash

# ==============================================
# run_benchmarks.sh
# Automated benchmarking script for libcoap with PQC support
# ==============================================

# Script directory and repository root
BENCHMARK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$BENCHMARK_SCRIPT_DIR")"

# Source certificate configuration (note: this will overwrite SCRIPT_DIR, so we use BENCHMARK_SCRIPT_DIR)
source "${REPO_ROOT}/certs/config_certs.sh"

# Load environment configuration
if [ -f "${REPO_ROOT}/config.local.env" ]; then
    source "${REPO_ROOT}/config.local.env"
elif [ -f "${REPO_ROOT}/config.env" ]; then
    source "${REPO_ROOT}/config.env"
fi

# Set defaults for local mode configuration
LOCAL_MODE="${LOCAL_MODE:-false}"
ENERGY_MONITOR_TYPE="${ENERGY_MONITOR_TYPE:-fnirsi}"

# Set timing defaults (can be overridden in config.env or config.local.env)
TIMING_RETRY_LOCAL="${TIMING_RETRY_LOCAL:-2.0}"
TIMING_RETRY_REMOTE="${TIMING_RETRY_REMOTE:-5.0}"
TIMING_PAUSE_BETWEEN_RUNS_LOCAL="${TIMING_PAUSE_BETWEEN_RUNS_LOCAL:-2}"
TIMING_PAUSE_BETWEEN_RUNS_REMOTE="${TIMING_PAUSE_BETWEEN_RUNS_REMOTE:-10}"

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
NUM_CLIENTS=""
OBSERVE_TIME=""
PARALLELIZATION=""
CLIENT_AUTH="no"
# Protocol selection: coap (default) or mqttsn
PROTOCOL="coap"
# Pause between benchmark runs (uses config values, can be overridden with -pause)
if [ "$LOCAL_MODE" = "true" ]; then
    PAUSE_BETWEEN_RUNS=$TIMING_PAUSE_BETWEEN_RUNS_LOCAL
else
    PAUSE_BETWEEN_RUNS=$TIMING_PAUSE_BETWEEN_RUNS_REMOTE
fi
RASP_SERVER="false"
MEASURE_ENERGY="false"
CERT_CONFIGS_FILTER=""
SECURITY_MODES="pki psk nosec"
SKIP_CONFIRM="false"
VERBOSE="false"
MAX_RETRIES=2
RESOURCES="time,async"  # Default resources to test
ASYNC_DELAY=""          # Optional delay parameter for async resource
ITERATIONS=1            # Default to 1 iteration (no iteration mode)
SESSION_ID=""           # Unique identifier for this benchmark session
SCENARIOS="A,B,C"       # Default scenarios: A,B,C for CoAP; pub,sub for MQTT-SN
NETWORK_CONDITION=""    # Network condition label for data organization

# Algorithm configuration (new flags: -groups and -signatures)
GROUPS_LIST="KYBER_LEVEL3"  # Default KEM group
SIGNATURES_LIST="DILITHIUM_LEVEL3"  # Default signature algorithm

# Available algorithm lists for "all" option
ALL_GROUPS="KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,P256_KYBER_LEVEL1,P384_KYBER_LEVEL3,P521_KYBER_LEVEL5,P256,P384,P521,X25519"
ALL_SIGNATURES="RSA_2048,EC_P256,EC_ED25519,DILITHIUM_LEVEL2,DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,FALCON_LEVEL1,FALCON_LEVEL5"

# Benchmark data directories - all data stored under benchmark/data/
DATA_BASE="${BENCHMARK_SCRIPT_DIR}/data"
RAW_DATA_DIR="${DATA_BASE}/raw"
BENCH_DATA_DIR="${DATA_BASE}/current"
export BENCH_DATA_DIR  # Export so coap_benchmark.sh can see it

# ==============================================
# Function declarations
# ==============================================

# Display help information
show_help() {
    echo -e "${BLUE}Benchmark Automation Script for PQC Protocols${NC}"
    echo
    echo "Usage: $0 -n NUM_CLIENTS [OPTIONS]"
    echo
    echo "Required arguments:"
    echo "  -n NUM_CLIENTS        Number of clients for benchmarking"
    echo
    echo "Protocol selection:"
    echo "  -protocol PROTOCOL    Protocol to benchmark [coap|mqttsn] (default: coap)"
    echo "                        coap: CoAP over DTLS (libcoap)"
    echo "                        mqttsn: MQTT-SN over DTLS (paho gateway + wolfMQTT clients)"
    echo
    echo "Optional arguments:"
    echo "  -s TIME               Time for observer mode in seconds (enables observer mode)"
    echo "  -parallelization MODE Parallelization mode [background|parallel] (default: none)"
    echo "                        'background': clients run in the same core"
    echo "                        'parallel': clients run across different cores"
    echo "  -client-auth MODE     Client authentication mode [yes|no] (default: no)"
    echo "  -pause SECONDS        Seconds to pause between benchmark runs (default: 10)"
    echo "  -rasp                 Enable RASP server mode (default: local server)"
    echo "  -energy               Enable energy measurements (requires RD-USB setup)"
    echo "  -cert-filter PATTERN  Only run certificate configs matching pattern (comma-separated)"
    echo "  -security MODES       Security modes to test (comma-separated: pki,psk,nosec)"
    echo "                        Note: MQTT-SN only supports pki,nosec (no PSK)"
    echo "  -resources RES        Resources to test (comma-separated: time,async or async?2,example_data)"
    echo "                        For async, you can specify delay with async?N where N is seconds"
    echo "  -async-delay SECONDS  Set delay for async resource (alternative to async?N syntax)"
    echo "  -scenarios SCENARIOS  Scenarios to run (comma-separated):"
    echo "                        CoAP: A,B,C (A=time+con, B=async, C=time+non)"
    echo "                        MQTT-SN: pub,sub (pub=publisher, sub=subscriber)"
    echo "                        Default: A,B,C (CoAP) or pub (MQTT-SN)"
    echo "  -iterations N         Run each test configuration N times (enables iteration mode)"
    echo "  -network CONDITION    Network condition label (e.g., fiducial, smart-home, smart-factory, public-transport)"
    echo "                        REQUIRED: This label is used in the session ID to identify data"
    echo "  -groups GROUPS        Comma-separated list of KEM groups to test (default: KYBER_LEVEL3)"
    echo "                        Use 'all' for: KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,P256_KYBER_LEVEL1,"
    echo "                                       P384_KYBER_LEVEL3,P521_KYBER_LEVEL5,P256,P384,P521,X25519"
    echo "  -signatures SIGS      Comma-separated list or 'all' to filter certificates by signature algorithm (default: DILITHIUM_LEVEL3)"
    echo "                        Use 'all' for: RSA_2048,EC_P256,EC_ED25519,DILITHIUM_LEVEL2,"
    echo "                                       DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,FALCON_LEVEL1,FALCON_LEVEL5"
    echo "  -y                    Skip confirmation prompts"
    echo "  -v                    Verbose output"
    echo "  -h, --help            Show this help message"
    echo
    echo "CoAP Examples:"
    echo "  $0 -n 100 -protocol coap"
    echo "  $0 -n 50 -s 30 -parallelization parallel -client-auth yes"
    echo "  $0 -n 20 -signatures DILITHIUM_LEVEL3,FALCON_LEVEL1 -security pki,psk -energy"
    echo
    echo "MQTT-SN Examples:"
    echo "  $0 -n 100 -protocol mqttsn -security pki -scenarios pub"
    echo "  $0 -n 50 -protocol mqttsn -scenarios pub,sub -groups KYBER_LEVEL3"
    echo
}

# Log messages with timestamps and colors
log() {
    local level=$1
    local message=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    case "$level" in
        "INFO")
            echo -e "${BLUE}[$timestamp] [INFO] ${message}${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] [SUCCESS] ${message}${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] [WARNING] ${message}${NC}"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] [ERROR] ${message}${NC}"
            ;;
        "HEADER")
            echo -e "\n${CYAN}[$timestamp] ======================${NC}"
            echo -e "${CYAN}[$timestamp] ${message}${NC}"
            echo -e "${CYAN}[$timestamp] ======================${NC}"
            ;;
        *)
            echo -e "[$timestamp] ${message}"
            ;;
    esac
}

# Check if required dependencies are installed
check_dependencies() {
    log "INFO" "Checking dependencies for ${PROTOCOL} protocol..."
    
    # Check for tshark
    if ! command -v tshark &> /dev/null; then
        log "ERROR" "tshark is not installed. Please install it with: sudo apt install tshark"
        return 1
    fi
    
    # Check for parallel (if using parallel mode)
    if [ "$PARALLELIZATION" == "parallel" ] && ! command -v parallel &> /dev/null; then
        log "ERROR" "GNU parallel is not installed but required for parallel mode. Please install with: sudo apt install parallel"
        return 1
    fi
    
    # Protocol-specific checks
    if [ "$PROTOCOL" == "coap" ]; then
        # Check that libcoap is installed/built
        if [ ! -d "${REPO_ROOT}/libcoap" ]; then
            log "ERROR" "libcoap directory not found at ${REPO_ROOT}/libcoap"
            return 1
        fi
        
        if [ ! -x "${REPO_ROOT}/libcoap/build/bin/coap-client" ] || [ ! -x "${REPO_ROOT}/libcoap/build/bin/coap-server" ]; then
            log "ERROR" "libcoap executables not found. Please build libcoap first."
            return 1
        fi
        
        # Check for PSK key if psk security mode is enabled
        if [[ "$SECURITY_MODES" == *"psk"* ]] && [ ! -f "${REPO_ROOT}/pskeys/active_psk.txt" ]; then
            log "WARNING" "No active PSK key found. Please run: ./pskeys/psk_manager.sh activate <key>"
            return 1
        fi
    elif [ "$PROTOCOL" == "mqttsn" ]; then
        # Check that MQTT-SN clients are built
        if [ ! -x "${REPO_ROOT}/pq-mqtt-sn-clients/build/bin/sn-pub" ]; then
            log "ERROR" "MQTT-SN clients not found. Please run: ./scripts/install_mqttsn_clients.sh"
            return 1
        fi
        
        # Check that MQTT-SN Gateway is built
        if [ ! -x "${REPO_ROOT}/paho-mqttsn-gateway/MQTTSNGateway/bin/MQTT-SNGateway" ]; then
            log "ERROR" "MQTT-SN Gateway not found. Please run: ./scripts/install_paho_mqttsn_gateway.sh"
            return 1
        fi
        
        # Check if Mosquitto broker is available
        if ! command -v mosquitto &> /dev/null && ! pgrep -x "mosquitto" > /dev/null; then
            log "WARNING" "Mosquitto broker not found. Please run: ./scripts/install_mosquitto.sh"
            return 1
        fi
        
        # MQTT-SN doesn't support PSK mode - filter it out
        if [[ "$SECURITY_MODES" == *"psk"* ]]; then
            log "WARNING" "MQTT-SN does not support PSK mode. Removing PSK from security modes."
            SECURITY_MODES=$(echo "$SECURITY_MODES" | sed 's/psk//g' | tr -s ' ')
        fi
    fi
    
    # Everything is fine
    log "SUCCESS" "All dependencies are satisfied!"
    return 0
}

# Get available certificate configurations
get_available_cert_configs() {
    # Get list of certificate configs using the list_cert_configs function from config_certs.sh
    local temp_file="${REPO_ROOT}/temp_cert_list.txt"
    list_cert_configs > "$temp_file"
    
    # Parse and filter the certificate configurations
    local cert_configs=()
    local header_found=0
    
    while IFS= read -r line; do
        # Skip until we find the header line
        if [[ $line == "Available certificate configurations:"* ]]; then
            header_found=1
            continue
        fi
        
        # If we're past the header, process the lines
        if [ $header_found -eq 1 ]; then
            # Skip separator dashes
            if [[ $line == "---------------------------------"* ]]; then
                continue
            fi
            
            # Skip empty lines
            if [[ -z "$line" ]]; then
                continue
            fi
            
            # Extract the certificate config name (trim leading spaces)
            config_name=$(echo "$line" | sed 's/^[[:space:]]*//')
            
            # Skip DEFAULT if present
            if [[ "$config_name" == "DEFAULT"* ]]; then
                continue
            fi
            
            # Apply filter if specified
            if [ -n "$CERT_CONFIGS_FILTER" ]; then
                local match=0
                for filter in $(echo "$CERT_CONFIGS_FILTER" | tr ',' ' '); do
                    if [[ "$config_name" == *"$filter"* ]]; then
                        match=1
                        break
                    fi
                done
                if [ $match -eq 1 ]; then
                    cert_configs+=("$config_name")
                fi
            else
                cert_configs+=("$config_name")
            fi
        fi
    done < "$temp_file"
    
    # Remove temporary file
    rm -f "$temp_file"
    
    # Output the result
    echo "${cert_configs[@]}"
}

# Filter certificate configurations by signature algorithm
# Usage: filter_certs_by_signature <cert_configs> <signatures_list>
filter_certs_by_signature() {
    local cert_configs="$1"
    local signatures="$2"
    
    # If no signature filter specified, return all certs
    if [ -z "$signatures" ]; then
        echo "$cert_configs"
        return
    fi
    
    local filtered_certs=()
    
    # Convert signature list to array
    IFS=',' read -ra SIG_ARRAY <<< "$signatures"
    
    # Iterate through each cert config
    for cert in $cert_configs; do
        # Check if cert matches any signature
        for sig in "${SIG_ARRAY[@]}"; do
            if [[ "$cert" == *"$sig"* ]]; then
                filtered_certs+=("$cert")
                break
            fi
        done
    done
    
    echo "${filtered_certs[@]}"
}

# Parse resource string to extract resource and parameters
parse_resource() {
    local resource_str="$1"
    local resource_name=""
    local delay_param=""
    
    # Check if this is an async resource with delay parameter
    if [[ "$resource_str" =~ ^async\?([0-9]+)$ ]]; then
        resource_name="async"
        delay_param="${BASH_REMATCH[1]}"
    else
        resource_name="$resource_str"
    fi
    
    # Output resource_name and delay_param separated by semicolon
    echo "${resource_name};${delay_param}"
}

# Run all selected scenarios for a given security mode and certificate config
# Arguments: sec_mode, cert_config (can be empty), iteration
run_scenarios_for_config() {
    local sec_mode="$1"
    local cert_config="$2"
    local iteration="$3"
    
    # MQTT-SN uses different scenarios (pub, sub) than CoAP (A, B, C)
    if [ "$PROTOCOL" == "mqttsn" ]; then
        # MQTT-SN scenarios: pub and sub
        [[ "$SCENARIOS" == *"pub"* ]] && run_benchmark "$sec_mode" "pub" "" "$cert_config" "" "$iteration"
        [[ "$SCENARIOS" == *"sub"* ]] && run_benchmark "$sec_mode" "sub" "" "$cert_config" "" "$iteration"
        return
    fi
    
    # CoAP scenarios: A, B, C
    for resource_item in "${RESOURCE_ARRAY[@]}"; do
        # Parse resource to extract name and parameters
        local parsed=$(parse_resource "$resource_item")
        local resource=$(echo "$parsed" | cut -d';' -f1)
        local delay=$(echo "$parsed" | cut -d';' -f2)
        
        # Use ASYNC_DELAY if specified and no specific delay in resource
        if [ "$resource" == "async" ] && [ -z "$delay" ] && [ -n "$ASYNC_DELAY" ]; then
            delay="$ASYNC_DELAY"
        fi
        
        # Run scenarios based on resource type
        case "$resource" in
            time)
                # Scenario A: time + confirmable
                [[ "$SCENARIOS" == *"A"* ]] && run_benchmark "$sec_mode" "$resource" "con" "$cert_config" "$delay" "$iteration"
                # Scenario C: time + non-confirmable
                [[ "$SCENARIOS" == *"C"* ]] && run_benchmark "$sec_mode" "$resource" "non" "$cert_config" "$delay" "$iteration"
                ;;
            async|example_data)
                # Scenario B: async/observe
                [[ "$SCENARIOS" == *"B"* ]] && run_benchmark "$sec_mode" "$resource" "" "$cert_config" "$delay" "$iteration"
                ;;
            *)
                log "WARNING" "Unknown resource type: $resource, skipping"
                ;;
        esac
    done
}

# Setup directory for a new iteration
setup_iteration_directory() {
    local iteration=$1
    
    # Ensure base directories exist
    mkdir -p "$RAW_DATA_DIR"
    
    # Clean and recreate current directory for new iteration to avoid mixing data
    if [ -d "$BENCH_DATA_DIR" ] && [ "$(ls -A $BENCH_DATA_DIR)" ]; then
        log "WARNING" "Cleaning existing data in ${BENCH_DATA_DIR} before starting iteration ${iteration}"
        rm -rf "$BENCH_DATA_DIR"/*
    fi
    mkdir -p "$BENCH_DATA_DIR"
    
    # Create a marker file to indicate which iteration this is
    echo "Session: ${SESSION_ID}" > "${BENCH_DATA_DIR}/iteration.txt"
    echo "Iteration: ${iteration}" >> "${BENCH_DATA_DIR}/iteration.txt"
    echo "Timestamp: $(date)" >> "${BENCH_DATA_DIR}/iteration.txt"
    
    log "INFO" "Prepared ${BENCH_DATA_DIR} for iteration ${iteration}"
}

# Function to organize energy data files into energy-data subdirectory
organize_energy_data() {
    local bench_data_dir="$1"
    
    # Create energy-data subdirectory if it doesn't exist
    if [ ! -d "${bench_data_dir}/energy-data" ]; then
        mkdir -p "${bench_data_dir}/energy-data"
    fi
    
    # Find and move all energy data files
    local energy_files=$(find "${bench_data_dir}" -maxdepth 1 -name "energy_*" -type f)
    if [ -n "$energy_files" ]; then
        # Move energy files to the energy-data directory
        find "${bench_data_dir}" -maxdepth 1 -name "energy_*" -type f -exec mv {} "${bench_data_dir}/energy-data/" \;
        echo "Moved energy data files to ${bench_data_dir}/energy-data/"
    else
        echo "No energy data files found in ${bench_data_dir}"
    fi
}

# Finalize an iteration by moving data to the session's iteration folder
finalize_iteration_directory() {
    local iteration=$1
    local target_dir="${SESSION_DIR}/iter_${iteration}"

    # First organize energy data into subdirectory
    if [ "$MEASURE_ENERGY" == "true" ]; then
        echo "Organizing energy data for iteration ${iteration}..."
        organize_energy_data "$BENCH_DATA_DIR"
    fi
    
    # If data/current exists and has content, move it to the iteration-specific directory
    if [ -d "$BENCH_DATA_DIR" ] && [ "$(ls -A $BENCH_DATA_DIR)" ]; then
        log "INFO" "Moving iteration ${iteration} data to ${target_dir}"
        mkdir -p "$target_dir"
        # Move all contents from current to iter_N folder
        mv "$BENCH_DATA_DIR"/* "$target_dir/" 2>/dev/null || true
        # Also move hidden files if any
        mv "$BENCH_DATA_DIR"/.[!.]* "$target_dir/" 2>/dev/null || true
    else
        log "WARNING" "No data found in ${BENCH_DATA_DIR} for iteration ${iteration}"
        # Create empty directory as placeholder
        mkdir -p "$target_dir"
    fi
    
    # Clean up data/current for the next iteration
    rm -rf "$BENCH_DATA_DIR"/*
    mkdir -p "$BENCH_DATA_DIR"
}

# Execute a benchmark run with retries
run_benchmark() {
    local sec_mode=$1
    local resource=$2
    local confirm=$3
    local cert_config=$4
    local delay_param=$5
    local iteration=$6
    local retry_count=0
    local max_retries=$MAX_RETRIES
    local cmd_args=""

    # Construct the common command arguments
    # Protocol-specific: CoAP uses -r <resource>, MQTT-SN uses -role <pub|sub>
    if [ "$PROTOCOL" == "mqttsn" ]; then
        cmd_args="-n $NUM_CLIENTS -sec-mode $sec_mode -role $resource"
    else
        cmd_args="-n $NUM_CLIENTS -sec-mode $sec_mode -r $resource"
    fi

    if [ "$RASP_SERVER" == "true" ]; then
        cmd_args="$cmd_args -rasp"
    else
        # Local server mode - no additional flags needed; benchmark scripts default to local
        log "INFO" "Using local server mode"
    fi

    # Add resource-specific arguments (CoAP only; MQTT-SN has no per-resource flags)
    if [ "$PROTOCOL" != "mqttsn" ]; then
        if [ "$resource" == "time" ]; then
            cmd_args="$cmd_args -confirm $confirm"
        elif [ "$resource" == "async" ] && [ -n "$delay_param" ]; then
            # For async with delay parameter, modify the resource
            cmd_args=$(echo "$cmd_args" | sed "s/-r async/-r async?$delay_param/")
        fi
    fi

    # Add optional arguments (CoAP-only flags)
    if [ "$PROTOCOL" != "mqttsn" ] && [ -n "$OBSERVE_TIME" ]; then
        cmd_args="$cmd_args -s $OBSERVE_TIME"
    fi

    if  [ -n "$PARALLELIZATION" ]; then
        cmd_args="$cmd_args -parallelization $PARALLELIZATION"
    fi

    # Add certificate config for PKI mode
    if [ "$sec_mode" == "pki" ] && [ -n "$cert_config" ]; then
        if [ "$PROTOCOL" == "mqttsn" ]; then
            # MQTT-SN doesn't support -client-auth (server-only DTLS auth)
            cmd_args="$cmd_args -cert-config $cert_config"
        else
            cmd_args="$cmd_args -cert-config $cert_config -client-auth $CLIENT_AUTH"
        fi
    fi
    
    # Set environment variables for energy measurements and local mode
    if [ "$MEASURE_ENERGY" == "true" ]; then
        export MEASURE_ENERGY=true
    fi
    
    # Export local mode settings for child scripts
    export LOCAL_MODE
    export ENERGY_MONITOR_TYPE
    
    # Prepare log message
    local res_display="$resource"
    [ -n "$delay_param" ] && res_display="$resource?$delay_param"
    
    if [ $ITERATIONS -gt 1 ]; then
        log "HEADER" "Running benchmark: $sec_mode / $res_display / $confirm ${cert_config:+/ $cert_config} (Iteration $iteration/$ITERATIONS)"
    else
        log "HEADER" "Running benchmark: $sec_mode / $res_display / $confirm ${cert_config:+/ $cert_config}"
    fi
    
    while [ $retry_count -lt $max_retries ]; do
        if [ $retry_count -gt 0 ]; then
            log "WARNING" "Retry attempt $retry_count of $max_retries"
            # Use configurable retry timing
            if [ "$LOCAL_MODE" = "true" ]; then
                sleep $TIMING_RETRY_LOCAL
            else
                sleep $TIMING_RETRY_REMOTE
            fi
        fi
        
        # Select benchmark script based on protocol
        local benchmark_script
        if [ "$PROTOCOL" == "mqttsn" ]; then
            benchmark_script="${REPO_ROOT}/benchmark/mqttsn_benchmark.sh"
        else
            benchmark_script="${REPO_ROOT}/benchmark/coap_benchmark.sh"
        fi
        
        log "INFO" "Executing: ${benchmark_script} $cmd_args"
        if [ "$VERBOSE" == "true" ]; then
            ${benchmark_script} $cmd_args
        else
            ${benchmark_script} $cmd_args > /tmp/benchmark_output.log 2>&1
        fi
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log "SUCCESS" "Benchmark completed successfully"
            break
        else
            log "ERROR" "Benchmark failed with exit code $exit_code"
            retry_count=$((retry_count + 1))
            
            # If this was the last attempt, fail
            if [ $retry_count -ge $max_retries ]; then
                log "ERROR" "Maximum retry attempts reached. Moving to next benchmark."
                # Save log file for debugging
                local error_log_file="${BENCH_DATA_DIR}/error_log_${sec_mode}_${resource}_${confirm}_${cert_config}_iter${iteration}.log"
                if [ -f "/tmp/benchmark_output.log" ]; then
                    cat /tmp/benchmark_output.log > "$error_log_file"
                    log "INFO" "Error log saved to $error_log_file"
                else
                    log "WARNING" "No output log found to save"
                fi
                break
            fi
        fi
    done
    
    # Clear any zombie processes
    local zombies=$(pgrep -f 'coap-client|coap-server' || true)
    if [ -n "$zombies" ]; then
        log "WARNING" "Clearing zombie processes: $zombies"
        echo "$zombies" | xargs -r sudo kill -9
    fi
    
    # Add extra pause after async tests or if there was a failure
    if [ "$resource" == "async" ] || [ $exit_code -ne 0 ]; then
        local extra_pause=$(echo "$PAUSE_BETWEEN_RUNS * 2" | bc -l)
        log "INFO" "Adding extra pause ($extra_pause seconds) after async test or failure..."
        sleep $extra_pause
    else
        # Regular pause between runs to let system stabilize
        log "INFO" "Pausing for $PAUSE_BETWEEN_RUNS seconds before next run..."
        sleep $PAUSE_BETWEEN_RUNS
    fi
}

# Create a summary report of all benchmark results
create_summary_report() {
    local output_file="${DATA_BASE}/summaries/summary_${SESSION_ID}.txt"
    mkdir -p "${DATA_BASE}/summaries"
    
    log "HEADER" "Creating benchmark summary"
    
    echo "===============================================" > "$output_file"
    echo "      libcoap PQC Benchmark Summary Report     " >> "$output_file"
    echo "===============================================" >> "$output_file"
    echo "Generated: $(date)" >> "$output_file"
    echo "Session ID: ${SESSION_ID}" >> "$output_file"
    echo "" >> "$output_file"
    echo "Benchmark Parameters:" >> "$output_file"
    echo "- Number of clients: $NUM_CLIENTS" >> "$output_file"
    if [ -n "$OBSERVE_TIME" ]; then
        echo "- Observer mode: Yes ($OBSERVE_TIME seconds)" >> "$output_file"
        echo "- Parallelization: $PARALLELIZATION" >> "$output_file"
    else
        echo "- Observer mode: No" >> "$output_file"
    fi
    echo "- Resources tested: $RESOURCES" >> "$output_file"
    [ -n "$ASYNC_DELAY" ] && echo "- Async delay: $ASYNC_DELAY seconds" >> "$output_file"
    echo "- Client authentication: $CLIENT_AUTH" >> "$output_file"
    echo "- Energy measurements: $MEASURE_ENERGY" >> "$output_file"
    echo "- KEM groups tested: $GROUPS_LIST" >> "$output_file"
    echo "- Signature algorithms: $SIGNATURES_LIST" >> "$output_file"
    if [ $ITERATIONS -gt 1 ]; then 
        echo "- Iterations per test: $ITERATIONS" >> "$output_file"
        echo "- Iteration directories:" >> "$output_file"
        for ((i=1; i<=ITERATIONS; i++)); do
            echo "  - raw/${SESSION_ID}-${i}" >> "$output_file"
        done
    fi
    echo "" >> "$output_file"
    
    if [ $ITERATIONS -gt 1 ]; then
        echo "For detailed results, please run the metrics_merge.py --aggregate --session <SESSION_ID>." >> "$output_file"
    else
        echo "Results Summary:" >> "$output_file"
        echo "----------------" >> "$output_file"
        
        # Create a temp file list to avoid subshell issues
        local file_list="/tmp/benchmark_files.txt"
        # Protocol-aware stats filename pattern
        local stats_pattern
        if [ "$PROTOCOL" == "mqttsn" ]; then
            stats_pattern="udp${RASP_SERVER:+_rasp}_mqttsn_stats_*.csv"
            [ "$RASP_SERVER" != "true" ] && stats_pattern="udp_mqttsn_stats_*.csv"
        else
            if [ "$RASP_SERVER" == "true" ]; then
                stats_pattern="udp_rasp_conv_stats_*.csv"
            else
                stats_pattern="udp_conv_stats_*.csv"
            fi
        fi
        find "$BENCH_DATA_DIR" -name "$stats_pattern" -type f | sort > "$file_list"
        
        # Check if any files were found
        if [ ! -s "$file_list" ]; then
            echo "No benchmark results found!" >> "$output_file"
            log "WARNING" "No benchmark result files found in ${BENCH_DATA_DIR}"
            return
        fi
        
        # Process each file
        while read -r file; do
            filename=$(basename "$file")
            
            # Debug output
            log "INFO" "Processing result file: $filename"
            
            # Extract metrics from the CSV file (second-to-last row has mean values)
            # Use tail -2 to get the second-to-last row
            local duration=$(awk -F';' 'NR==2 {print $1}' <(tail -3 "$file") 2>/dev/null || echo "N/A")
            local cycles=$(awk -F';' 'NR==2 {print $2}' <(tail -3 "$file") 2>/dev/null || echo "N/A")
            local energy=""
            
            # Debug info
            log "INFO" "Extracted duration: $duration"
            log "INFO" "Extracted cycles: $cycles"

            # Check if energy data is available
            if grep -q "Energy" "$file"; then
                energy=$(awk -F';' 'NR==2 {print $(NF)}' <(tail -3 "$file") 2>/dev/null || echo "N/A")
                log "INFO" "Extracted energy: $energy"
            fi
            
            # Extract test configuration from filename
            local config=$(echo "$filename" | sed 's/udp_rasp_conv_stats_//; s/.csv//')
            
            # Format the output
            echo "$config:" >> "$output_file"
            echo "  - Avg. Duration: $duration s" >> "$output_file"
            echo "  - CPU Cycles: $cycles" >> "$output_file"
            if [ -n "$energy" ]; then
                echo "  - Energy: $energy Wh" >> "$output_file"
            fi
            echo "" >> "$output_file"
        done < "$file_list"
        
        # Clean up
        rm -f "$file_list"
    fi
    
    log "SUCCESS" "Summary report created at $output_file"
}

# Function to create a summary file with all iteration directories
create_iteration_summary() {
    local summary_file="${DATA_BASE}/sessions.txt"
    
    echo "Session: ${SESSION_ID}" >> "$summary_file"
    echo "Network Condition: ${NETWORK_CONDITION}" >> "$summary_file"
    echo "Timestamp: $(date)" >> "$summary_file"
    echo "Iterations: ${ITERATIONS}" >> "$summary_file"
    echo "Directory: raw/${SESSION_ID}/" >> "$summary_file"
    echo "Iteration folders:" >> "$summary_file"
    for ((i=1; i<=ITERATIONS; i++)); do
        echo "  - iter_${i}/" >> "$summary_file"
    done
    echo "-------------------------------------" >> "$summary_file"
    
    log "INFO" "Created session summary in ${summary_file}"
    
    # Also update session metadata with end time
    if [ -f "$METADATA_FILE" ]; then
        echo "End Time: $(date -Iseconds)" >> "$METADATA_FILE"
        echo "Total Iterations Completed: $ITERATIONS" >> "$METADATA_FILE"
    fi
}

# ==============================================
# Parse command-line arguments
# ==============================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        -s)
            OBSERVE_TIME="$2"
            shift 2
            ;;
        -parallelization)
            PARALLELIZATION="$2"
            shift 2
            ;;
        -client-auth)
            CLIENT_AUTH="$2"
            shift 2
            ;;
        -pause)
            PAUSE_BETWEEN_RUNS="$2"
            shift 2
            ;;
        -rasp)
            RASP_SERVER="true"
            shift
            ;;
        -protocol)
            PROTOCOL="$2"
            if [[ "$PROTOCOL" != "coap" && "$PROTOCOL" != "mqttsn" ]]; then
                log "ERROR" "Protocol must be 'coap' or 'mqttsn'"
                exit 1
            fi
            shift 2
            ;;
        -energy)
            MEASURE_ENERGY="true"
            shift
            ;;
        -cert-filter)
            CERT_CONFIGS_FILTER="$2"
            shift 2
            ;;
        -security)
            SECURITY_MODES=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        -resources)
            RESOURCES="$2"
            shift 2
            ;;
        -async-delay)
            ASYNC_DELAY="$2"
            shift 2
            ;;
        -iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -network)
            NETWORK_CONDITION="$2"
            shift 2
            ;;
        -scenarios)
            SCENARIOS="$2"
            SCENARIOS_USER_SET=1
            shift 2
            ;;
        -groups)
            if [ "$2" = "all" ]; then
                GROUPS_LIST="$ALL_GROUPS"
            else
                GROUPS_LIST="$2"
            fi
            shift 2
            ;;
        -signatures)
            if [ "$2" = "all" ]; then
                SIGNATURES_LIST="$ALL_SIGNATURES"
            else
                SIGNATURES_LIST="$2"
            fi
            shift 2
            ;;
        -y)
            SKIP_CONFIRM="true"
            shift
            ;;
        -v)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ==============================================
# Validate arguments and environment
# ==============================================

# Check required arguments
if [ -z "$NUM_CLIENTS" ]; then
    log "ERROR" "Number of clients (-n) is required"
    show_help
    exit 1
fi

# Validate NUM_CLIENTS is a positive integer
if ! [[ "$NUM_CLIENTS" =~ ^[0-9]+$ ]] || [ "$NUM_CLIENTS" -lt 1 ]; then
    log "ERROR" "Number of clients must be a positive integer"
    exit 1
fi

# Validate OBSERVE_TIME is a positive integer if provided
if [ -n "$OBSERVE_TIME" ] && { ! [[ "$OBSERVE_TIME" =~ ^[0-9]+$ ]] || [ "$OBSERVE_TIME" -lt 1 ]; }; then
    log "ERROR" "Observer time must be a positive integer"
    exit 1
fi

# Validate PARALLELIZATION
if [ -z parallelization ] && [ "$PARALLELIZATION" != "background" ] && [ "$PARALLELIZATION" != "parallel" ]; then
    log "ERROR" "Parallelization must be either 'background' or 'parallel'"
    exit 1
fi

# Validate CLIENT_AUTH
if [ "$CLIENT_AUTH" != "yes" ] && [ "$CLIENT_AUTH" != "no" ]; then
    log "ERROR" "Client authentication must be either 'yes' or 'no'"
    exit 1
fi

# Validate ITERATIONS
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    log "ERROR" "Number of iterations must be a positive integer"
    exit 1
fi

# Apply protocol-aware scenario default
if [ -z "${SCENARIOS_USER_SET:-}" ] && [ "$PROTOCOL" == "mqttsn" ]; then
    SCENARIOS="pub,sub"
fi

# Validate and normalize SCENARIOS
if [ -n "$SCENARIOS" ]; then
    # Remove spaces
    SCENARIOS=$(echo "$SCENARIOS" | tr -d ' ')
    if [ "$PROTOCOL" == "mqttsn" ]; then
        # MQTT-SN scenarios are lowercase pub/sub
        SCENARIOS=$(echo "$SCENARIOS" | tr '[:upper:]' '[:lower:]')
        # Validate that only pub, sub are present
        _valid=1
        IFS=',' read -ra _scn_arr <<< "$SCENARIOS"
        for _s in "${_scn_arr[@]}"; do
            [[ "$_s" != "pub" && "$_s" != "sub" ]] && _valid=0 && break
        done
        if [ $_valid -eq 0 ]; then
            log "ERROR" "Invalid MQTT-SN scenarios. Use comma-separated list of pub, sub"
            log "ERROR" "  pub = publisher client (handshake + publish)"
            log "ERROR" "  sub = subscriber client (handshake + subscribe)"
            exit 1
        fi
    else
        # CoAP scenarios are uppercase A/B/C
        SCENARIOS=$(echo "$SCENARIOS" | tr '[:lower:]' '[:upper:]')
        if ! [[ "$SCENARIOS" =~ ^[A-C,]+$ ]]; then
            log "ERROR" "Invalid scenarios specified. Use comma-separated list of A, B, and/or C"
            log "ERROR" "  A = time+con (handshake test)"
            log "ERROR" "  B = async (separate response)"
            log "ERROR" "  C = time+non (observe mode)"
            exit 1
        fi
    fi
    # Remove duplicates and sort
    SCENARIOS=$(echo "$SCENARIOS" | tr ',' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
fi

# Validate resources
# Check if async?N format is used, and extract the delay parameter
async_with_delay=$(echo "$RESOURCES" | grep -oE 'async\?[0-9]+')
if [ -n "$async_with_delay" ]; then
    # Extract the delay value
    extracted_delay=$(echo "$async_with_delay" | cut -d'?' -f2)
    
    # Only set ASYNC_DELAY if not explicitly set with -async-delay
    if [ -z "$ASYNC_DELAY" ]; then
        ASYNC_DELAY="$extracted_delay"
    fi
    
    # Replace async?N with just async in RESOURCES
    RESOURCES=$(echo "$RESOURCES" | sed 's/async?[0-9]\+/async/g')
fi

# Create benchmark data directory if it doesn't exist
mkdir -p "$BENCH_DATA_DIR"

# Check for required dependencies
if ! check_dependencies; then
    log "ERROR" "Please install required dependencies before running benchmark"
    exit 1
fi

# ==============================================
# Show configuration and confirm execution
# ==============================================

log "HEADER" "Benchmark Configuration"
log "INFO" "Number of clients: $NUM_CLIENTS"
log "INFO" "Security modes: $SECURITY_MODES"
log "INFO" "Resources to test: $RESOURCES"
log "INFO" "Server mode: $([ "$RASP_SERVER" == "true" ] && echo "Raspberry Pi (remote)" || echo "Local")"
log "INFO" "Local mode config: $LOCAL_MODE"
log "INFO" "Parallelization mode: $PARALLELIZATION"
log "INFO" "KEM groups to test: $GROUPS_LIST"
log "INFO" "Signature filter: $SIGNATURES_LIST"
log "INFO" "Scenarios to run: $SCENARIOS (A=time+con, B=async, C=time+non)"
[ -n "$ASYNC_DELAY" ] && log "INFO" "Async delay parameter: $ASYNC_DELAY seconds"

if [ -n "$CERT_CONFIGS_FILTER" ]; then
    log "INFO" "Certificate filter (legacy): $CERT_CONFIGS_FILTER"
fi

if [ -n "$OBSERVE_TIME" ]; then
    log "INFO" "Observer mode enabled: $OBSERVE_TIME seconds"
    log "INFO" "Parallelization mode: $PARALLELIZATION"
else
    log "INFO" "Observer mode: disabled"
fi

log "INFO" "Client authentication: $CLIENT_AUTH"
log "INFO" "Pause between runs: $PAUSE_BETWEEN_RUNS seconds"
log "INFO" "Energy measurements: $MEASURE_ENERGY"
if [ "$MEASURE_ENERGY" == "true" ]; then
    log "INFO" "Energy monitor type: $ENERGY_MONITOR_TYPE"
fi

if [ $ITERATIONS -gt 1 ]; then
    log "INFO" "Iteration mode: enabled (${ITERATIONS} iterations per test)"
fi

# Get available certificate configurations for PKI mode
if [[ "$SECURITY_MODES" == *"pki"* ]]; then
    cert_configs=$(get_available_cert_configs)
    log "INFO" "Available certificate configurations: ${cert_configs[*]}"
fi

# Confirm execution unless -y flag is provided
if [ "$SKIP_CONFIRM" != "true" ]; then
    echo
    read -p "Start benchmarks with these settings? (y/n): " confirm
    if [[ "$confirm" != [yY] && "$confirm" != [yY][eE][sS] ]]; then
        log "INFO" "Benchmark canceled by user"
        exit 0
    fi
fi

# ==============================================
# Execute benchmarks
# ==============================================

log "HEADER" "Starting Benchmark Suite"

# Track start time for overall benchmarks
BENCHMARK_START_TIME=$(date +%s)

# Convert comma-separated resources to array
IFS=',' read -ra RESOURCE_ARRAY <<< "$RESOURCES"

# Convert comma-separated groups (KEM algorithms) to array
IFS=',' read -ra GROUPS_ARRAY <<< "$GROUPS_LIST"

# Get available certificate configurations for PKI mode if needed
if [[ "$SECURITY_MODES" == *"pki"* ]]; then
    cert_configs=$(get_available_cert_configs)
    
    # Apply signature filter if specified
    if [ -n "$SIGNATURES_LIST" ]; then
        cert_configs=$(filter_certs_by_signature "$cert_configs" "$SIGNATURES_LIST")
        log "INFO" "Filtered certificate configurations by signature: ${cert_configs[*]}"
    else
        log "INFO" "Available certificate configurations: ${cert_configs[*]}"
    fi
fi

# Generate one session ID for the entire benchmark run
# Auto-detect network condition from net_config.sh
NET_CONFIG_SCRIPT="${REPO_ROOT}/network_emulation/net_config.sh"
if [ -x "$NET_CONFIG_SCRIPT" ]; then
    NETWORK_CONDITION=$("$NET_CONFIG_SCRIPT" get-current 2>/dev/null)
    if [ -z "$NETWORK_CONDITION" ] || [ "$NETWORK_CONDITION" == "unknown" ]; then
        log "WARNING" "Could not detect network condition, defaulting to 'fiducial'"
        NETWORK_CONDITION="fiducial"
    fi
else
    log "WARNING" "Network config script not found, defaulting to 'fiducial'"
    NETWORK_CONDITION="fiducial"
fi

# Generate session ID: local_MMDD_NETCOND_RANDOM
# Folder structure: raw/local_1219_fiducial_ab/iter_1/, iter_2/, etc.
RANDOM_STR=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 2 | head -n 1)
SESSION_PREFIX="$([ "$RASP_SERVER" == "true" ] && echo "rasp" || echo "local")"
SESSION_ID="${SESSION_PREFIX}_$(date +%m%d)_${NETWORK_CONDITION}_${RANDOM_STR}"
SESSION_DIR="${RAW_DATA_DIR}/${SESSION_ID}"

log "INFO" "Session ID: $SESSION_ID"
log "INFO" "Network condition: $NETWORK_CONDITION (auto-detected)"
log "INFO" "Session directory: $SESSION_DIR"

# Create session directory structure
mkdir -p "$SESSION_DIR"

# Create metadata file for this session
METADATA_FILE="${SESSION_DIR}/session_metadata.txt"
{
    echo "Session ID: $SESSION_ID"
    echo "Network Condition: $NETWORK_CONDITION"
    echo "Start Time: $(date -Iseconds)"
    echo "Number of Clients: $NUM_CLIENTS"
    echo "Security Modes: $SECURITY_MODES"
    echo "Scenarios: $SCENARIOS"
    echo "Iterations: $ITERATIONS"
    echo "Groups: $GROUPS_LIST"
    echo "Signatures: $SIGNATURES_LIST"
    echo "Parallelization: $PARALLELIZATION_MODE"
    echo "Energy Measurement: $MEASURE_ENERGY"
    echo "RASP Server: $RASP_SERVER"
} > "$METADATA_FILE"
log "INFO" "Session metadata saved to: $METADATA_FILE"

# Iterate through each iteration
for ((iteration=1; iteration<=ITERATIONS; iteration++)); do
    log "HEADER" "Starting Iteration $iteration of $ITERATIONS"
    
    # Setup directory for this iteration
    if [ $ITERATIONS -gt 1 ]; then
        setup_iteration_directory $iteration
    fi
    
    # Iterate through security modes
    for sec_mode in $SECURITY_MODES; do
        log "HEADER" "Starting $sec_mode mode benchmarks (Iteration $iteration)"
        
        if [ "$sec_mode" == "pki" ]; then
            # PKI mode: iterate through KEM groups and certificate configurations
            for group in "${GROUPS_ARRAY[@]}"; do
                log "INFO" "Setting KEM group to: $group"
                echo "$group" > "${REPO_ROOT}/algorithm.txt"
                
                for cert_config in $cert_configs; do
                    run_scenarios_for_config "$sec_mode" "$cert_config" "$iteration"
                done
                
                log "SUCCESS" "Completed PKI benchmarks for KEM group $group"
            done
        else
            # PSK and NOSEC modes: no algorithm iteration needed
            echo "N/A" > "${REPO_ROOT}/algorithm.txt"
            run_scenarios_for_config "$sec_mode" "" "$iteration"
        fi
        
        log "SUCCESS" "Completed $sec_mode mode benchmarks (Iteration $iteration)"
    done
    
    # Finalize this iteration's directory
    log "SUCCESS" "Completed iteration $iteration of $ITERATIONS"
    finalize_iteration_directory $iteration
done

# Create iteration summary if multiple iterations were run
if [ $ITERATIONS -gt 1 ]; then
    create_iteration_summary
fi

# Calculate total benchmark duration
BENCHMARK_END_TIME=$(date +%s)
BENCHMARK_DURATION=$((BENCHMARK_END_TIME - BENCHMARK_START_TIME))
HOURS=$((BENCHMARK_DURATION / 3600))
MINUTES=$(( (BENCHMARK_DURATION % 3600) / 60 ))
SECONDS=$((BENCHMARK_DURATION % 60))

log "HEADER" "Benchmark Suite Completed"
log "SUCCESS" "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"

# Create summary report (skip detailed summary for multi-iteration runs)
if [ $ITERATIONS -gt 1 ]; then
    log "INFO" "Multiple iterations completed. Results stored in: ${SESSION_DIR}/"
    log "INFO" "  - Iterations: iter_1/ through iter_${ITERATIONS}/"
    log "INFO" "Session metadata available at: ${SESSION_DIR}/session_metadata.txt"
    log "INFO" "Use bench-data-manager.py --aggregate --session ${SESSION_ID} to aggregate results across iterations."
else
    # Single iteration - create detailed summary
    create_summary_report
fi

log "SUCCESS" "All benchmarks completed successfully!"
exit 0