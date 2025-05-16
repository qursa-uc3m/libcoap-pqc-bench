#!/bin/bash

# ==============================================
# run_benchmarks.sh
# Automated benchmarking script for libcoap with PQC support
# ==============================================

# Script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Source certificate configuration
source "${REPO_ROOT}/certs/config_certs.sh"

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
PAUSE_BETWEEN_RUNS=10
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

# Benchmark data directories
BENCH_DATA_DIR="${REPO_ROOT}/libcoap-bench/bench-data"

# ==============================================
# Function declarations
# ==============================================

# Display help information
show_help() {
    echo -e "${BLUE}Benchmark Automation Script for libcoap PQC${NC}"
    echo
    echo "Usage: $0 -n NUM_CLIENTS [OPTIONS]"
    echo
    echo "Required arguments:"
    echo "  -n NUM_CLIENTS        Number of clients for benchmarking"
    echo
    echo "Optional arguments:"
    echo "  -s TIME               Time for observer mode in seconds (enables observer mode)"
    echo "  -parallelization MODE Parallelization mode [background|parallel] (default: none)"
    echo "                        'background': clients run in the same core"
    echo "                        'parallel': clients run across different cores"
    echo "  -client-auth MODE     Client authentication mode [yes|no] (default: no)"
    echo "  -pause SECONDS        Seconds to pause between benchmark runs (default: 10)"
    echo "  -energy               Enable energy measurements (requires RD-USB setup)"
    echo "  -cert-filter PATTERN  Only run certificate configs matching pattern (comma-separated)"
    echo "  -security MODES       Security modes to test (comma-separated: pki,psk,nosec)"
    echo "  -resources RES        Resources to test (comma-separated: time,async or async?2,example_data)"
    echo "                        For async, you can specify delay with async?N where N is seconds"
    echo "  -async-delay SECONDS  Set delay for async resource (alternative to async?N syntax)"
    echo "  -iterations N         Run each test configuration N times (enables iteration mode)"
    echo "  -y                    Skip confirmation prompts"
    echo "  -v                    Verbose output"
    echo "  -h, --help            Show this help message"
    echo
    echo "Examples:"
    echo "  $0 -n 100"
    echo "  $0 -n 50 -s 30 -parallelization parallel -client-auth yes"
    echo "  $0 -n 20 -cert-filter DILITHIUM -security pki,psk -energy"
    echo "  $0 -n 10 -resources time -v"
    echo "  $0 -n 5 -resources async?3 -pause 30"
    echo "  $0 -n 25 -iterations 10 -security psk -resources async -cert-filter KYBER_LEVEL3"
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
    log "INFO" "Checking dependencies..."
    
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

# Setup directory for a new iteration
setup_iteration_directory() {
    local iteration=$1
    
    # Create fresh bench-data directory for the new iteration
    # If it exists but has content, warn the user
    if [ -d "$BENCH_DATA_DIR" ] && [ "$(ls -A $BENCH_DATA_DIR)" ]; then
        log "WARNING" "Bench data directory already contains files. These will be included in iteration ${iteration}."
    else
        mkdir -p "$BENCH_DATA_DIR"
    fi
    
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

# Finalize an iteration by renaming the directory
finalize_iteration_directory() {
    local iteration=$1
    local target_dir="${BENCH_DATA_DIR}-${SESSION_ID}-${iteration}"

    # First organize energy data into subdirectory
    if [ "$MEASURE_ENERGY" == "true" ]; then
        echo "Organizing energy data for iteration ${iteration}..."
        organize_energy_data "$BENCH_DATA_DIR"
    fi
    
    # If bench-data exists and has content, move it to the iteration-specific directory
    if [ -d "$BENCH_DATA_DIR" ] && [ "$(ls -A $BENCH_DATA_DIR)" ]; then
        log "INFO" "Moving iteration ${iteration} data to ${target_dir}"
        mv "$BENCH_DATA_DIR" "$target_dir"
    else
        log "WARNING" "No data found in ${BENCH_DATA_DIR} for iteration ${iteration}"
        # Create empty directory as placeholder
        mkdir -p "$target_dir"
    fi
    
    # Create a fresh empty bench-data directory for the next iteration or future use
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
    cmd_args="-n $NUM_CLIENTS -sec-mode $sec_mode -r $resource -rasp"
    
    # Add resource-specific arguments
    if [ "$resource" == "time" ]; then
        cmd_args="$cmd_args -confirm $confirm"
    elif [ "$resource" == "async" ] && [ -n "$delay_param" ]; then
        # For async with delay parameter, modify the resource
        cmd_args=$(echo "$cmd_args" | sed "s/-r async/-r async?$delay_param/")
    fi
    
    # Add optional arguments
    if [ -n "$OBSERVE_TIME" ]; then
        cmd_args="$cmd_args -s $OBSERVE_TIME"
    fi

    if  [ -n "$PARALLELIZATION" ]; then
        cmd_args="$cmd_args -parallelization $PARALLELIZATION"
    fi
    
    # Add certificate config for PKI mode
    if [ "$sec_mode" == "pki" ] && [ -n "$cert_config" ]; then
        cmd_args="$cmd_args -cert-config $cert_config -client-auth $CLIENT_AUTH"
    fi
    
    # Set environment variable for energy measurements
    if [ "$MEASURE_ENERGY" == "true" ]; then
        export MEASURE_ENERGY=true
    fi
    
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
            sleep 5  # Short pause before retry
        fi
        
        log "INFO" "Executing: ${REPO_ROOT}/libcoap-bench/coap_benchmark.sh $cmd_args"
        if [ "$VERBOSE" == "true" ]; then
            ${REPO_ROOT}/libcoap-bench/coap_benchmark.sh $cmd_args
        else
            ${REPO_ROOT}/libcoap-bench/coap_benchmark.sh $cmd_args > /tmp/benchmark_output.log 2>&1
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
        local extra_pause=$((PAUSE_BETWEEN_RUNS * 2))
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
    local output_file="${REPO_ROOT}/libcoap-bench/benchmark_summary_${SESSION_ID}.txt"
    
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
    if [ $ITERATIONS -gt 1 ]; then 
        echo "- Iterations per test: $ITERATIONS" >> "$output_file"
        echo "- Iteration directories:" >> "$output_file"
        for ((i=1; i<=ITERATIONS; i++)); do
            echo "  - ${BENCH_DATA_DIR}-${SESSION_ID}-${i}" >> "$output_file"
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
        find "$BENCH_DATA_DIR" -name "udp_rasp_conv_stats_*.csv" -type f | sort > "$file_list"
        
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
    local summary_file="${REPO_ROOT}/libcoap-bench/bench-sessions.txt"
    
    echo "Session: ${SESSION_ID}" >> "$summary_file"
    echo "Timestamp: $(date)" >> "$summary_file"
    echo "Iterations: ${ITERATIONS}" >> "$summary_file"
    echo "Directories:" >> "$summary_file"
    for ((i=1; i<=ITERATIONS; i++)); do
        echo "  - ${BENCH_DATA_DIR}-${SESSION_ID}-${i}" >> "$summary_file"
    done
    echo "-------------------------------------" >> "$summary_file"
    
    log "INFO" "Created session summary in ${summary_file}"
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

# Generate a unique session ID for this benchmark run
RANDOM_STR=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 2 | head -n 1)
SESSION_ID="$(cat ${REPO_ROOT}/algorithm.txt)_$(date +%m%d)_${RANDOM_STR}"


# ==============================================
# Show configuration and confirm execution
# ==============================================

log "HEADER" "Benchmark Configuration"
log "INFO" "Session ID: $SESSION_ID"
log "INFO" "Number of clients: $NUM_CLIENTS"
log "INFO" "Security modes: $SECURITY_MODES"
log "INFO" "Resources to test: $RESOURCES"
log "INFO" "Parallelization mode: $PARALLELIZATION"
[ -n "$ASYNC_DELAY" ] && log "INFO" "Async delay parameter: $ASYNC_DELAY seconds"

if [ -n "$CERT_CONFIGS_FILTER" ]; then
    log "INFO" "Certificate filter: $CERT_CONFIGS_FILTER"
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

if [ $ITERATIONS -gt 1 ]; then
    log "INFO" "Iteration mode: enabled (${ITERATIONS} iterations per test)"
    log "INFO" "Results will be stored in: ${BENCH_DATA_DIR}-${SESSION_ID}-[1-${ITERATIONS}]"
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

# Iterate through each iteration
for ((iteration=1; iteration<=ITERATIONS; iteration++)); do
    # Setup directory for this iteration
    if [ $ITERATIONS -gt 1 ]; then
        log "HEADER" "Starting Iteration $iteration of $ITERATIONS"
        setup_iteration_directory $iteration
    fi
    
    # Iterate through security modes
    for sec_mode in $SECURITY_MODES; do
        # Setup for each security mode
        log "HEADER" "Starting $sec_mode mode benchmarks"
        
        if [ "$sec_mode" == "pki" ]; then
            # If PKI mode, iterate through certificate configurations
            for cert_config in $cert_configs; do
                # Iterate through requested resources
                for resource_item in "${RESOURCE_ARRAY[@]}"; do
                    # Parse resource to extract name and parameters
                    parsed=$(parse_resource "$resource_item")
                    resource=$(echo "$parsed" | cut -d';' -f1)
                    delay=$(echo "$parsed" | cut -d';' -f2)
                    
                    # Use ASYNC_DELAY if specified and no specific delay in resource
                    if [ "$resource" == "async" ] && [ -z "$delay" ] && [ -n "$ASYNC_DELAY" ]; then
                        delay="$ASYNC_DELAY"
                    fi
                    
                    # Run appropriate tests based on resource type
                    if [ "$resource" == "time" ] || [ "$resource" == "example_data" ]; then
                        # Run scenarioA 
                        run_benchmark "$sec_mode" "$resource" "con" "$cert_config" "$delay" "$iteration"
                        
                        # Run scenarioC 
                        run_benchmark "$sec_mode" "$resource" "non" "$cert_config" "$delay" "$iteration"
                    elif [ "$resource" == "async" ]; then
                        # Run scenarioB 
                        run_benchmark "$sec_mode" "$resource" "" "$cert_config" "$delay" "$iteration"
                    else
                        log "WARNING" "Unknown resource type: $resource, skipping"
                    fi
                done
            done
        else
            # For PSK and NOSEC modes, no certificate configs needed
            # Iterate through requested resources
            for resource_item in "${RESOURCE_ARRAY[@]}"; do
                # Parse resource to extract name and parameters
                parsed=$(parse_resource "$resource_item")
                resource=$(echo "$parsed" | cut -d';' -f1)
                delay=$(echo "$parsed" | cut -d';' -f2)
                
                # Use ASYNC_DELAY if specified and no specific delay in resource
                if [ "$resource" == "async" ] && [ -z "$delay" ] && [ -n "$ASYNC_DELAY" ]; then
                    delay="$ASYNC_DELAY"
                fi
                
                # Run appropriate tests based on resource type
                if [ "$resource" == "time" ] || [ "$resource" == "example_data" ]; then
                    # Run scenarioA 
                    run_benchmark "$sec_mode" "$resource" "con" "" "$delay" "$iteration"
                    
                    # Run scenarioC 
                    run_benchmark "$sec_mode" "$resource" "non" "" "$delay" "$iteration"
                elif [ "$resource" == "async" ]; then
                    # Run scenarioB 
                    run_benchmark "$sec_mode" "$resource" "" "" "$delay" "$iteration"
                else
                    log "WARNING" "Unknown resource type: $resource, skipping"
                fi
            done
        fi
    done
    
    # Finalize this iteration's directory immediately after completion
    if [ $ITERATIONS -gt 1 ]; then
        log "SUCCESS" "Completed iteration $iteration of $ITERATIONS"
        finalize_iteration_directory $iteration
    fi
done

# Create a summary file with all iteration directories if we ran multiple iterations
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

# Create summary report
create_summary_report

if [ $ITERATIONS -gt 1 ]; then
    log "INFO" "Multiple iterations completed. Results stored in:"
    for ((i=1; i<=ITERATIONS; i++)); do
        log "INFO" "  - ${BENCH_DATA_DIR}-${SESSION_ID}-${i}"
    done
    log "INFO" "Use bench-data-manger.py --aggregate --session <SESSION_ID> to aggregate results across iterations."
fi

log "SUCCESS" "All benchmarks completed successfully!"
exit 0