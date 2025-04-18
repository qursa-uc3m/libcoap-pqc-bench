#!/bin/bash

# Import certificate configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(pwd)/certs/config_certs.sh"
export REPO_ROOT
BENCH_DIR="${REPO_ROOT}/libcoap-bench"
COAP_BIN="${REPO_ROOT}/libcoap/build/bin"
PSK_DIR="${REPO_ROOT}/pskeys"
ACTIVE_PSK="${PSK_DIR}/active_psk.txt"

# Global variables and defaults
bridge_interface="br0"
server_ip="192.168.0.157"
tshark_pid=""

# Default values
n=""
sec_mode=""
r_param=""
confirm_param=""
confirm_flag=""
custom_param=""
custom_param_value=0
rasp_param=""
#parallelization_mode=""
cert_config="DEFAULT"
client_auth="no"  # Default to mutual authentication

# Function to display usage information
usage() {
    echo "Usage: $0 -n <positive_integer> -sec-mode <pki|psk|nosec> -r <time|async> [-confirm <con|non>] [-s <integer>=1] [-rasp] [-parallelization <background|parallel>] [-cert-config <CONFIG>] [-client-auth <yes|no>]"
    echo ""
    echo "Required parameters:"
    echo "  -n <integer>                 Number of clients that will make requests to the server"
    echo "  -sec-mode <pki|psk|nosec>    Security mode to use"
    echo "  -r <time|async>              Resource that the client asks for"
    echo ""
    echo "Optional parameters:"
    echo "  -confirm <con|non>           Whether messages between client and server are confirmable"
    echo "                               Required if -r is set to time"
    echo "  -s <integer>                 Sets clients in observer mode (positive integer required)"
    echo "  -rasp                        Indicates server is running on Raspberry Pi"
    echo "  -parallelization <option>    How clients run:"
    echo "                               'background' : clients run in the same core"
    echo "                               'parallel': clients run in different cores"
    echo "  -cert-config <CONFIG>        Certificate configuration to use (for PKI mode)"
    echo "  -client-auth <yes|no>        Enable/disable client certificate authentication"
    echo "                               Default is 'no' (only server authentication)"
    echo "  -list-certs                  List available certificate configurations"
    echo "  -h, --help                   Show this help message"
    exit 1
}

echo "Creating benchmark data directory in ${BENCH_DIR}/bench-data ..."
mkdir -p ${BENCH_DIR}/bench-data

# Cleanup temp files
sudo rm -f "${BENCH_DIR}/bench-data/time_output.txt"
sudo rm -f "${BENCH_DIR}/bench-data/auxiliary.txt"

# Function to clean up and exit on interruption
cleanup() {
    echo "Script interrupted. Cleaning up..."
    [ -n "$tshark_pid" ] && kill -9 "$tshark_pid" 2>/dev/null
    rm -f "${BENCH_DIR}/bench-data/udp_conversations.pcapng" 2>/dev/null
    exit 1
}

# Trap interrupt signal (Ctrl+C) to perform cleanup
trap cleanup INT

# Function to start energy monitoring with guaranteed initialization using a named pipe
start_energy_monitoring() {
    energy_name="energy_$energy_filename"
    start_sock="${BENCH_DIR}/.energy_monitor_start.sock"
    stop_sock="${BENCH_DIR}/.energy_monitor_stop.sock"
    
    echo "Starting energy monitoring..."
    
    # Remove any existing sockets
    rm -f "$start_sock" "$stop_sock"
    
    # Create sockets for synchronization
    mkfifo "$start_sock" 2>/dev/null
    mkfifo "$stop_sock" 2>/dev/null
    
    # Start energy monitor with sync mechanism
    python ${BENCH_DIR}/energy_monitor.py --force-reset \
           --output "${BENCH_DIR}/bench-data/$energy_name" \
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
    sleep 0.5
    
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
    sleep 1
}


# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        -n)
            shift
            if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null && [ "$1" -gt 0 ]; then
                n="$1"
            else
                echo "Error: -n must be followed by a positive integer."
                usage
            fi
            ;;
        -sec-mode)
            shift
            case "$1" in
                pki|psk|nosec) sec_mode="$1" ;;
                *) 
                    echo "Error: Invalid value for -sec-mode. Use pki, psk, or nosec."
                    usage
                    ;;
            esac
            ;;
        -r)
            shift
            case "$1" in
                time) r_param="$1" ;;
                async*)  # Accept 'async' with any suffix
                    # Extract the base parameter (async)
                    r_param="async"
                    # Check if there's a query parameter (format: async?X)
                    if [[ "$1" == *\?* ]]; then
                        # Extract the query value after the ?
                        query_value="${1#*\?}"
                        # Add the full parameter with query to r_param
                        r_param="$1"
                        echo "Using async with query parameter: $query_value"
                    fi
                    ;;
                *)
                    echo "Error: Invalid value for -r. Use time or async[?value]."
                    usage
                    ;;
            esac
            ;;
        -confirm)
            shift
            case "$1" in
                con) confirm_param="$1"; confirm_flag="" ;;
                non) confirm_param="$1"; confirm_flag="-N" ;;
                *)
                    echo "Error: Invalid value for -confirm. Use con or non."
                    usage
                    ;;
            esac
            ;;
        -s)
            shift
            if [ -n "$1" ] && [ "$1" -eq "$1" ] 2>/dev/null && [ "$1" -ge 1 ]; then
                custom_param="-s"
                custom_param_value="$1"
            else
                echo "Error: -s must be followed by an integer greater than or equal to 1."
                usage
            fi
            ;;
        -rasp)
            rasp_param="-rasp"
            ;;
        -parallelization)
            shift
            case "$1" in
                background|parallel) parallelization_mode="$1" ;;
                *)
                    echo "Error: Invalid value for -parallelization. Use background or parallel."
                    usage
                    ;;
            esac
            ;;
        -cert-config)
            shift
            cert_config="$1"
            ;;
        -client-auth)
            shift
            case "$1" in
                yes|no) client_auth="$1" ;;
                *)
                    echo "Error: Invalid value for -client-auth. Use yes or no."
                    usage
                    ;;
            esac
            ;;
        -list-certs)
            list_cert_configs
            exit 0
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1."
            usage
            ;;
    esac
    shift
done

echo "-----------------------------------------------------------------------------------------"

# Validate required parameters
if [ -z "$n" ] || [ -z "$sec_mode" ] || [ -z "$r_param" ]; then
    echo "Error: -n, -sec-mode, and -r parameters are required."
    usage
fi

# Check if -confirm is required and provided
if [ "$r_param" == "time" ] && [ -z "$confirm_param" ]; then
    echo "Error: -confirm parameter is required when -r is set to time."
    usage
fi

# Get certificate paths if using PKI mode
if [ "$sec_mode" == "pki" ]; then
    if ! validate_cert_files "$cert_config"; then
        echo "Certificate validation failed. Exiting."
        exit 1
    fi
    
    cert_paths=$(get_cert_paths "$cert_config")
    IFS=';' read -r cert_file key_file ca_file <<< "$cert_paths"
    
    echo "Using certificate configuration: $cert_config"
    echo "  Certificate: $cert_file"
    echo "  Key: $key_file"
    echo "  CA: $ca_file"
    echo "  Client Authentication: $client_auth"
fi

# Check for active PSK key when in PSK mode
if [ "$sec_mode" == "psk" ]; then
    if [ ! -f "${ACTIVE_PSK}" ]; then
        echo "Error: No active PSK key found. Please activate a key using:"
        echo "./pskeys/psk_key_manager.sh activate <key_filename>"
        exit 1
    fi
    
    # Get the key name for display purposes
    active_key_value=$(cat "${ACTIVE_PSK}")
    active_key_name=""
        
    for key_file in "${PSK_DIR}"/*.key; do
        if [[ -f "$key_file" ]] && [[ "$(cat "$key_file")" == "$active_key_value" ]]; then
            active_key_name=$(basename "$key_file")
            break
        fi
    done
    
    if [[ -n "$active_key_name" ]]; then
        echo "Using active PSK key: $active_key_name"
    else
        echo "Using active PSK key: Custom key"
    fi
    echo "Key value: $(cat ${ACTIVE_PSK})"
fi

# Read KEM algorithm from first line of algorithm.txt
if [ -f "${REPO_ROOT}/algorithm.txt" ]; then
    kem_algorithm=$(head -n 1 "${REPO_ROOT}/algorithm.txt")
    varalg="${kem_algorithm}"
else
    varalg="UNKNOWN_KEM"
fi        

# Set port based on security mode
coap_port=$([ "$sec_mode" = "nosec" ] && echo "5683" || echo "5684")

# Print parameter summary
echo "Parameters:"
echo "  n        : $n"
echo "  sec-mode : $sec_mode"
echo "  r        : $r_param"
[ -n "$confirm_param" ] && echo "  confirm  : $confirm_param"
[ -n "$custom_param" ] && echo "  custom   : $custom_param_value"
[ -n "$rasp_param" ] && echo "  rasp     : enabled" 
echo "  parallelization : $parallelization_mode"
[ "$sec_mode" == "psk" ] || [ "$sec_mode" == "pki" ] && echo "  kem-algorithm : $varalg"
[ "$sec_mode" == "psk" ] || [ "$sec_mode" == "pki" ] && echo "  client-auth : $client_auth"
[ "$sec_mode" == "pki" ] && echo "  cert-config : $cert_config"

echo "-----------------------------------------------------------------------------------------"

# Set up network configuration based on rasp_param
if [ -n "$rasp_param" ]; then
    bridge_ip=$(ip addr show $bridge_interface | grep -Po 'inet \K[\d.]+') 
    client_ip=$(ip addr show enp0s31f6 | grep -Po 'inet \K[\d.]+')
    address="[$server_ip]"
    echo "Listening on $bridge_interface"
    echo "Bridge IP: $bridge_ip"
    echo "Client IP: $client_ip"
    echo "Server IP: $server_ip"

    tshark -i $bridge_interface -f "udp port $coap_port and host $bridge_ip" -w "${BENCH_DIR}/bench-data/udp_conversations.pcapng" -z conv,udp &
    tshark_pid=$!
else
    address="[::1]"
    tshark -i loopback -w "${BENCH_DIR}/bench-data/udp_conversations.pcapng" -z conv,udp &
    tshark_pid=$!
fi

# Allow time for tshark to start capturing
sleep 2

if [ -n "$rasp_param" ]; then
        # Clean up any lingering coap-server processes first
        echo "-----------------------------------------------------------------------------------------"
        echo "Cleaning up any existing coap-server processes on $server_ip..."
        ssh root@$server_ip "pkill -9 -f 'coap-server' || true"
        
        # Small delay to ensure ports are fully released
        sleep 1

        # Start the server on the Raspberry Pi
        echo "-----------------------------------------------------------------------------------------"
        echo "Starting server on $server_ip..."
        ssh root@$server_ip "cd ~/libcoap-pqc-bench && ./libcoap-bench/coap_benchmark_server.sh -sec-mode $sec_mode -rasp -cert-config $cert_config -client-auth $client_auth" &
        SERVER_SSH_PID=$!
        
        # Give the server time to start
        echo "Give a moment for the server to start..."
        sleep 3
        
fi

# Start energy monitoring if enabled
if [ "${MEASURE_ENERGY:-false}" == "true" ]; then
    # First create the energy_filename so we can use it for energy monitoring
    # Define energy_filename_add based on scenario
    if [ "$r_param" == "time" ] && [ "$confirm_param" == "con" ]; then
        energy_filename_add="_scenarioA"
    elif [ "$r_param" == "time" ] && [ "$confirm_param" == "non" ]; then
        energy_filename_add="_scenarioC"
    else
        energy_filename_add="_scenarioB"
    fi

    # Prepare base energy_filename for results
    if [ "$sec_mode" == "pki" ] || [ "$sec_mode" == "psk" ]; then
        # Add algorithm, cert type, and client auth indicator
        if [ "$sec_mode" == "pki" ]; then
            # Include cert type and optional client auth flag
            cert_indicator="_${cert_config}"
            client_auth_suffix=$([ "$client_auth" == "yes" ] && echo "_client-auth" || echo "")
        else
            cert_indicator=""
            client_auth_suffix=""
        fi
        
        if [ -n "$custom_param" ]; then
            energy_filename="${rasp_param:+rasp}_conv_stats_${varalg}${cert_indicator}_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}${client_auth_suffix}"
        elif [ -n "$parallelization_mode" ]; then
            energy_filename="${rasp_param:+rasp}_conv_stats_${varalg}${cert_indicator}_n${n}_${parallelization_mode}_${sec_mode}${client_auth_suffix}"
        else
            energy_filename="${rasp_param:+rasp}_conv_stats_${varalg}${cert_indicator}_n${n}_${sec_mode}${client_auth_suffix}"
        fi
    else
        if [ -n "$custom_param" ]; then
            energy_filename="${rasp_param:+rasp}_conv_stats_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}"
        elif [ -n "$parallelization_mode" ]; then
            energy_filename="${rasp_param:+rasp}_conv_stats_n${n}_${parallelization_mode}_${sec_mode}"
        else
            energy_filename="${rasp_param:+rasp}_conv_stats_n${n}_${sec_mode}"
        fi
    fi
    energy_filename="${energy_filename}${energy_filename_add}"
    
    # Now start energy monitoring with the correct energy_filename
    echo "-----------------------------------------------------------------------------------------"
    start_energy_monitoring
fi

# Construct the coap client base command based on security mode
client_cmd="${COAP_BIN}/coap-client -m get ${confirm_flag}"

# Add protocol and destination
protocol=$([ "$sec_mode" = "nosec" ] && echo "coap" || echo "coaps")

# Add observer parameters if specified
[ -n "$custom_param" ] && client_cmd="$client_cmd -s $custom_param_value"

# Add security-specific parameters
case "$sec_mode" in
    pki)
        # Add client certificate parameters only if client authentication is enabled
        if [ "$client_auth" = "yes" ]; then
            client_cmd="$client_cmd -c \"${cert_file}\" -j \"${key_file}\""
        fi
        # Always include CA to validate server's certificate
        client_cmd="$client_cmd -C \"${ca_file}\""
        ;;
    psk)
        # Use the active PSK key from the pskeys directory
        client_cmd="$client_cmd -k \"$(cat ${ACTIVE_PSK})\" -u uc3m"
        ;;
    nosec)
        # No additional parameters needed
        ;;
esac

# Add protocol, address and resource
client_cmd="$client_cmd ${protocol}://${address}/${r_param}"

# Add log output redirection
client_cmd="$client_cmd >> ${BENCH_DIR}/bench-data/auxiliary.txt"

# Capture initial time
initial_time=$(date +%s.%N)

# Execute the client commands based on parameters
if [ -n "$custom_param" ]; then
    echo "Running with observer mode (-s $custom_param_value)"
    
    if [ "$parallelization_mode" = "background" ]; then
        # Run clients in background
        background_pids=()
        for ((i = 1; i <= $n; i++)); do
            eval "$client_cmd" &
            background_pids+=($!)
        done
        wait "${background_pids[@]}"
    elif [ "$parallelization_mode" = "parallel" ]; then
        # Run clients in parallel across cores
        dynamic_commands=()
        for ((i = 1; i <= $n; i++)); do
            dynamic_commands+=("$client_cmd")
        done
        
        parallel -j$n ::: "${dynamic_commands[@]}" &
        parallel_pid=$!
        wait $parallel_pid
    else
        echo "Observer mode requires parallelization. Please run with --parallelization <background|parallel>." 
        echo "Exiting ..."
        exit 1
    fi
else
    # Non-observer mode execution
    echo "Running in standard mode (non-observer)"
    
    if [ "$parallelization_mode" = "parallel" ]; then
        # Parallel execution using GNU parallel
        echo "Executing $n clients in parallel..."
        
        # Create an array of identical commands to run in parallel
        dynamic_commands=()
        for ((i = 1; i <= $n; i++)); do
            dynamic_commands+=("$client_cmd")
        done
        
        # Run all commands in parallel
        parallel -j$n ::: "${dynamic_commands[@]}" &
        parallel_pid=$!
        wait $parallel_pid
    elif [ "$parallelization_mode" = "background" ]; then
        # Background execution
        echo "Executing $n clients in background..."
        # Run clients in background
        background_pids=()
        for ((i = 1; i <= $n; i++)); do
            eval "$client_cmd" &
            background_pids+=($!)
        done
        wait "${background_pids[@]}"
    else
        # Sequential execution (default mode)
        echo "Executing $n clients sequentially..."
        for ((i = 1; i < $n; i++)); do
            eval "$client_cmd"
            sleep 0.2
        done
        # Final execution (avoid sleep after last one)
        eval "$client_cmd"
    fi
fi

# Capture final time
final_time=$(date +%s.%N)

# Stop energy monitoring before getting CPU cycles
if [ "${MEASURE_ENERGY:-false}" == "true" ]; then
    echo "-----------------------------------------------------------------------------------------"
    stop_energy_monitoring
fi

# Define filename_add based on scenario
if [ "$r_param" == "time" ] && [ "$confirm_param" == "con" ]; then
    filename_add="_scenarioA"
elif [ "$r_param" == "time" ] && [ "$confirm_param" == "non" ]; then
    filename_add="_scenarioC"
else
    filename_add="_scenarioB"
fi

# Prepare base filename for results
if [ "$sec_mode" == "pki" ] || [ "$sec_mode" == "psk" ]; then
    # Add algorithm, cert type, and client auth indicator
    if [ "$sec_mode" == "pki" ]; then
        # Include cert type and optional client auth flag
        cert_indicator="_${cert_config}"
        client_auth_suffix=$([ "$client_auth" == "yes" ] && echo "_client-auth" || echo "")
    else
        cert_indicator=""
        client_auth_suffix=""
    fi
    
    if [ -n "$custom_param" ]; then
        filename="udp${rasp_param:+_rasp}_conv_stats_${varalg}${cert_indicator}_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}${client_auth_suffix}"
    elif [ -n "$parallelization_mode" ]; then 
        filename="udp${rasp_param:+_rasp}_conv_stats_${varalg}${cert_indicator}_n${n}_${parallelization_mode}_${sec_mode}${client_auth_suffix}"
    else
        filename="udp${rasp_param:+_rasp}_conv_stats_${varalg}${cert_indicator}_n${n}_${sec_mode}${client_auth_suffix}"
    fi
else
    if [ -n "$custom_param" ]; then
        filename="udp${rasp_param:+_rasp}_conv_stats_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}"
    elif [ -n "$parallelization_mode" ]; then 
        filename="udp${rasp_param:+_rasp}_conv_stats_n${n}_${parallelization_mode}_${sec_mode}"
    else
        filename="udp${rasp_param:+_rasp}_conv_stats_n${n}_${sec_mode}"
    fi
fi
filename="${filename}${filename_add}"

# Process the results
if [ -z "$rasp_param" ]; then
    # Process local results
    kill -9 $(pidof tshark) 2>/dev/null
    
    # Stop the libcoap server
    server_PID=$(ps -e -f | grep "coap-se" | tail -2 | head -1 | awk '{print $2}')
    [ -n "$server_PID" ] && sudo kill -2 $server_PID
    
    # Allow time for tshark to finish capturing
    sleep 2
    
    # Write captured conversations
    rm -f "${BENCH_DIR}/bench-data/${filename}.txt"
    tshark -r "${BENCH_DIR}/bench-data/udp_conversations.pcapng" -z conv,udp | grep "::1:" > "${BENCH_DIR}/bench-data/${filename}.txt"
else
    # Process remote (Raspberry Pi) results
    echo "-----------------------------------------------------------------------------------------"
    kill -9 $(pidof tshark) 2>/dev/null
    # Allow time for tshark to finish capturing
    sleep 2

    # Write captured conversations
    rm -f "${BENCH_DIR}/bench-data/${filename}.txt"
    tshark -r "${BENCH_DIR}/bench-data/udp_conversations.pcapng" -z conv,udp | grep "<-> $server_ip" > "${BENCH_DIR}/bench-data/${filename}.txt"
    
    # Save timing information
{
    echo "Started at:  $initial_time"
    echo "Finished at: $final_time"

    # Calculate elapsed time (in seconds)
    elapsed=$(echo "$final_time - $initial_time" | bc)

    echo "Elapsed time: ${elapsed} s"
} > "${BENCH_DIR}/bench-data/initial_and_final_time.txt"

    # Wait for server to be shut down
    echo "Automatically stopping the server on the Raspberry Pi..."
    ssh root@$server_ip "pkill -2 coap-server || pkill -2 -f coap-server" || echo "Warning: Failed to stop server"
    sleep 3
    echo "-----------------------------------------------------------------------------------------" 
    
    # Get CPU cycles from server
    cpu_cycles=$(ssh root@$server_ip "awk '/cycles/ {print \$1}' ~/libcoap-pqc-bench/libcoap-bench/bench-data/auxiliary_server.txt")
    if [ -z "$cpu_cycles" ]; then
        echo "Warning: Could not retrieve CPU cycles from server. Using default value of 0."
        cpu_cycles=0
    else
        cpu_cycles=$((cpu_cycles))
    fi
    echo $cpu_cycles > "${BENCH_DIR}/bench-data/cycles_output.txt"
    
    # Process metrics with the unified benchmark data manager
    python3 "${BENCH_DIR}/bench-data-manager.py" process \
        --stats-file "${BENCH_DIR}/bench-data/${filename}.txt" \
        --time-file "${BENCH_DIR}/bench-data/time_output.txt" \
        --cycles-file "${BENCH_DIR}/bench-data/cycles_output.txt" \
        --output-file "${BENCH_DIR}/bench-data/${filename}.csv"

    # Add energy data to the CSV file if energy monitoring was enabled
    if [ "${MEASURE_ENERGY:-false}" == "true" ] && [ -e "${BENCH_DIR}/bench-data/${filename}.csv" ]; then
        # Find the energy measurements file
        energy_file="${BENCH_DIR}/bench-data/energy_${energy_filename}.csv"
        if [ -e "$energy_file" ]; then
            echo "Adding energy data from $energy_file to ${BENCH_DIR}/bench-data/${filename}.csv"
            python3 "${BENCH_DIR}/bench-data-manager.py" merge \
                --energy-file "$energy_file" \
                --benchmark-file "${BENCH_DIR}/bench-data/${filename}.csv"
        else
            echo "Warning: Energy file $energy_file not found"
        fi
    fi
    
    # Clean up temporary files
    echo "Cleaning up temporary files..."
    #if [ -f "${BENCH_DIR}/bench-data/${filename}.txt" ]; then
    #    sudo rm "${BENCH_DIR}/bench-data/${filename}.txt"
    #fi
    if [ -f "${BENCH_DIR}/bench-data/udp_conversations.pcapng" ]; then
        sudo rm "${BENCH_DIR}/bench-data/udp_conversations.pcapng"
    fi
fi

echo "Benchmark completed successfully: $filename"
exit 0