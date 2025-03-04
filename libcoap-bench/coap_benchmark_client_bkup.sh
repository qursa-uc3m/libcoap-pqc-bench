#!/bin/bash

certs_path="./certs"
custom_param=""
custom_param_value=0
rasp_param=""
parallelization_param=""
r_param=""
confirm_param=""
confirm_flag=""
libcoap_dir="$(pwd)/libcoap"

# Network variables
bridge_interface="br0"
server_ip="192.168.0.157"
client_ip=""

sudo rm ./time_output.txt

# Function to display usage information
usage() {
    echo "Usage: $0 -n <positive_integer> -sec-mode <pki|psk|nosec> -r <time|async> [-confirm <con|non>] [-s <integer>=1] [-rasp] [-parallelization <background|parallel>]"
    exit 1
}

# Function to clean up and exit on interruption
cleanup() {
    echo "Script interrupted. Cleaning up..."
    kill "$tshark_pid"  # Kill tshark process
    rm -f libcoap-bench/bench-data/udp_conversations.pcapng
    exit 1
}

# Trap interrupt signal (Ctrl+C) to perform cleanup
trap cleanup INT

# Default values for optional parameters
default_parallelization="background"

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
                pki|psk|nosec)
                    sec_mode="$1"
                    ;;
                *)
                    echo "Error: Invalid value for -sec-mode. Use pki, psk, or nosec."
                    usage
                    ;;
            esac
            ;;
        -r)
            shift
            case "$1" in
                time|async)
                    r_param="$1"
                    ;;
                *)
                    echo "Error: Invalid value for -r. Use time or async."
                    usage
                    ;;
            esac
            ;;
        -confirm)
            shift
            case "$1" in
                con|non)
                    confirm_param="$1"
                    if [ "$confirm_param" == "non" ]; then
                        confirm_flag="-N"
                    fi
                    ;;
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
                parallelization_param="-parallelization $default_parallelization"
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
                background|parallel)
                    parallelization_param="-parallelization $1"
                    parallelization_mode="$1"
                    ;;
                *)
                    echo "Error: Invalid value for -parallelization. Use background or parallel."
                    usage
                    ;;
            esac
            ;;
        *)
            echo "Error: Unknown option $1."
            usage
            ;;
    esac
    shift
done

# Check if required parameters are provided
if [ -z "$n" ] || [ -z "$sec_mode" ] || [ -z "$r_param" ]; then
    echo "Error: -n, -sec-mode, and -r parameters are required."
    usage
fi

# Check if -confirm is required and provided
if [ "$r_param" == "time" ] && [ -z "$confirm_param" ]; then
    echo "Error: -confirm parameter is required when -r is set to time."
    usage
fi

# Your script logic using $n, $sec_mode, and $r_param goes here
echo "Parameters:"
echo "  n        : $n"
echo "  sec-mode : $sec_mode"
echo "  r        : $r_param"
[ -n "$confirm_param" ] && echo "  confirm  : $confirm_param"
[ -n "$custom_param" ] && echo "  custom   : $custom_param_value"
[ -n "$rasp_param" ] && echo "  rasp     : enabled"
[ -n "$parallelization_param" ] && echo "  parallelization : $default_parallelization"

# Set the port based on security mode
if [ "$sec_mode" = "nosec" ]; then
    coap_port=5683
else
    coap_port=5684  # For both PSK and PKI modes
fi

# Capture UDP conversations using tshark before the loops and save the PID of the tshark process
if [ -n "$rasp_param" ]; then
    bridge_ip=$(ip addr show $bridge_interface | grep -Po 'inet \K[\d.]+') # Get the IP address of the bridge
    client_ip=$(ip addr show enp3s0 | grep -Po 'inet \K[\d.]+')

    tshark -f "udp port $coap_port and host $bridge_ip" -w libcoap-bench/bench-data/udp_conversations.pcapng -z conv,udp &
    tshark_pid=$!
    address="[$server_ip]"
else
    address="[::1]"
    tshark -i loopback -w libcoap-bench/bench-data/udp_conversations.pcapng -z conv,udp &
    tshark_pid=$!
fi

# Allow time for tshark to start capturing
sleep 1

# Perform some action based on the parameters

sudo rm -f ./libcoap-bench/bench-data/auxiliary.txt

if [ -n "$custom_param" ]; then
# Loop logic if -s parameter is provided
    echo "Optional parameter -s is provided with value $custom_param_value."

    if [ "$sec_mode" == "pki" ]; then
        echo "PKI mode selected. Running coap-client $n times..."
        i=0
        initial_time=$(date +"%T")
        if [ parallelization_param=="-parallelization background" ]; then
            # This parallelizes the clients leaving them in the background
            background_pids=()

            # Launch your background processes and store their PIDs in the array
            for ((i = 1; i <= $n; i++)); do 
                ./libcoap/build/bin/coap-client -m get ${confirm_flag} -s $custom_param_value coaps://${address}/${r_param} >> ./libcoap-bench/bench-data/auxiliary.txt &
                
                # Capture the PID of the last background process launched
                background_pids+=($!)
            done
            
            wait "${background_pids[@]}"
            final_time=$(date +"%T")
        elif [ parallelization_param=="-parallelization parallel" ]; then
            # This parallelizes the clients in different cores
            # Generate functions and add them to an array
            dynamic_commands=()
            for ((i = 1; i <= $n; i++)); do
                dynamic_commands+=("./libcoap/build/bin/coap-client -m get $confirm_flag -s $custom_param_value coaps://$address/$r_param -c "${certs_path}/client_cert.pem" -j "${certs_path}/client_key.pem" >> ./libcoap-bench/bench-data/auxiliary.txt")
            done
            
            # Run dynamically generated commands in parallel
            parallel_pids=()
            parallel -j$n ::: "${dynamic_commands[@]}"&
            parallel_pids+=($!)
            
            # Wait for all background processes spawned by parallel to finish
            wait "${parallel_pids[@]}"
            final_time=$(date +"%T")
        fi
        varalg=$(cat "$libcoap_dir/../algorithm.txt")

    elif [ "$sec_mode" == "psk" ]; then
        echo "PSK mode selected. Running coap-client $n times..."
        i=0
        initial_time=$(date +"%T")
        if [ parallelization_param=="-parallelization background" ]; then
            background_pids=()

            #Launch your background processes and store their PIDs in the array
            for ((i = 1; i <= $n; i++)); do 
            ./libcoap/build/bin/coap-client -m get ${confirm_flag} -s $custom_param_value -k "$(cat psk.txt)" -u uc3m coaps://${address}/${r_param} >> ./libcoap-bench/bench-data/auxiliary.txt & 
        
            # Capture the PID of the last background process launched
            background_pids+=($!)
            done
            
            wait "${background_pids[@]}"
            final_time=$(date +"%T")
           
        elif [ parallelization_param=="-parallelization parallel" ]; then
            # Generate functions and add them to an array
            dynamic_commands=()
            for ((i = 1; i <= $n; i++)); do
                dynamic_commands+=("./libcoap/build/bin/coap-client -m get $confirm_flag -s $custom_param_value -k "$(cat psk.txt)" -u uc3m coaps://$address/$r_param >> ./libcoap-bench/bench-data/auxiliary.txt")
            done

            # Run dynamically generated commands in parallel
            parallel_pids=()
            parallel -j$n ::: "${dynamic_commands[@]}"&
            parallel_pids+=($!)
            
            # Wait for all background processes spawned by parallel to finish
            wait "${parallel_pids[@]}"
            final_time=$(date +"%T")
            
        fi 
        varalg=$(cat "$libcoap_dir/../algorithm.txt")
        
    elif [ "$sec_mode" == "nosec" ]; then
        echo "No security mode selected. Running coap-client $n times..."
        i=0
        initial_time=$(date +"%T")
        if [ parallelization_param=="-parallelization background" ]; then
            background_pids=()

            #Launch your background processes and store their PIDs in the array
            for ((i = 1; i <= $n; i++)); do 
            ./libcoap/build/bin/coap-client -m get ${confirm_flag} -s $custom_param_value coap://${address}/${r_param}  >> ./libcoap-bench/bench-data/auxiliary.txt &
        
            # Capture the PID of the last background process launched
            background_pids+=($!)
            done
            wait "${background_pids[@]}"
            final_time=$(date +"%T")
           
        elif [ parallelization_param=="-parallelization parallel" ]; then
            # Generate functions and add them to an array
            dynamic_commands=()
            for ((i = 1; i <= $n; i++)); do
                dynamic_commands+=("./libcoap/build/bin/coap-client -m get $confirm_flag -s $custom_param_value coap://$address/$r_param  >> ./libcoap-bench/bench-data/auxiliary.txt")
            done

            # Run dynamically generated commands in parallel
            parallel_pids=()
            parallel -j$n ::: "${dynamic_commands[@]}"&
            parallel_pids+=($!)

            # Wait for all background processes spawned by parallel to finish
            wait "${parallel_pids[@]}"
            final_time=$(date +"%T")
            
        fi 

    else
        echo "Invalid security mode selected."
    fi

 
else
    # Loop logic when -s parameter is not provided
    echo "Optional parameter -s is not provided."
    echo "Executing custom logic for -s is not provided."

    if [ "$sec_mode" == "pki" ]; then
        echo "PKI mode selected. Running coap-client $n times..."
        i=0
        initial_time=$(date +"%T")
        while [ $i -lt $n ]; do
            ./libcoap/build/bin/coap-client -m get ${confirm_flag} coaps://${address}/ -c ${certs_path}/client_cert.pem -j ${certs_path}/client_key.pem >> ./libcoap-bench/bench-data/auxiliary.txt       
            let i=i+1
            sleep 0.2
        done
        final_time=$(date +"%T")
        varalg=$(cat "$libcoap_dir/../algorithm.txt")

    elif [ "$sec_mode" == "psk" ]; then
        echo "PSK mode selected. Running coap-client $n times..."
        i=0
        initial_time=$(date +"%T")
        while [ $i -lt $n ]; do
            ./libcoap/build/bin/coap-client -m get ${confirm_flag} -k "$(cat psk.txt)" -u uc3m coaps://${address}/ >> ./libcoap-bench/bench-data/auxiliary.txt
            let i=i+1
            sleep 0.2
        done
        final_time=$(date +"%T")
        varalg=$(cat "$libcoap_dir/../algorithm.txt")
        
    elif [ "$sec_mode" == "nosec" ]; then
        echo "No security mode selected. Running coap-client $n times..."
        i=0
        initial_time=$(date +"%T")
        while [ $i -lt $n ]; do
            ./libcoap/build/bin/coap-client -m get ${confirm_flag} coap://${address}/ >> ./libcoap-bench/bench-data/auxiliary.txt
            let i=i+1
            sleep 0.2
        done
        final_time=$(date +"%T")
    else
        echo "Invalid security mode selected."
    fi
fi

## Closing commands.

if [ -z "$rasp_param" ]; then
    # Closing commands if -rasp parameter is not provided
    # Stop tshark after the loops
    kill -9 $(pidof tshark)

    # Stop the libcoap server
    server_PID=$(ps -e -f | grep "coap-se" | tail -2 | head -1 | awk '{print $2}')
    sudo kill -2 $server_PID


    # Allow time for tshark to finish capturing
    sleep 1

    # Define filename_add
    if [ "$r_param" == "time" ] && [ "$confirm_param" == "con" ]; then
        filename_add="_scenarioA"
    elif [ "$r_param" == "time" ] && [ "$confirm_param" == "non" ]; then
        filename_add="_scenarioC"
    else
        filename_add="_scenarioB"
    fi

    # Check if variable varalg and s are defined
    if [ "$sec_mode" == "pki" ] || [ "$sec_mode" == "psk" ] && [ -n "$custom_param" ]; then
        filename="udp_conv_stats_${varalg}_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}"
    elif [ "$sec_mode" == "nosec" ] && [ -n "$custom_param" ]; then
        filename="udp_conv_stats_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}"
    elif [ "$sec_mode" == "pki" ] || [ "$sec_mode" == "psk" ] && [ -z "$custom_param" ]; then
        filename="udp_conv_stats_${varalg}_n${n}_${sec_mode}"
    elif [ "$sec_mode" == "nosec" ] && [ -z "$custom_param" ]; then
        filename="udp_conv_stats_n${n}_${sec_mode}"
    fi
    filename="$filename$filename_add"

    # Write captured conversations
    rm -f ./libcoap-bench/bench-data/${filename}.txt
    # tshark -r udp_conversations.pcapng -z conv,udp | grep "|" > udp_conv_stats.txt
    tshark -r ./libcoap-bench/bench-data/udp_conversations.pcapng -z conv,udp | grep "::1:" > ./libcoap-bench/bench-data/${filename}.txt

    # Energy testing
 
    # # Create csv
    # cpu_cycles=$(grep "cycles" ./libcoap-bench/bench-data/auxiliary_server.txt | awk '{print $1}')
    # cpu_cycles=$(echo $cpu_cycles| tr -d '.')
    # cpu_cycles=$((cpu_cycles))
    # python3 ./libcoap-bench/stats_extractor.py ./libcoap-bench/bench-data/${filename}.txt ./libcoap-bench/bench-data/${filename}.csv ${cpu_cycles}

else
    # Closing commands if -rasp parameter is provided
    # Stop tshark after the loops
    kill -9 $(pidof tshark)

    # Allow time for tshark to finish capturing
    sleep 1

    # Define filename_add
    if [ "$r_param" == "time" ] && [ "$confirm_param" == "con" ]; then
        filename_add="_scenarioA"
    elif [ "$r_param" == "time" ] && [ "$confirm_param" == "non" ]; then
        filename_add="_scenarioC"
    else
        filename_add="_scenarioB"
    fi


    # Check if variable varalg is defined
    if [ "$sec_mode" == "pki" ] || [ "$sec_mode" == "psk" ] && [ -n "$custom_param" ]; then
        filename="udp_rasp_conv_stats_${varalg}_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}"
    elif [ "$sec_mode" == "nosec" ] && [ -n "$custom_param" ]; then
        filename="udp_rasp_conv_stats_n${n}_s${custom_param_value}_${parallelization_mode}_${sec_mode}"
    elif [ "$sec_mode" == "pki" ] || [ "$sec_mode" == "psk" ] && [ -z "$custom_param" ]; then
        filename="udp_rasp_conv_stats_${varalg}_n${n}_${sec_mode}"
    elif [ "$sec_mode" == "nosec" ] && [ -z "$custom_param" ]; then
        filename="udp_rasp_conv_stats_n${n}_${sec_mode}"
    fi
    filename="$filename$filename_add"

    # Write captured conversations
    rm -f ./libcoap-bench/bench-data/${filename}.txt
    tshark -r ./libcoap-bench/bench-data/udp_conversations.pcapng -z conv,udp | grep "<-> $client_ip" > ./libcoap-bench/bench-data/${filename}.txt

    echo $initial_time > ./libcoap-bench/initial_and_final_time.txt
    echo $final_time >> ./libcoap-bench/initial_and_final_time.txt
    # # Create csv
    RED='\033[0;31m'
    printf "${RED} You have 5 seconds to close the server\n"
    sleep 5
    #scp root@$server_ip:~/libcoap-test/cycles_output.txt /home/vlorenzo/GitLab/libcoap-test/cycles_output.txt
    cpu_cycles=$(ssh root@$server_ip "awk '/cycles/ {print \$1}' ~/libcoap-test/libcoap-bench/bench-data/auxiliary_server.txt")
    cpu_cycles=$((cpu_cycles))
    echo $cpu_cycles > $libcoap_dir/../cycles_output.txt
    
    python3 $(pwd)/libcoap-bench/metrics_extractor.py $(pwd)/libcoap-bench/bench-data/${filename}.csv
    python3 $(pwd)/libcoap-bench/ws_stats_extractor.py $(pwd)/libcoap-bench/bench-data/${filename}.txt $(pwd)/libcoap-bench/bench-data/${filename}_ws.csv
    python3 $(pwd)/libcoap-bench/metrics_merge.py $(pwd)/libcoap-bench/bench-data/${filename}_ws.csv $(pwd)/libcoap-bench/bench-data/${filename}.csv

    sudo rm $(pwd)/libcoap-bench/bench-data/${filename}_ws.csv
    sudo rm $(pwd)/libcoap-bench/bench-data/${filename}.txt

fi
