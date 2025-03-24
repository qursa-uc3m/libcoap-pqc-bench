#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 csv_file_path_1 csv_file_path_2"
    exit 1
fi

# Assign command-line arguments to variables
csv_file_path_1="$1"
csv_file_path_2="$2"

# Calculate initial time
initial_time=$(sed -n '1p' "./libcoap-bench/bench-data/initial_and_final_time.txt")
# initial_time=$(tshark -r ./libcoap-bench/bench-data/udp_conversations.pcapng -c 1 -T fields -e frame.time | awk '{print $4}' | sed 's/\..*//')
if [ -z "$initial_time" ]; then
    echo "Error: Failed to get initial time"
    exit 1
fi

# Calculate final time
final_time=$(sed -n '2p' "./libcoap-bench/bench-data/initial_and_final_time.txt")
# final_time=$(tshark -r ./libcoap-bench/bench-data/udp_conversations.pcapng -T fields -e frame.time | tail -n 1 | awk '{print $4}' | sed 's/\..*//')
if [ -z "$final_time" ]; then
    echo "Error: Failed to get final time"
    exit 1
fi

# Run the Python script with the calculated times
python3 ./libcoap-bench/energy_reader.py "$csv_file_path_2" "$csv_file_path_1" "$initial_time" "$final_time"

