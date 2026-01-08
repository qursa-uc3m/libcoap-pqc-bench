#!/bin/bash

# Default values
iterations=1
mode="pub"
parallel_mode="sequential"  # Default to sequential

# Parse flags
while getopts "n:m:p" opt; do
    case $opt in
        n) iterations=$OPTARG ;;
        m) mode=$OPTARG ;;
        p) parallel_mode="parallel" ;;
        *) echo "Usage: $0 [-n iterations] [-m pub|sub] [-p (enable parallel)]" >&2
           exit 1 ;;
    esac
done

# Create tests_logs directory if it doesn't exist
mkdir -p ./tests_logs

# Clear log files
> "./tests_logs/out.log"
> "./tests_logs/sn-$mode.log"

if [ "$parallel_mode" = "sequential" ]; then
    # Sequential execution (default behavior)
    for i in $(seq 1 $iterations); do
        a=$(date +%s%3N)
        echo "----------------- START ITERATION $i -----------------" >> "./tests_logs/out.log"
        if [ "$mode" = "pub" ]; then
            echo "RUNNING PUBLISHER"
            ./build/bin/sn-pub -t >> "./tests_logs/out.log" 2>&1
        else
            echo "RUNNING SUBSCRIBER"
            ./build/bin/sn-sub -t >> "./tests_logs/out.log" 2>&1
        fi
        echo "----------------- END ITERATION $i -----------------" >> "./tests_logs/out.log"
        b=$(date +%s%3N)
        resta=$(($b-$a))
        echo "Cliente $i : $resta ms" >> "./tests_logs/sn-$mode.log"
        sleep 0.2  # Small delay between sequential runs
    done
else
    # Parallel execution using background processes
    echo "Running $iterations clients in parallel mode"
    
    # Array to store background process PIDs
    background_pids=()
    
    # Launch all clients in parallel
    for i in $(seq 1 $iterations); do
        (
            a=$(date +%s%3N)
            echo "----------------- START ITERATION $i -----------------" >> "./tests_logs/out.log"
            if [ "$mode" = "pub" ]; then
                echo "RUNNING PUBLISHER $i"
                ./build/bin/sn-pub -t >> "./tests_logs/out.log" 2>&1
            else
                echo "RUNNING SUBSCRIBER $i"
                ./build/bin/sn-sub -t >> "./tests_logs/out.log" 2>&1
            fi
            echo "----------------- END ITERATION $i -----------------" >> "./tests_logs/out.log"
            b=$(date +%s%3N)
            resta=$(($b-$a))
            echo "Cliente $i : $resta ms" >> "./tests_logs/sn-$mode.log"
        ) &
        
        # Store the PID of the background process
        background_pids+=($!)
    done
    
    # Wait for all background processes to complete
    wait "${background_pids[@]}"
fi