#!/bin/bash

# Usage: ./plots_wrapper.sh <metrics> <plot_type> <scenario> [--data-dir DIR] [--session SESSION]

METRICS_STR="$1"
PLOT_TYPE="$2"
SCENARIO="${3:-A}"

# Default directory (relative to libcoap-plots)
DATA_DIR="../libcoap-bench/data"
SESSION=""

# Parse optional named arguments
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --filtered) FILTERING="--filtered"; shift ;;
        *) shift ;;
    esac
done

# Matplotlib backend
BACKEND=""

# Convert string to array
IFS=',' read -ra METRICS <<< "$METRICS_STR"

# Fixed parameters
N=25

if [ -n "$SESSION" ]; then
    # Single session mode
    echo "Processing session $SESSION..."
    for METRIC in "${METRICS[@]}"; do
        MPLBACKEND="$BACKEND" python bench-data-plots.py "$METRIC" $N \
            --$PLOT_TYPE \
            --scenarios "$SCENARIO" \
            --rasp \
            --custom-suffix "$SESSION" \
            --data-dir "$DATA_DIR" \
            --p "parallel" \
            $FILTERING
    done
else
    echo "Error: --session is required"
    echo "Usage: ./plots_wrapper.sh <metrics> <plot_type> <scenario> --session SESSION_ID"
    exit 1
fi