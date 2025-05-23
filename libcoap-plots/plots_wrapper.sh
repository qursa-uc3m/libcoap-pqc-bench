#!/bin/bash

# Usage: ./plot_all_networks.sh <metric> <plot_type> <scenario>

METRICS_STR="$1"
PLOT_TYPE="$2"
SCENARIO="${3:-A}"  # Default to A if not provided
DIR="bench-data-pll"

# Networks to process
#NETWORKS=("fiducial" "smarthome" "smartfactory" "publictransport")
NETWORKS=("fiducial" "smarthome")
# CONVERT STRING TO ARRAY
IFS=',' read -ra METRICS <<< "$METRICS_STR"

# Fixed parameters
N=25

for NETWORK in "${NETWORKS[@]}"; do
    echo "Generating $PLOT_TYPE for $NETWORK..."
    for METRIC in "${METRICS[@]}"; do
        MPLBACKEND=Agg python bench-data-plots.py "$METRIC" $N \
            --$PLOT_TYPE \
            --scenarios "$SCENARIO" \
            --p "parallel" \
            --rasp \
            --custom-suffix "$NETWORK" \
            --data-dir "$DIR"
    done
done