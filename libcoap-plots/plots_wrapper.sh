#!/bin/bash

# Usage: ./plot_all_networks.sh <metric> <plot_type> <scenario>

METRICS_STR="$1"
PLOT_TYPE="$2"
SCENARIO="${3:-A}"  # Default to A if not provided
FILTERING="$4"
DIR="bench-data-pll-15"

# Matplotlib backend
BACKEND=""
#BACKEND="Agg" # Use this to prevent direct plot visualization


# Add "--p "parallel" \" in the loop if analyzing parallel mode data

# Networks to process
#NETWORKS=("fiducial" "smarthome" "smartfactory" "publictransport")
NETWORKS=("fiducial" "smarthome")
#NETWORKS=("smarthome")
# CONVERT STRING TO ARRAY
IFS=',' read -ra METRICS <<< "$METRICS_STR"

# Fixed parameters
N=25

for NETWORK in "${NETWORKS[@]}"; do
    echo "Generating $PLOT_TYPE for $NETWORK..."
    for METRIC in "${METRICS[@]}"; do
        MPLBACKEND="$BACKEND" python bench-data-plots.py "$METRIC" $N \
            --$PLOT_TYPE \
            --scenarios "$SCENARIO" \
            --rasp \
            --custom-suffix "$NETWORK" \
            --data-dir "$DIR" \
            --p "parallel" \
            $FILTERING
    done
done