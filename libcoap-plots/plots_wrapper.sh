#!/bin/bash

# Usage: ./plots_wrapper.sh <metrics> <plot_type> <scenario> [--data-dir DIR] [--session SESSION] [--local] [--filtered] [-n N]

METRICS_STR="$1"
PLOT_TYPE="$2"
SCENARIO="${3:-A}"

# Default directory (relative to libcoap-plots)
DATA_DIR="../benchmark/data"
SESSION=""
LOCAL_MODE="false"
N=""  # Will be auto-detected if not provided
PARALLELIZATION=""
LATEX_FLAG=""

# Parse optional named arguments
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --local) LOCAL_MODE="true"; shift ;;
        --filtered) FILTERING="--filtered"; shift ;;
        --no-latex) LATEX_FLAG="--no-latex"; shift ;;
        --parallelization) PARALLELIZATION="$2"; shift 2 ;;
        --parallel) PARALLELIZATION="parallel"; shift ;;
        -n) N="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Matplotlib backend
BACKEND=""

# Convert string to array
IFS=',' read -ra METRICS <<< "$METRICS_STR"

# Function to auto-detect N from aggregated CSV filenames
auto_detect_n() {
    local data_path="$1"
    local session="$2"
    local agg_dir="${data_path}/aggregated/${session}"
    
    if [ -d "$agg_dir" ]; then
        # Find a CSV file and extract N from filename pattern: *_n{N}_*
        local sample_file=$(ls "$agg_dir"/*.csv 2>/dev/null | head -1)
        if [ -n "$sample_file" ]; then
            # Extract N from pattern like "_n1_" or "_n25_"
            local detected_n=$(basename "$sample_file" | grep -oP '_n\K[0-9]+(?=_)')
            if [ -n "$detected_n" ]; then
                echo "$detected_n"
                return 0
            fi
        fi
    fi
    P_FLAG=""
    if [ -n "$PARALLELIZATION" ]; then
        P_FLAG="--p $PARALLELIZATION"
    fi
    echo "1"  # Default fallback
    return 1
}

if [ -n "$SESSION" ]; then
    # Auto-detect N if not provided
    if [ -z "$N" ]; then
        N=$(auto_detect_n "$DATA_DIR" "$SESSION")
        echo "Auto-detected N=$N from filenames"
    else
        echo "Using provided N=$N"
    fi
    
    # Determine if we should use --rasp flag (only for remote/Raspberry Pi benchmarks)
    if [ "$LOCAL_MODE" = "true" ]; then
        RASP_FLAG=""
        echo "Processing session $SESSION (LOCAL mode - no rasp prefix)..."
    else
        RASP_FLAG="--rasp"
        echo "Processing session $SESSION (REMOTE mode - with rasp prefix)..."
    fi
    
    # Single session mode
    for METRIC in "${METRICS[@]}"; do
        MPLBACKEND="$BACKEND" python bench-data-plots.py "$METRIC" $N \
            --$PLOT_TYPE \
            --scenarios "$SCENARIO" \
            $RASP_FLAG \
            --custom-suffix "$SESSION" \
            --data-dir "$DATA_DIR" \
            $P_FLAG \
            $LATEX_FLAG \
            $FILTERING
    done
else
    echo "Error: --session is required"
    echo "Usage: ./plots_wrapper.sh <metrics> <plot_type> <scenario> --session SESSION_ID [--local] [--filtered] [--parallel|--parallelization MODE] [-n N]"
    echo ""
    echo "Options:"
    echo "  --session SESSION_ID  Session identifier (required)"
    echo "  --local               Use for local benchmarks (no rasp prefix in filenames)"
    echo "  --filtered            Use filtered data files"
    echo "  --no-latex            Disable LaTeX text rendering"
    echo "  --parallel            Look for files with _parallel in their names"
    echo "  --parallelization MODE  Look for files with _MODE in their names"
    echo "  --data-dir DIR        Data directory (default: ../benchmark/data)"
    echo "  -n N                  Number of clients (auto-detected from filenames if not provided)"
    exit 1
fi