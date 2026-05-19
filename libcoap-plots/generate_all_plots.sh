#!/bin/bash
# Generate all plots for all 4 benchmark sessions
# Run from libcoap-plots/ directory with venv activated

set -e

ALGS="KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,P256_KYBER_LEVEL1,P384_KYBER_LEVEL3,P521_KYBER_LEVEL5,P256,P384,P521,X25519"
CERTS="RSA_2048,EC_P256,EC_ED25519,DILITHIUM_LEVEL2,DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,FALCON_LEVEL1,FALCON_LEVEL5"
SESSIONS="local_1219_fiducial_pw local_1219_smart-home_st local_1219_smart-factory_0w local_1222_public-transport_2e"

COMMON="--algorithms $ALGS --cert-types $CERTS --p parallel"

for SESSION in $SESSIONS; do
    echo ""
    echo "=============================="
    echo "Session: $SESSION"
    echo "=============================="

    # Barplots (scenarios A,C)
    for METRIC in "duration ms" "cpu_cycles" "total_bytes" "total_frames"; do
        echo "  Barplot: $METRIC"
        python bench-data-plots.py "$METRIC" 25 --barplot --scenarios A,C \
            $COMMON --custom-suffix "$SESSION" 2>&1 | grep -E "Plot saved|Warning: Could not find"
    done

    # Scatter plots (scenario A and C)
    for SCENARIO in A C; do
        for METRIC in "duration ms" "cpu_cycles"; do
            echo "  Scatter: $METRIC scenario $SCENARIO"
            python bench-data-plots.py "$METRIC" 25 --scatter --scenarios "$SCENARIO" \
                $COMMON --custom-suffix "$SESSION" 2>&1 | grep -E "Plot saved|Warning: Could not find"
        done
    done

    # Heatmaps (scenario A and C)
    for SCENARIO in A C; do
        for METRIC in "duration ms" "cpu_cycles"; do
            echo "  Heatmap: $METRIC scenario $SCENARIO"
            python bench-data-plots.py "$METRIC" 25 --heatmap --scenarios "$SCENARIO" \
                $COMMON --custom-suffix "$SESSION" 2>&1 | grep -E "Plot saved|Warning: Could not find"
        done
    done

    # Boxplots (scenario A)
    for METRIC in "duration ms" "cpu_cycles"; do
        echo "  Boxplot: $METRIC scenario A"
        python bench-data-plots.py "$METRIC" 25 --boxplot --scenarios A \
            $COMMON --custom-suffix "$SESSION" 2>&1 | grep -E "Plot saved|Warning: Could not find|No valid"
    done

    # Candlestick plots (scenarios A and C)
    for SCENARIO in A C; do
        for METRIC in "total_bytes" "total_frames"; do
            echo "  Candlestick: $METRIC scenario $SCENARIO"
            python bench-data-plots.py "$METRIC" 25 --candlestick --scenarios "$SCENARIO" \
                $COMMON --custom-suffix "$SESSION" 2>&1 | grep -E "Plot saved|Warning: Could not find|No valid"
        done
    done

done

echo ""
echo "All plots generated."
