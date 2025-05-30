# LibCoAP Plots - CoAP Benchmark Analysis Suite

A comprehensive Python toolkit for analyzing and visualizing CoAP (Constrained Application Protocol) benchmark data across different security configurations, algorithms, and network conditions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Scripts Overview](#scripts-overview)
- [Usage Guide](#usage-guide)
- [Data Format](#data-format)
- [Visualization Types](#visualization-types)
- [Network Comparison](#network-comparison)
- [Data Filtering](#data-filtering)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This toolkit analyzes performance metrics from CoAP benchmarks across:
- **Security Modes**: PKI, PSK, NoSec
- **Post-Quantum Algorithms**: KYBER, DILITHIUM, FALCON
- **Classical Algorithms**: RSA, ECDSA, Ed25519
- **Network Conditions**: Fiducial (baseline), Smart Home, Smart Factory, Public Transport
- **Metrics**: Duration, Energy consumption, CPU cycles, Power usage, Frame/byte counts

## ✨ Features

- **Multiple Visualization Types**: Scatter plots, bar charts, heat maps, box plots, candlestick plots, spider plots
- **Statistical Analysis**: Comprehensive significance testing for network and algorithm differences
- **Data Filtering**: Advanced outlier detection and removal with statistical recalculation
- **Publication-Ready Output**: LaTeX-formatted plots with customizable styling
- **Batch Processing**: Automated analysis across multiple networks and configurations
- **Comparative Analysis**: Cross-network performance impact assessment

## 📦 Installation

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy matplotlib scipy adjustText

# Optional: LaTeX for publication-quality plots
# Ubuntu/Debian:
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra

# macOS:
brew install --cask mactex
```

### Setup

```bash
git clone <repository-url>
cd libcoap-plots
chmod +x plots_wrapper.sh
```

### Directory Structure Examples

**Typical Multi-Network Setup:**
```
bench-data-pll-15/           # Root data directory
├── bench-data-fiducial/     # Baseline network data
│   ├── udp_rasp_conv_stats_KYBER_LEVEL1_RSA_2048_n25_pki_scenarioA.csv
│   ├── udp_rasp_conv_stats_KYBER_LEVEL1_n25_psk_scenarioA.csv
│   └── ...
├── bench-data-smarthome/    # Smart home network data
│   ├── udp_rasp_conv_stats_KYBER_LEVEL1_RSA_2048_n25_pki_scenarioA.csv
│   └── ...
├── plots-fiducial/          # Generated plots for fiducial
│   ├── scatter_rasp_duration_n25_scenarioA.pdf
│   └── ...
└── plots-smarthome/         # Generated plots for smarthome
    ├── scatter_rasp_duration_n25_scenarioA.pdf
    └── ...
```

## 📖 Usage Guide

### Basic Single-Network Analysis

```bash
# Generate scatter plot for duration vs energy (uses default: ./bench-data/ -> ./plots/)
python bench-data-plots.py "duration" 25 --scatter --scenarios A --rasp

# Generate bar plot comparing security modes
python bench-data-plots.py "Energy (Wh)" 25 --barplot --scenarios A,C --rasp

# Box plot for variability analysis
python bench-data-plots.py "cpu_cycles" 25 --boxplot --scenarios A --rasp

# Discrete metrics candlestick plot
python bench-data-plots.py "total_frames" 25 --candlestick --scenarios A --rasp
```

The `--data-dir` and `--custom-suffix` parameters control where the script looks for input data and saves output plots:

```bash
# DEFAULT BEHAVIOR (no custom suffix):
# Input:  ./bench-data/*.csv
# Output: ./plots/*.pdf

# WITH CUSTOM SUFFIX:
# Input:  ./bench-data/bench-data-{suffix}/*.csv  
# Output: ./bench-data/plots-{suffix}/*.pdf

# Examples:
python bench-data-plots.py "duration" 25 --scatter --scenarios A --rasp \
    --custom-suffix "smarthome"
# Input:  ./bench-data/bench-data-smarthome/*.csv
# Output: ./bench-data/plots-smarthome/*.pdf

python bench-data-plots.py "duration" 25 --scatter --scenarios A --rasp \
    --custom-suffix "fiducial" 
# Input:  ./bench-data/bench-data-fiducial/*.csv
# Output: ./bench-data/plots-fiducial/*.pdf

# CUSTOM DATA DIRECTORY:
python bench-data-plots.py "duration" 25 --scatter --scenarios A --rasp \
    --data-dir "experiment-results" --custom-suffix "run1"
# Input:  ./experiment-results/bench-data-run1/*.csv
# Output: ./experiment-results/plots-run1/*.pdf

# NO SUFFIX + CUSTOM DIR:
python bench-data-plots.py "duration" 25 --scatter --scenarios A --rasp \
    --data-dir "my-experiments"
# Input:  ./my-experiments/bench-data/*.csv  
# Output: ./my-experiments/plots/*.pdf
```

There are additional flags for selecting subsets of configurations (`--algorithms`, `--cert-types`) where you can pass a string with the desired elements for the analysis.

### Network Comparison Analysis

```bash
# Load and analyze cross-network data
python bench-data-compare.py

# This script automatically:
# - Loads data from bench-data-pll-15/ directory
# - Generates summary CSVs for continuous and discrete metrics
# - Creates trade-off plots (duration vs energy, cpu_cycles vs energy)
# - Produces spider plots showing network impact
# - Performs statistical significance testing
```

### Data Filtering

Clean up your datasets from outliers.

```bash
# Filter single file
python bench-data-filter.py data.csv

# Filter entire directory
python bench-data-filter.py bench-data/ --file-pattern "*.csv"

# Custom CV threshold
python bench-data-filter.py data.csv --cv-threshold 2.5
```

### Batch Processing

We include an additional utility for plotting multiple metrics for all the available network datasets. 

```bash
# Process multiple metrics across networks
./plots_wrapper.sh "duration,Energy (Wh),cpu_cycles" scatter A

# With filtering enabled
./plots_wrapper.sh "duration,Energy (Wh)" barplot A "--filtered"

# Multiple scenarios
./plots_wrapper.sh "total_frames,total_bytes" candlestick "A,C"
```

## 📊 Data Format

### Expected CSV Structure

The final csv files will contain the summary statistics for each experiment iteration, and the aggregated statistics for the full experiment. The benchmark suite is designed for plotting files with this structure:

```
var_name_col1, var_name_col2, ...
# Iteration 1 data (5 rows)
mean_col1, mean_col2, ...
std_col1, std_col2, ...
mode_col1, mode_col2, ...
above_mode_col1, above_mode_col2, ...
below_mode_col1, below_mode_col2, ...
;;;;;;;;;;;;;;  # Iteration separator
# Iteration 2 data
...

------------;  # Main separator
aggregated_mean_row
aggregated_std_row
mode_row
below_mode_row
above_mode_row
======;  # Min/max separator (for discrete metrics, absolute values across all iterations)
min_row
max_row
range_row
```

### Supported Metrics

**Continuous Metrics**:
- `duration` - Test execution time (seconds/milliseconds)
- `cpu_cycles` - CPU cycles consumed
- `Energy (Wh)` - Energy consumption (Wh/mWh)
- `Power (W)` - Power consumption
- `Max Power (W)` - Peak power consumption

**Discrete Metrics**:
- `total_frames` - Total frame count
- `total_bytes` - Total byte count
- `frames_sent` - Frames transmitted
- `frames_received` - Frames received
- `bytes_sent` - Bytes transmitted
- `bytes_received` - Bytes received

### File Naming Convention

```
udp_[rasp_]conv_stats_[ALGORITHM]_[CERTIFICATE]_n[N]_[parallel_][MODE]_scenario[SCENARIO][_filtered].csv

Examples:
- udp_rasp_conv_stats_KYBER_LEVEL1_RSA_2048_n25_pki_scenarioA.csv
- udp_rasp_conv_stats_KYBER_LEVEL3_n25_parallel_psk_scenarioC_filtered.csv
- udp_rasp_conv_stats_n25_nosec_scenarioA.csv
```

## 📈 Visualization Types

### 1. Scatter Plots (`--scatter`)
- **Purpose**: Algorithm comparison across security modes
- **Best for**: Continuous metrics, single scenario
- **Features**: Error bars, trend lines, security mode color coding

### 2. Bar Plots (`--barplot`)
- **Purpose**: Multi-scenario comparison
- **Best for**: Both continuous and discrete metrics
- **Features**: Pattern coding for certificate families, significance indicators

### 3. Heat Maps (`--heatmap`)
- **Purpose**: Algorithm vs certificate performance matrix
- **Best for**: Continuous metrics, pattern identification
- **Features**: Color intensity mapping, value annotations

### 4. Box Plots (`--boxplot`)
- **Purpose**: Variability analysis across iterations
- **Best for**: Identifying consistency and outliers
- **Features**: Raw data points with jitter, iteration statistics

### 5. Candlestick Plots (`--candlestick`)
- **Purpose**: Discrete metric range visualization
- **Best for**: Count data (frames, bytes)
- **Features**: Min/max ranges, mode indicators, line style coding

### 6. Spider Plots (Compare Script)
- **Purpose**: Multi-dimensional network impact assessment
- **Best for**: Overall performance comparison
- **Features**: Multiple normalization methods, configuration grouping

## 🌐 Network Comparison

### Supported Networks
- **Fiducial**: Baseline laboratory conditions
- **Smart Home**: Residential IoT environment
- **Smart Factory**: Industrial IoT conditions  
- **Public Transport**: Mobile/vehicular network

### Statistical Analysis Features

**Network Difference Analysis**:
- Welch's t-tests for continuous metrics
- Approximate z-tests for discrete metrics
- 95% confidence intervals
- Cohen's d effect sizes
- Sample size variation handling

**Algorithm Scaling Analysis**:
- Performance impact of algorithm complexity
- Resource cost quantification
- Configuration-specific recommendations

**Trade-off Visualizations**:
- Duration vs Energy efficiency
- CPU cycles vs Energy consumption
- Multi-dimensional performance mapping

## 🧹 Data Filtering

### Outlier Detection Methods

**Coefficient of Variation (CV) Analysis**:
- Duration CV > 3.0: Timeout iterations
- Duration CV > 1.5: Retransmission iterations
- Duration CV < 1.5: Normal iterations

**Absolute Bounds Checking**:
- Energy > 15 mWh: Automatic timeout classification
- Handles extreme outliers beyond statistical methods

### Statistical Recalculation

After filtering, the system recalculates:
- **Aggregated means**: Proper averaging across kept iterations
- **Pooled standard deviations**: Correct variance pooling
- **Mode statistics**: Updated discrete metric modes
- **Min/Max/Range**: Recalculated bounds for discrete metrics

## 💡 Examples

### Complete Analysis Workflow

```bash
# 1. Filter data to remove outliers
python bench-data-filter.py bench-data-smarthome/
python bench-data-filter.py bench-data-fiducial/

# 2. Generate individual network analysis
./plots_wrapper.sh "duration,Energy (Wh),cpu_cycles" scatter A "--filtered"
./plots_wrapper.sh "total_frames,total_bytes" candlestick A "--filtered"

# 3. Cross-network comparison analysis
python bench-data-compare.py

# 4. Examine specific configurations
python bench-data-plots.py "Energy (Wh)" 25 --boxplot \
    --algorithms "KYBER_LEVEL1,KYBER_LEVEL5" \
    --cert-types "RSA_2048,DILITHIUM_LEVEL2" \
    --scenarios A --rasp --filtered
```

### Custom Algorithm Analysis

```bash
# Focus on post-quantum algorithms only
python bench-data-plots.py "duration" 25 --heatmap \
    --algorithms "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5" \
    --cert-types "DILITHIUM_LEVEL2,DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,FALCON_LEVEL1,FALCON_LEVEL5" \
    --scenarios A --rasp
```

### Energy Efficiency Analysis

```bash
# Energy consumption across scenarios
python bench-data-plots.py "Energy (mWh)" 25 --barplot \
    --scenarios A,C --rasp --filtered

# Duration vs Energy trade-off
python bench-data-plots.py "duration ms" 25 --scatter \
    --scenarios A --rasp --filtered
```

## 🔧 Advanced Usage

### Custom Styling and LaTeX

The toolkit supports publication-quality output with:
- LaTeX text rendering
- Computer Modern fonts
- Customizable color palettes
- Professional plot styling
- PDF output with editable text

### Configuration Files

Modify default settings in script headers:
```python
# Algorithm lists
default_algorithms = "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5"
default_cert_types = "RSA_2048,EC_P256,EC_ED25519,DILITHIUM_LEVEL2,..."

# Color schemes
security_mode_colors = {
    "pki": "#9fcf69",
    "psk": "#33acdc", 
    "nosec": "#ff6b35"
}
```

### Matplotlib Backend Control

```bash
# Prevent GUI display for server environments
MPLBACKEND=Agg python bench-data-plots.py ...

# Force interactive display
MPLBACKEND=Qt5Agg python bench-data-plots.py ...
```

## 📁 Output Files

### Plot Files
- **Location**: `./plots/` or `./plots-{custom-suffix}/`
- **Format**: PDF (publication-ready)
- **Naming**: `{plot_type}_{rasp_}_{metric}_n{n}_{params}_{scenario}.pdf`

### Statistical Reports
- **Network Differences**: `*_report.txt` files with statistical analysis
- **Outlier Analysis**: `outliers_*.csv` files with iteration-level details
- **Summary Data**: `summary_cts.csv` and `summary_dsc.csv`

### Filtered Data
- **Location**: Same directory as input
- **Format**: CSV with `_filtered` suffix
- **Content**: Cleaned data with recalculated statistics

## 🐛 Troubleshooting

### Common Issues

**LaTeX Errors**:
```bash
# Disable LaTeX if not available
# Edit script and set: setup_matplotlib_style(use_latex=False)
```

**Memory Issues with Large Datasets**:
```bash
# Process subsets of data
python bench-data-plots.py "duration" 25 --scatter \
    --algorithms "KYBER_LEVEL1" \
    --cert-types "RSA_2048,EC_P256" \
    --scenarios A
```

**File Not Found Errors**:
```bash
# Check data directory structure
ls bench-data/
ls bench-data-{network}/

# Verify file naming convention matches expected pattern
```

**Statistical Warnings**:
- Sample size warnings are normal with filtered data
- Different networks may have different outlier rates
- Statistical tests automatically handle unequal sample sizes

### Debug Mode

Enable detailed logging by adding print statements or using:
```bash
python -u bench-data-plots.py ... 2>&1 | tee analysis.log
```