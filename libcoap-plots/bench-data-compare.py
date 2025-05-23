import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
from adjustText import adjust_text 
from datetime import datetime

# 1. CONFIGURATION
ROOT_DIR = './bench-data-pll-15'  # where bench-data-fiducial/ etc. live
OUT_DIR = ROOT_DIR+'/compare_plots'
#NETWORKS = ['fiducial', 'smarthome', 'smartfactory' , 'publictransport']
NETWORKS = ['fiducial', 'smarthome']

# Default algorithm & certificate lists
default_algorithms = "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5"
default_cert_types = (
    "RSA_2048,EC_P256,EC_ED25519,"
    "DILITHIUM_LEVEL2,DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,"
    "FALCON_LEVEL1,FALCON_LEVEL5"
)
security_modes = ['pki', 'psk', 'nosec']

# Split defaults into lists
algorithms = [a.strip() for a in default_algorithms.split(',') if a.strip()]
cert_types = [c.strip() for c in default_cert_types.split(',') if c.strip()]

# Metrics & scenarios
METRIC_CTS = ['duration', 'cpu_cycles', 'Energy (Wh)', 'Power (W)', 'Max Power (W)']
METRIC_DSC = ['total_frames','total_bytes','frames_sent','bytes_sent','frames_received','bytes_received']
SCENARIOS = ['A', 'C']

def setup_matplotlib_style(use_latex=True):
    """
    Set up global matplotlib style settings for consistent, publication-quality plots.
    
    Args:
        use_latex (bool): Whether to use LaTeX for text rendering. 
                          Requires LaTeX to be installed on the system.
    """
    
    # Enable LaTeX rendering if requested
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{siunitx}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"]
        })
    
    # Set consistent font sizes
    plt.rcParams.update({
        "font.size": 16,           # Base font size
        "axes.titlesize": 18,       # Title font size
        "axes.labelsize": 16,       # Axis label font size
        "xtick.labelsize": 16,      # x-tick label font size
        "ytick.labelsize": 16,      # y-tick label font size
        "legend.fontsize": 18,      # Legend font size
        "figure.titlesize": 22      # Figure title font size
    })
    
    # Improve figure aesthetics
    plt.rcParams.update({
        "figure.figsize": (14, 10),  # Default figure size
        "figure.dpi": 100,          # Figure resolution
        "savefig.dpi": 300,         # Saved figure resolution
        "savefig.format": "pdf",    # Default save format
        "savefig.bbox": "tight",    # Tight bounding box
        "savefig.pad_inches": 0.2   # Padding
    })
    
    # Improve axes, grid, and ticks
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,     # Grid below data points
        "axes.linewidth": 0.8,      # Axis line width
        "axes.labelpad": 4,         # Spacing between axis and label
        "xtick.direction": "in",    # Ticks point inward
        "ytick.direction": "in",
        "xtick.major.size": 5,    # Major tick size
        "ytick.major.size": 5,
        "xtick.minor.size": 3.5,      # Minor tick size
        "ytick.minor.size": 3.5,
        "xtick.major.width": 0.8,   # Tick width
        "ytick.major.width": 0.8
    })
    
    # Legend settings
    plt.rcParams.update({
        "legend.framealpha": 0.8,   # Legend transparency
        "legend.edgecolor": "gray", # Legend border
        "legend.fancybox": True     # Rounded corners
    })
    
    # Export settings
    plt.rcParams.update({
        "pdf.fonttype": 42,         # Ensures text is editable in PDF
        "ps.fonttype": 42           # Ensures text is editable in PS
    })
    
    # Color cycle for consistent colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    )
    
    # Return the rcParams in case they're needed
    return plt.rcParams

def format_labels(metric):
    """
    Format metric names for LaTeX with proper units and mathematical notation.
    Removes all underscore characters from display.
    
    Args:
        metric (str): The metric name to format
        
    Returns:
        str: Properly formatted LaTeX string with underscores removed
    """
    if not metric:
        return ""
        
    # Format based on metric type
    if "cpu_cycles" in metric.lower():
        return r'CPU Cycles'
    elif "energy" in metric.lower() and "wh" in metric.lower():
        return r'Energy (Wh)'
    elif "power" in metric.lower() and "max" in metric.lower():
        return r'Max. Power (W)'
    elif "power" in metric.lower():
        return r'Power (W)'
    elif "frames_sent" in metric.lower():
        return r'Frames Sent'
    elif "frames_received" in metric.lower():
        return r'Frames Received'
    elif "total_frames" in metric.lower():
        return r'Total Frames'
    elif "bytes_sent" in metric.lower():
        return r'Bytes Sent'
    elif "bytes_received" in metric.lower():
        return r'Bytes Received'
    elif "total_bytes" in metric.lower():
        return r'Total Bytes'
    elif "duration" in metric.lower():
        return r'Duration (s)'
    else:
        # Generic case - REMOVE underscores (not escape them)
        return metric.replace("_", " ").title()

# 2. FILENAME PARSING
def parse_filename(fpath):
    base = os.path.basename(fpath)
    
    # Handle nosec case - no algorithm or cert
    if 'nosec' in base:
        nosec_re = re.compile(r"^udp_rasp_conv_stats_n25(_parallel)?_nosec_scenario(?P<scen>[AC])\.csv$")
        m = nosec_re.match(base)
        if m:
            return None, None, 'nosec', m.group('scen')
    
    # Handle psk case - algorithm but no cert
    psk_re = re.compile(
        r"^udp_rasp_conv_stats_"
        r"(?P<alg>[A-Z0-9_]+)"
        r"_n25(_parallel)?_psk_scenario(?P<scen>[AC])\.csv$"
    )
    m = psk_re.match(base)
    if m:
        return m.group('alg'), None, 'psk', m.group('scen')
    
    # Handle pki case - both algorithm and cert
    pki_re = re.compile(
        r"^udp_rasp_conv_stats_"
        r"(?P<alg>[A-Z0-9_]+)"
        r"_(?P<cert>RSA_2048|EC_P256|EC_ED25519|DILITHIUM_LEVEL[235]|FALCON_LEVEL[15])"
        r"_n25(_parallel)?_pki_scenario(?P<scen>[AC])\.csv$"
    )
    m = pki_re.match(base)
    if m:
        return m.group('alg'), m.group('cert'), 'pki', m.group('scen')
    
    return None, None, None, None

# 3. DATA PARSING AND LOADING
def parse_benchmark_file(file_path):
    """Parse a benchmark file with iteration blocks and global summary"""
    df = pd.read_csv(file_path, sep=';')
    
    # Find all separator rows
    separator_indices = []
    for idx, row in df.iterrows():
        if row.astype(str).str.contains('---').any():
            separator_indices.append(idx)
    
    # Find equals separator (marks start of global min/max/range)
    equals_idx = None
    for idx, row in df.iterrows():
        if row.astype(str).str.contains('===').any():
            equals_idx = idx
            break
    
    # Extract iteration data
    iterations = []
    
    # Process each iteration block
    for i, sep_idx in enumerate(separator_indices):
        # Start index for this iteration
        if i == 0:
            start_idx = 0
        else:
            # Start after the previous block's summary rows
            # Previous separator + 6 rows (mean, std, mode, below, above)
            start_idx = separator_indices[i-1] + 6
        
        # Extract raw data rows for this iteration
        raw_data = df.iloc[start_idx:sep_idx]
        
        # Extract summary stats (mean, std, mode after separator)
        if sep_idx + 5 < len(df):
            mean_row = df.iloc[sep_idx + 1]
            std_row = df.iloc[sep_idx + 2]
            mode_row = df.iloc[sep_idx + 3]
            below_row = df.iloc[sep_idx + 4]
            above_row = df.iloc[sep_idx + 5]
            
            iteration_data = {
                'raw_data': raw_data,
                'mean': mean_row,
                'std': std_row,
                'mode': mode_row,
                'below': below_row,
                'above': above_row
            }
            iterations.append(iteration_data)
    
    # Extract global summary (last separator block)
    if separator_indices:
        last_sep_idx = separator_indices[-1]
        global_mean = df.iloc[last_sep_idx + 1]
        global_std = df.iloc[last_sep_idx + 2]
        global_mode = df.iloc[last_sep_idx + 3]
        global_below = df.iloc[last_sep_idx + 4]
        global_above = df.iloc[last_sep_idx + 5]
        
        # Extract min/max/range if available (after equals separator)
        global_min = None
        global_max = None
        global_range = None
        if equals_idx is not None:
            if equals_idx + 1 < len(df):
                global_min = df.iloc[equals_idx + 1]
            if equals_idx + 2 < len(df):
                global_max = df.iloc[equals_idx + 2]
            if equals_idx + 3 < len(df):
                global_range = df.iloc[equals_idx + 3]
    
    return {
        'iterations': iterations,
        'global_mean': global_mean,
        'global_std': global_std,
        'global_mode': global_mode,
        'global_below': global_below,
        'global_above': global_above,
        'global_min': global_min,
        'global_max': global_max,
        'global_range': global_range
    }

def load_all():
    """Load all data using pre-calculated summary statistics"""
    all_data = []
    
    for net in NETWORKS:
        data_dir = os.path.join(ROOT_DIR, f'bench-data-{net}')
        if not os.path.isdir(data_dir):
            print(f"Warning: '{data_dir}' not found, skipping.")
            continue
            
        for f in glob.glob(os.path.join(data_dir, 'udp_rasp_conv_stats_*.csv')):
            alg, cert, mode, scen = parse_filename(f)
            if mode not in security_modes or scen not in SCENARIOS:
                continue
            
            # Parse the benchmark file
            parsed_data = parse_benchmark_file(f)
            
            # Process continuous metrics (using mean and std)
            for col in METRIC_CTS:
                if col in parsed_data['global_mean'].index:
                    mean_val = pd.to_numeric(parsed_data['global_mean'][col], errors='coerce')
                    std_val = pd.to_numeric(parsed_data['global_std'][col], errors='coerce')
                    
                    # Store mean and std for continuous metrics
                    if not pd.isna(mean_val):
                        record = {
                            'alg': alg,
                            'cert': cert,
                            'mode': mode,
                            'scenario': scen,
                            'network': net,
                            'metric': col,
                            'mean': mean_val,
                            'std': std_val if not pd.isna(std_val) else 0.0,
                            'file': f
                        }
                        all_data.append(record)
                        
            # Process discrete metrics (using mode, below, above, min, max, range)
            for col in METRIC_DSC:
                if parsed_data['global_mode'] is not None and col in parsed_data['global_mode'].index:
                    mode_val = pd.to_numeric(parsed_data['global_mode'][col], errors='coerce')
                    
                    # Also get below and above counts
                    below_val = None
                    above_val = None
                    if parsed_data['global_below'] is not None:
                        below_val = pd.to_numeric(parsed_data['global_below'][col], errors='coerce')
                    if parsed_data['global_above'] is not None:
                        above_val = pd.to_numeric(parsed_data['global_above'][col], errors='coerce')
                    
                    # Get min/max/range if available
                    min_val = None
                    max_val = None
                    range_val = None
                    
                    if parsed_data['global_min'] is not None:
                        min_val = pd.to_numeric(parsed_data['global_min'][col], errors='coerce')
                    if parsed_data['global_max'] is not None:
                        max_val = pd.to_numeric(parsed_data['global_max'][col], errors='coerce')
                    if parsed_data['global_range'] is not None:
                        range_val = pd.to_numeric(parsed_data['global_range'][col], errors='coerce')
                    
                    # For discrete metrics, we'll use mode as the primary value
                    if not pd.isna(mode_val):
                        record = {
                            'alg': alg,
                            'cert': cert,
                            'mode': mode,
                            'scenario': scen,
                            'network': net,
                            'metric': col,
                            'mean': mode_val,  # Use mode as the "mean" for interface consistency
                            'std': range_val if not pd.isna(range_val) else 0.0,  # Use range as "std"
                            'mode_val': mode_val,
                            'below': below_val,
                            'above': above_val,
                            'min': min_val,
                            'max': max_val,
                            'range': range_val,
                            'file': f
                        }
                        all_data.append(record)
                    
            for item in all_data:
                if 'metric' in item:
                    item['metric'] = pd.Categorical([item['metric']], 
                                                categories=METRIC_CTS + METRIC_DSC, 
                                                ordered=True)[0]
    
    return all_data

# 4. STATISTICAL SUMMARY
def summarize(all_data, metrics=None):
    """Create summary with correct statistics from pre-calculated values"""
    if metrics is None:
        metrics = METRIC_CTS
        
    # Determine if we're dealing with continuous or discrete metrics
    is_discrete = all(m in METRIC_DSC for m in metrics)
        
    # Convert to DataFrame for easier manipulation
    records = []
    for item in all_data:
        if 'mean' in item and 'iteration_values' not in item:  # Skip iteration data
            # Only include data for requested metrics
            if item['metric'] not in metrics:
                continue
                
            record = {
                'alg': item['alg'] or 'none',
                'cert': item['cert'] or 'none',
                'mode': item['mode'],
                'scenario': item['scenario'],
                'network': item['network'],
                'metric': item['metric'],
                'mean': item['mean'],
                'std': item['std']
            }
            # Add min/max for discrete metrics if available
            if item['metric'] in METRIC_DSC:
                if 'min' in item:
                    record['min'] = item['min']
                if 'max' in item:
                    record['max'] = item['max']
            records.append(record)
    
    if not records:
        print(f"No records found for metrics: {metrics}")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df['metric'] = pd.Categorical(df['metric'], 
                             categories=metrics,  # Use only requested metrics
                             ordered=True)
    
    # For discrete metrics, we need to get mode/above/below/min/max/range
    if is_discrete:
        # Process discrete metric data differently
        summary_records = []
        
        # Get unique configurations
        unique_configs = df[['alg', 'cert', 'mode', 'scenario']].drop_duplicates()
        
        for _, config in unique_configs.iterrows():
            alg, cert, mode, scenario = config['alg'], config['cert'], config['mode'], config['scenario']
            
            record = {
                'alg': alg if alg != 'none' else '',
                'cert': cert if cert != 'none' else '',
                'mode': mode,
                'scenario': scenario
            }
            
            # For each metric and network, extract appropriate values
            for metric in metrics:
                for net in NETWORKS:
                    subset = df[(df['alg'] == alg) & 
                               (df['cert'] == cert) & 
                               (df['mode'] == mode) & 
                               (df['scenario'] == scenario) & 
                               (df['metric'] == metric) & 
                               (df['network'] == net)]
                    
                    if not subset.empty:
                        row = subset.iloc[0]
                        # For discrete metrics, mean is actually mode, std is actually range
                        record[f"('{metric}', 'mode', '{net}')"] = row['mean']
                        record[f"('{metric}', 'range', '{net}')"] = row['std']
                        
                        # Also add min/max if available
                        if 'min' in row and pd.notna(row['min']):
                            record[f"('{metric}', 'min', '{net}')"] = row['min']
                        if 'max' in row and pd.notna(row['max']):
                            record[f"('{metric}', 'max', '{net}')"] = row['max']
            
            # Calculate percentage changes based on mode values
            for metric in metrics:
                for net in ['smarthome', 'smartfactory', 'publictransport']:
                    fiducial_key = f"('{metric}', 'mode', 'fiducial')"
                    net_key = f"('{metric}', 'mode', '{net}')"
                    
                    if fiducial_key in record and net_key in record:
                        fiducial_val = record[fiducial_key]
                        net_val = record[net_key]
                        
                        if pd.notna(fiducial_val) and pd.notna(net_val) and fiducial_val != 0:
                            pct_change = ((net_val - fiducial_val) / fiducial_val) * 100
                            record[f"('{metric}', 'pct_change', '{net}')"] = pct_change
            
            summary_records.append(record)
            
        # Convert to DataFrame and sort properly
        summary_df = pd.DataFrame(summary_records)
        
        if not summary_df.empty:
            # Create proper sorting keys
            summary_df['alg_sort'] = pd.Categorical(summary_df['alg'], 
                                                    categories=[''] + algorithms, 
                                                    ordered=True)
            summary_df['mode_sort'] = pd.Categorical(summary_df['mode'], 
                                                    categories=['pki', 'psk', 'nosec'], 
                                                    ordered=True)
            summary_df['cert_sort'] = pd.Categorical(summary_df['cert'], 
                                                    categories=[''] + cert_types, 
                                                    ordered=True)
            summary_df['scenario_sort'] = pd.Categorical(summary_df['scenario'], 
                                                        categories=SCENARIOS, 
                                                        ordered=True)
            
            # Sort by algorithm, mode, certificate type, scenario
            summary_df = summary_df.sort_values(['alg_sort', 'mode_sort', 'cert_sort', 'scenario_sort'])
            
            # Drop the sort columns and set index
            summary_df = summary_df.drop(['alg_sort', 'mode_sort', 'cert_sort', 'scenario_sort'], axis=1)
            summary_df = summary_df.set_index(['alg', 'cert', 'mode', 'scenario'])
            
        return summary_df
    
    else:
        # Continue with existing logic for continuous metrics
        # Create pivot tables for mean and std
        mean_pivot = df.pivot_table(
            index=['alg', 'cert', 'mode', 'scenario', 'metric'],
            columns='network',
            values='mean',
            observed=False
        )
        
        std_pivot = df.pivot_table(
            index=['alg', 'cert', 'mode', 'scenario', 'metric'],
            columns='network',
            values='std',
            observed=False
        )
        
        # Calculate percentage changes
        pct_changes = {}
        for net in ['smarthome', 'smartfactory', 'publictransport']:
            if 'fiducial' in mean_pivot.columns and net in mean_pivot.columns:
                pct_changes[net] = ((mean_pivot[net] - mean_pivot['fiducial']) / mean_pivot['fiducial']) * 100
        
        # Combine all results into final summary
        summary_records = []
        
        # Get unique configurations from the mean_pivot
        unique_configs = set()
        for idx in mean_pivot.index:
            alg, cert, mode, scenario, metric = idx
            unique_configs.add((alg, cert, mode, scenario))
        
        # Create records for each configuration
        for config in unique_configs:
            alg, cert, mode, scenario = config
            
            record = {
                'alg': alg if alg != 'none' else '',
                'cert': cert if cert != 'none' else '',
                'mode': mode,
                'scenario': scenario
            }
            
            # Add data for each requested metric
            for metric in metrics:
                idx = (alg, cert, mode, scenario, metric)
                
                # Add mean, std, and percentage change for each network
                for net in NETWORKS:
                    if idx in mean_pivot.index and net in mean_pivot.columns:
                        mean_val = mean_pivot.loc[idx, net]
                        std_val = std_pivot.loc[idx, net]
                        
                        # Only add if the value exists
                        if pd.notna(mean_val):
                            record[f"('{metric}', 'mean', '{net}')"] = mean_val
                            record[f"('{metric}', 'std', '{net}')"] = std_val
                
                # Add percentage changes
                for net in ['smarthome', 'smartfactory', 'publictransport']:
                    if idx in mean_pivot.index and net in pct_changes and idx in pct_changes[net].index:
                        pct_val = pct_changes[net].loc[idx]
                        if pd.notna(pct_val):
                            record[f"('{metric}', 'pct_change', '{net}')"] = pct_val
            
            # Only add the record if it has data for at least one metric
            has_data = False
            for metric in metrics:
                if any(f"('{metric}'," in col for col in record if col not in ['alg', 'cert', 'mode', 'scenario']):
                    has_data = True
                    break
            
            if has_data:
                summary_records.append(record)
        
        # Convert to DataFrame and sort properly
        summary_df = pd.DataFrame(summary_records)
        
        if not summary_df.empty:
            # Create proper sorting keys
            summary_df['alg_sort'] = pd.Categorical(summary_df['alg'], 
                                                    categories=[''] + algorithms, 
                                                    ordered=True)
            summary_df['mode_sort'] = pd.Categorical(summary_df['mode'], 
                                                    categories=['pki', 'psk', 'nosec'], 
                                                    ordered=True)
            summary_df['cert_sort'] = pd.Categorical(summary_df['cert'], 
                                                    categories=[''] + cert_types, 
                                                    ordered=True)
            summary_df['scenario_sort'] = pd.Categorical(summary_df['scenario'], 
                                                        categories=SCENARIOS, 
                                                        ordered=True)
            
            # Sort by algorithm, mode, certificate type, scenario
            summary_df = summary_df.sort_values(['alg_sort', 'mode_sort', 'cert_sort', 'scenario_sort'])
            
            # Drop the sort columns and set index
            summary_df = summary_df.drop(['alg_sort', 'mode_sort', 'cert_sort', 'scenario_sort'], axis=1)
            summary_df = summary_df.set_index(['alg', 'cert', 'mode', 'scenario'])
        
        return summary_df

# 5. VISUALIZATION FUNCTIONS

def plot_tradeoff(all_data, metric_x, metric_y, scenario, include_nosec=False, normalize=False, log_scale=False, output_dir='./bench-plots-compare'):
    """
    Plot mean metric_y vs metric_x per configuration, colored by network for a given scenario
    """
    
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Convert data to DataFrame for easier filtering
    records = []
    for item in all_data:
        if 'mean' in item:
            # Filter out nosec unless explicitly included
            if item['mode'] == 'nosec' and not include_nosec:
                continue
            records.append({
                'alg': item['alg'] or 'none',
                'cert': item['cert'] or 'none',
                'mode': item['mode'],
                'scenario': item['scenario'],
                'network': item['network'],
                'metric': item['metric'],
                'mean': item['mean']
            })
    
    df = pd.DataFrame(records)
    
    # Filter for the specified scenario
    df = df[df['scenario'] == scenario]
    
    # Pivot to get x and y metrics
    x_data = df[df['metric'] == metric_x].pivot_table(
        index=['alg', 'cert', 'mode'],
        columns='network',
        values='mean'
    )
    
    y_data = df[df['metric'] == metric_y].pivot_table(
        index=['alg', 'cert', 'mode'],
        columns='network',
        values='mean'
    )
    
    # If normalize is True, normalize both axes to 0-1 range
    x_values_all = []
    y_values_all = []
    
    # Collect all values for normalization
    for idx in x_data.index:
        for net in x_data.columns:
            if pd.notna(x_data.loc[idx, net]):
                x_values_all.append(x_data.loc[idx, net])
            if pd.notna(y_data.loc[idx, net]):
                y_values_all.append(y_data.loc[idx, net])
    
    if normalize and x_values_all and y_values_all:
        x_min, x_max = min(x_values_all), max(x_values_all)
        y_min, y_max = min(y_values_all), max(y_values_all)
        
        # Normalize data
        x_data_norm = (x_data - x_min) / (x_max - x_min)
        y_data_norm = (y_data - y_min) / (y_max - y_min)
        
        # Use normalized data
        x_data_plot = x_data_norm
        y_data_plot = y_data_norm
        
        # Update axis labels to indicate normalization
        xlabel = f'{format_labels(metric_x)} (normalized)'
        ylabel = f'{format_labels(metric_y)} (normalized)'
    else:
        x_data_plot = x_data
        y_data_plot = y_data
        xlabel = format_labels(metric_x)
        ylabel = format_labels(metric_y)
    
    # Define base colors for networks (solid colors)
    network_colors = {'fiducial': 'green', 'smarthome': 'blue', 'smartfactory': 'purple', 'publictransport': 'red'}
    # network_colors = {'fiducial': 'green', 'smarthome': 'blue'}
    
    # Define markers for algorithms
    alg_markers = {
        'KYBER_LEVEL1': 'o',    # Circle
        'KYBER_LEVEL3': 's',    # Square
        'KYBER_LEVEL5': '^',    # Triangle
        'none': 'P'             # Plus for nosec
    }
    
    # Create shorter labels for certificates
    cert_short = {
        'DILITHIUM_LEVEL2': 'DIL2',
        'DILITHIUM_LEVEL3': 'DIL3',
        'DILITHIUM_LEVEL5': 'DIL5',
        'EC_ED25519': 'Ed25519',
        'EC_P256': 'P256',
        'FALCON_LEVEL1': 'FAL1',
        'FALCON_LEVEL5': 'FAL5',
        'RSA_2048': 'RSA'
    }
    
    texts = []  # Store text objects for adjustment
    plotted_labels = []  # Store (x, y, label) for text placement
    
    # Plot each configuration
    for idx in x_data_plot.index:
        alg, cert, mode = idx
        
        # Create label based on security mode
        if mode == 'nosec':
            label = 'NoSec'
        elif mode == 'psk':
            label = 'PSK'  # Simplified label for PSK
        else:  # pki
            cert_label = cert_short.get(cert, cert[:4])
            label = cert_label  # Just use cert type for PKI
        
        # Collect points for the same configuration across networks
        config_points = []
        x_positions = []
        y_positions = []
        
        for net in NETWORKS:  # Fixed order
            if net in x_data_plot.columns and net in y_data_plot.columns:
                x = x_data_plot.loc[idx, net]
                y = y_data_plot.loc[idx, net]
                
                if pd.isna(x) or pd.isna(y):
                    continue
                
                config_points.append((x, y, net))
                x_positions.append(x)
                y_positions.append(y)
                # Get color and marker
                color = network_colors[net]
                marker = alg_markers.get(alg, 'o')
                
                # Plot point
                plt.scatter(x, y, color=color, marker=marker, s=120, 
                           alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Connect points from the same configuration with thin lines
        if len(config_points) > 1:
            xs = [p[0] for p in config_points]
            ys = [p[1] for p in config_points]
            plt.plot(xs, ys, 'k-', alpha=0.5, linewidth=0.75, zorder=1)
            
        # Place label at the centroid of the configuration group
        if x_positions and y_positions:
            # Use geometric mean for log scale, arithmetic mean for linear scale
            if not normalize and log_scale:
                # For log scale, use geometric mean
                x_label = np.exp(np.mean(np.log(x_positions)))
                y_label = np.exp(np.mean(np.log(y_positions)))
            else:
                # For normalized/linear scale, use arithmetic mean
                x_label = np.mean(x_positions)
                y_label = np.mean(y_positions)
                
            plotted_labels.append((x_label, y_label, label))
    
    # Apply log scale only if not normalized
    if not normalize and log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Add labels at centroids
    for x, y, label in plotted_labels:
        txt = ax.text(x, y, label, fontsize=9, alpha=0.9,
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                              edgecolor='gray', linewidth=0.5, alpha=0.8))
        texts.append(txt)
    
    # Adjust text positions to minimize overlaps
    if texts:
        adjust_text(texts, 
                    force_points=0.5,  # How much to push away from points
                    force_text=0.5,    # How much to push away from other texts
                    expand_points=(1.2, 1.2),  # Expand bounding box around points
                    expand_text=(1.1, 1.1),    # Expand bounding box around texts
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = f"{format_labels(metric_y)} vs {format_labels(metric_x)} (Scenario {scenario})"
    
    if normalize:
        title += " - Normalized"
    plt.title(title)
    
    # Create custom legend
    legend_elements = []
    
    # Network legend
    for net, color in network_colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, 
                                     label=f'Network: {net}'))
    
    # Add separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Algorithm legend
    for alg, marker in alg_markers.items():
        if alg == 'none' and not include_nosec:
            continue  # Skip NoSec legend if not included
        if alg != 'none':
            legend_elements.append(Line2D([0], [0], marker=marker, color='gray', 
                                         markersize=10, label=f'Algorithm: {alg}', 
                                         markerfacecolor='lightgray'))
        else:
            legend_elements.append(Line2D([0], [0], marker=marker, color='gray', 
                                         markersize=10, label='No Security', 
                                         markerfacecolor='lightgray'))
    
    plt.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1.01, 1), ncol=1)
    plt.grid(True, ls='--', alpha=0.3, which='both')
    plt.tight_layout()
    
    # Adjust filename to indicate if nosec is excluded and if normalized
    suffix = ''
    if normalize:
        suffix += '_normalized'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/tradeoff_{metric_x}_{metric_y}_{scenario}{suffix}.pdf', bbox_inches='tight')
    plt.close()
    
def create_waterfall_plot(all_data, metric, scenario, plot_type='network', baseline_alg='KYBER_LEVEL1', 
                       baseline_cert='RSA_2048', baseline_network='fiducial', security_mode='pki',
                       output_dir='./bench-plots-compare'):
    """
    Create a waterfall plot showing either:
    1. Impact of different networks on performance (using fiducial as baseline)
    2. Impact of algorithm scaling on performance (KYBER_LEVEL1/3/5)
    
    Parameters:
    -----------
    all_data : list
        The benchmark data loaded from load_all()
    metric : str
        The metric to analyze (e.g., 'cpu_cycles', 'Energy (Wh)')
    scenario : str
        The scenario to use ('A' or 'C')
    plot_type : str
        'network' to show network impact or 'algorithm' to show algorithm scaling
    baseline_alg : str
        The baseline algorithm (default: 'KYBER_LEVEL1')
    baseline_cert : str
        The certificate type for PKI mode (default: 'RSA_2048')
    baseline_network : str
        The network to use as baseline for algorithm plots (default: 'fiducial')
    security_mode : str
        Security mode to use ('pki', 'psk', 'nosec')
    output_dir : str
        Directory to save the output plot
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    if plot_type == 'network':
        # Network impact waterfall plot
        create_network_waterfall(all_data, ax, metric, scenario, security_mode, 
                              baseline_cert if security_mode == 'pki' else None, 
                              baseline_alg if security_mode != 'nosec' else None)
    else:
        # Algorithm scaling waterfall plot
        create_algorithm_waterfall(all_data, ax, metric, scenario, security_mode,
                               baseline_cert if security_mode == 'pki' else None,
                               baseline_network)
    
    plt.tight_layout()
    
    # Generate appropriate filename
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    if plot_type == 'network':
        filename = f"{output_dir}/waterfall_network_{clean_metric}_{security_mode}"
        if security_mode == 'pki':
            filename += f"_{baseline_alg}_{baseline_cert}"
        elif security_mode == 'psk':
            filename += f"_{baseline_alg}"
    else:
        filename = f"{output_dir}/waterfall_algorithm_{clean_metric}_{security_mode}"
        if security_mode == 'pki':
            filename += f"_{baseline_cert}"
        elif security_mode == 'psk':
            filename += "_psk"
        filename += f"_{baseline_network}"
    
    filename += f"_scenario{scenario}.pdf"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(filename)
    print(f"Saved {plot_type} waterfall plot to: {filename}")
    plt.close()
    
    return filename

def create_network_waterfall(all_data, ax, metric, scenario, security_mode='pki', 
                         cert_type=None, algorithm=None):
    """
    Helper function to create normalized network impact waterfall plot
    Modified to show changes more effectively
    
    Parameters match the parent function, with ax for plotting.
    """
    # Extract and filter data
    records = []
    for item in all_data:
        if 'mean' in item and item['metric'] == metric and item['scenario'] == scenario:
            # For PKI mode
            if security_mode == 'pki' and item['mode'] == 'pki' and item['alg'] == algorithm and item['cert'] == cert_type:
                records.append({
                    'network': item['network'],
                    'value': item['mean']
                })
            # For PSK mode
            elif security_mode == 'psk' and item['mode'] == 'psk' and item['alg'] == algorithm and (item['cert'] == '' or item['cert'] == 'none'):
                records.append({
                    'network': item['network'],
                    'value': item['mean']
                })
            # For NoSec mode
            elif security_mode == 'nosec' and item['mode'] == 'nosec':
                records.append({
                    'network': item['network'],
                    'value': item['mean']
                })
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(records)
    if df.empty:
        print(f"No data found for {metric}, {scenario}, {security_mode}, {cert_type}, {algorithm}")
        return
    
    # Sort by network order
    network_order = {'fiducial': 0, 'smarthome': 1, 'smartfactory': 2, 'publictransport': 3}
    df['order'] = df['network'].map(network_order)
    df = df.sort_values('order')
    
    # Get the baseline (fiducial) value
    fiducial_value = df[df['network'] == 'fiducial']['value'].values[0]
    
    # Normalize all values relative to fiducial
    df['normalized'] = df['value'] / fiducial_value
    
    # Calculate percentage changes
    df['pct_change'] = (df['normalized'] - 1) * 100
    
    # Filter out the baseline network - we'll only show changes
    df_changes = df[df['network'] != 'fiducial']
    
    # Set up bars and labels directly
    positions = np.arange(len(df_changes))
    labels = [network.capitalize() for network in df_changes['network']]
    values = df_changes['normalized'].values - 1.0  # Changes relative to baseline (which is 0)
    pct_changes = df_changes['pct_change'].values
    
    # Define colors based on whether the change is an improvement or degradation
    colors = ['#4CAF50' if v < 0 else '#F44336' for v in values]  # Green for improvement, red for degradation
    
    # Plot the bars directly - representing the absolute change from baseline
    bars = ax.bar(positions, values, color=colors, edgecolor='black', width=0.6)
    
    # Add percentage labels
    for i, (v, pct) in enumerate(zip(values, pct_changes)):
        # Position label based on bar direction
        if v > 0:
            y_pos = min(v + 0.5, v * 0.5)  # Cap the position for very large values
            va_pos = 'bottom'
        else:
            y_pos = max(v - 0.5, v * 0.5)  # Cap the position for very large negative values
            va_pos = 'top'
            
        # Format label with appropriate precision
        if abs(pct) >= 1000:
            label_text = f"{pct:.0f}\\%"  # No decimals for large percentages
        elif abs(pct) >= 100:
            label_text = f"{pct:.1f}\\%"  # One decimal for medium percentages
        else:
            label_text = f"{pct:.1f}\\%"  # One decimal for small percentages
            
        # Add the label
        ax.text(i, y_pos, label_text, ha='center', va=va_pos, fontweight='bold')
    
    # Set axis labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel(f"Change from Fiducial (normalized)")
    
    # Construct title based on configuration
    if security_mode == 'pki':
        title = f"Network Impact on {format_labels(metric)}\n{format_labels(algorithm)} with {format_labels(cert_type)} (Scenario {scenario})"
    elif security_mode == 'psk':
        title = f"Network Impact on {format_labels(metric)}\n{format_labels(algorithm)} with PSK (Scenario {scenario})"
    else:  # nosec
        title = f"Network Impact on {format_labels(metric)}\nNo Security (Scenario {scenario})"
    
    ax.set_title(title)
    
    # Add baseline reference as a horizontal dashed line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Annotate the baseline
    #ax.text(-0.5, 0, "Fiducial\n(Baseline)", ha='center', va='center', fontweight='bold')
    
    # Set y-axis limits with appropriate padding
    max_abs_change = max(abs(min(values)), abs(max(values)))
    y_padding = max_abs_change * 0.2
    
    # Use a symmetric scale if changes are not too extreme
    if max_abs_change < 10:
        ax.set_ylim(-max_abs_change - y_padding, max_abs_change + y_padding)
    else:
        # For large changes, use asymmetric scale but ensure zero is visible
        y_min = min(values) - y_padding
        y_max = max(values) + y_padding
        # Ensure zero is in the range with some margin
        if y_min > -0.5:
            y_min = -0.5
        if y_max < 0.5:
            y_max = 0.5
        ax.set_ylim(y_min, y_max)
    
    # Add grid for readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ensure figure has enough vertical space
    plt.subplots_adjust(bottom=0.15, top=0.85)

def create_algorithm_waterfall(all_data, ax, metric, scenario, security_mode='pki',
                           cert_type=None, network='fiducial'):
    """
    Helper function to create normalized algorithm scaling waterfall plot
    Modified to show small changes more effectively
    
    Parameters match the parent function, with ax for plotting.
    """
    # Define algorithms to compare
    algorithms = ['KYBER_LEVEL1', 'KYBER_LEVEL3', 'KYBER_LEVEL5']
    
    # Extract and filter data
    records = []
    for item in all_data:
        if ('mean' in item and item['metric'] == metric and item['scenario'] == scenario 
            and item['network'] == network and item['alg'] in algorithms):
            
            # For PKI mode
            if security_mode == 'pki' and item['mode'] == 'pki' and item['cert'] == cert_type:
                records.append({
                    'algorithm': item['alg'],
                    'value': item['mean']
                })
            # For PSK mode
            elif security_mode == 'psk' and item['mode'] == 'psk' and (item['cert'] == '' or item['cert'] == 'none'):
                records.append({
                    'algorithm': item['alg'],
                    'value': item['mean']
                })
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(records)
    if df.empty:
        print(f"No data found for algorithm waterfall plot with {metric}, {scenario}, {security_mode}, {cert_type}, {network}")
        return
    
    # Sort by algorithm order
    df['order'] = df['algorithm'].apply(lambda x: algorithms.index(x))
    df = df.sort_values('order')
    
    # Get the baseline (KYBER_LEVEL1) value
    base_alg = 'KYBER_LEVEL1'
    baseline_value = df[df['algorithm'] == base_alg]['value'].values[0]
    
    # Normalize all values relative to baseline
    df['normalized'] = df['value'] / baseline_value
    
    # Calculate percentage changes
    df['pct_change'] = (df['normalized'] - 1) * 100
    
    # Filter out the baseline algorithm - we'll only show changes
    df_changes = df[df['algorithm'] != base_alg]
    
    # Set up bars and labels directly
    positions = np.arange(len(df_changes))
    labels = [format_labels(lab) for lab in df_changes['algorithm']]
    values = df_changes['normalized'].values - 1.0  # Changes relative to baseline (which is 0)
    pct_changes = df_changes['pct_change'].values
    
    # Define colors based on whether the change is an improvement or degradation
    colors = ['#4CAF50' if v < 0 else '#F44336' for v in values]  # Green for improvement, red for degradation
    
    # Plot the bars directly - representing the absolute change from baseline
    bars = ax.bar(positions, values, color=colors, edgecolor='black', width=0.6)
    
    # Add percentage labels on top of bars
    for i, (v, pct) in enumerate(zip(values, pct_changes)):
        va_pos = 'bottom' if v > 0 else 'top'
        y_offset = 0.0001 if v > 0 else -0.0001  # Small offset to position text
        ax.text(i, v + y_offset, f"{pct:.1f}\\%", ha='center', va=va_pos, fontweight='bold')
    
    # Set axis labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel(f"Change from {format_labels(base_alg)} (normalized)")
    
    # Construct title based on configuration
    if security_mode == 'pki':
        title = f"Algorithm Scaling Impact on {format_labels(metric)}\n{format_labels(cert_type)} Certificate (Scenario {scenario}, {network.capitalize()})"
    else:  # psk
        title = f"Algorithm Scaling Impact on {format_labels(metric)}\nPSK (Scenario {scenario}, {network.capitalize()})"
    
    ax.set_title(title)
    
    # Add baseline reference as a horizontal dashed line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Annotate the baseline
    #ax.text(-0.5, 0, f"{base_alg}\n(Baseline)", ha='center', va='center', fontweight='bold')
    
    # Calculate y-axis limits - force a minimum scale to ensure small changes are visible
    max_abs_change = max(abs(min(values)), abs(max(values)))
    min_scale = 0.006  # Minimum scale for y-axis (+/- 0.6%)
    scale = max(max_abs_change, min_scale)
    
    # Add some padding to the scale
    y_padding = scale * 0.2
    ax.set_ylim(-scale - y_padding, scale + y_padding)
    
    # Add minor grid lines for better readability of small values
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.4)
    ax.minorticks_on()
    
    # Ensure figure has enough vertical space
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
def create_network_comparison_plot(all_data, metric, scenario, security_mode='pki', output_dir='./bench-plots-compare'):
    """
    Create a comparison plot showing network impacts across multiple security configurations.
    
    Parameters:
    -----------
    all_data : list
        The full benchmark data loaded from load_all()
    metric : str
        Metric to analyze (e.g., 'cpu_cycles', 'Energy (Wh)')
    scenario : str
        Scenario identifier ('A' or 'C')
    security_mode : str
        'pki', 'psk', or 'nosec' to determine which configurations to include
    output_dir : str
        Directory to save the output plot
    """
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    # Networks to include (excluding fiducial which is the baseline)
    networks = ['smarthome', 'smartfactory', 'publictransport']

    # DIAGNOSTIC: Count of items by mode
    mode_counts = {}
    for item in all_data:
        mode = item['mode']
        if mode not in mode_counts:
            mode_counts[mode] = 0
        mode_counts[mode] += 1
    print(f"DIAGNOSTIC: Items by mode: {mode_counts}")

    # Define configurations to include based on security_mode
    configurations = []
    if security_mode == 'pki':
        # Include all algorithm + certificate type combinations
        algorithms = ['KYBER_LEVEL1', 'KYBER_LEVEL3', 'KYBER_LEVEL5']
        cert_types = ['RSA_2048', 'EC_P256', 'EC_ED25519', 'DILITHIUM_LEVEL2',
                    'DILITHIUM_LEVEL3', 'DILITHIUM_LEVEL5', 'FALCON_LEVEL1', 'FALCON_LEVEL5']
        
        for alg in algorithms:
            for cert in cert_types:
                configurations.append({
                    'mode': 'pki',
                    'alg': alg,
                    'cert': cert,
                    'label': f"{alg} ({cert})"
                })
    elif security_mode == 'psk':
        # Include all algorithms with PSK
        algorithms = ['KYBER_LEVEL1', 'KYBER_LEVEL3', 'KYBER_LEVEL5']
        
        # Use the algorithms we specified, but be aware of actual configs
        for alg in algorithms:
            configurations.append({
                'mode': 'psk',
                'alg': alg,
                'cert': None,  # Use None instead of empty string
                'label': f"{alg} (PSK)"
            })
    else:  # nosec
        configurations.append({
            'mode': 'nosec',
            'alg': None,
            'cert': None,
            'label': "NoSec"
        })

    # Filter to only include configurations we have data for
    valid_configurations = []
    for config in configurations:
        # Check if we have fiducial data for this configuration (as baseline)
        has_data = False
        for item in all_data:
            if (item['metric'] == metric and item['scenario'] == scenario and
                item['mode'] == config['mode'] and item['alg'] == config['alg'] and
                item['network'] == 'fiducial'):
                
                # Compare cert values - handle None case explicitly
                if config['cert'] is None and item['cert'] is None:
                    cert_match = True
                elif config['cert'] == item['cert']:  # Regular string comparison
                    cert_match = True
                else:
                    cert_match = False
                    
                if cert_match:
                    has_data = True
                    break

        if has_data:
            valid_configurations.append(config)

    if not valid_configurations:
        print(f"No valid configurations found for {metric}, {scenario}, {security_mode}")
        plt.close()
        return
    
    # Number of configurations will determine bar width
    n_configs = len(valid_configurations)
    width = 1.2 / n_configs  # Adjust total width / number of configs
    
    # Dictionary to store configuration colors
    config_colors = {}
    
    # Create a color map for distinguishing configurations
    if security_mode == 'pki':
        # Group colors by algorithm
        algorithms = sorted(set(c['alg'] for c in valid_configurations))
        alg_colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            # Get cert types for this algorithm
            certs = sorted(set(c['cert'] for c in valid_configurations if c['alg'] == alg))
            # Generate shades for each cert type
            cert_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(certs)))
            
            for j, cert in enumerate(certs):
                config_colors[(alg, cert)] = cert_colors[j]
    else:
        # For PSK or NoSec, just use different colors for algorithms
        algorithms = sorted(set(c['alg'] for c in valid_configurations))
        alg_colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            config_colors[(alg, '')] = alg_colors[i]
    
    # Set up the bars for each network
    for n_idx, network in enumerate(networks):
        # Position for this network group
        pos = n_idx + 0.5  # Center of this network's group
        
        # Process each configuration for this network
        for c_idx, config in enumerate(valid_configurations):
            # Get fiducial baseline value for this configuration
            fiducial_value = None
            for item in all_data:
                if (item['metric'] == metric and item['scenario'] == scenario and
                    item['mode'] == config['mode'] and item['alg'] == config['alg'] and
                    item['cert'] == config['cert'] and item['network'] == 'fiducial'):
                    fiducial_value = item['mean']
                    break
            
            # Get value for this network and configuration
            network_value = None
            for item in all_data:
                if (item['metric'] == metric and item['scenario'] == scenario and
                    item['mode'] == config['mode'] and item['alg'] == config['alg'] and
                    item['network'] == network):
                    
                    # Compare cert values - handle None case explicitly
                    if config['cert'] is None and item['cert'] is None:
                        cert_match = True
                    elif config['cert'] == item['cert']:  # Regular string comparison
                        cert_match = True
                    else:
                        cert_match = False
                        
                    if cert_match:
                        network_value = item['mean']
                        break
            
            if fiducial_value is not None and network_value is not None:
                # Calculate relative change
                rel_change = (network_value / fiducial_value) - 1.0  # Percentage change
                
                # Calculate bar position
                bar_pos = pos + (c_idx - n_configs/2) * width
                
                # Bar color - red for increase, green for decrease
                color = 'red' if rel_change > 0 else 'green'
                
                # Special coloring for PKI mode
                if security_mode == 'pki':
                    color = config_colors.get((config['alg'], config['cert']), color)
                
                # Draw bar
                bar = ax.bar(bar_pos, rel_change * 100, width * 0.9, 
                        color=color, alpha=0.7, edgecolor='black')
                
                # Add percentage change label
                pct_change = rel_change * 100
                if abs(pct_change) > 5:  # Only label if significant
                    if pct_change > 0:
                        label_text = f"+{pct_change:.1f}\\%"
                    else:
                        label_text = f"{pct_change:.1f}\\%"
                    
                    ax.text(bar_pos, pct_change + (5 if pct_change > 0 else -5),
                           label_text, ha='center', va='center', fontsize=7,
                           rotation=90, color='black')
    
    # Set up x-axis ticks and labels
    ax.set_xticks([n_idx + 0.5 for n_idx in range(len(networks))])
    ax.set_xticklabels([net.capitalize() for net in networks])
    
    # Add a horizontal line at 0%
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Create legend for configurations
    if security_mode == 'pki':
        # Group by algorithm
        handles = []
        labels = []
        
        algorithms = sorted(set(c['alg'] for c in valid_configurations))
        for alg in algorithms:
            # Add algorithm header
            labels.append(alg)
            handles.append(plt.Rectangle((0,0),1,1, color='none'))
            
            # Add cert types for this algorithm
            certs = sorted(set(c['cert'] for c in valid_configurations if c['alg'] == alg))
            for cert in certs:
                labels.append(f"  {cert}")
                color = config_colors.get((alg, cert), 'gray')
                handles.append(plt.Rectangle((0,0),1,1, color=color))
    else:
        # Simple legend
        handles = []
        labels = []
        for config in valid_configurations:
            labels.append(config['label'])
            color = config_colors.get((config['alg'], config['cert']), 'gray')
            handles.append(plt.Rectangle((0,0),1,1, color=color))
    
    # Add the legend
    ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.01, 1), ncol=1)
    
    # Set axis labels and title
    ax.set_ylabel(f"Change from Fiducial Baseline (\\%)")
    
    if security_mode == 'pki':
        title = f"Network Impact on {metric}\nAll PKI Configurations (Scenario {scenario})"
    elif security_mode == 'psk':
        title = f"Network Impact on {metric}\nAll PSK Configurations (Scenario {scenario})"
    else:
        title = f"Network Impact on {metric}\nNo Security (Scenario {scenario})"
    
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"{output_dir}/network_comparison_{clean_metric}_{security_mode}_scenario{scenario}.pdf"
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved network comparison plot to: {filename}")
    plt.close()
    
    return filename

def create_spider_plot(all_data, metrics, scenario, security_modes=['pki', 'psk', 'nosec'], 
                      algorithms=None, certificates=None, include_legend=True,
                      normalization='min_max', baseline_network='fiducial', 
                      apply_metric_inversion=True, max_security_configs=9,
                      output_dir='./bench-plots-compare'):
    """
    Create a spider (radar) plot comparing metrics across different networks.
    
    Parameters:
    -----------
    all_data : list
        The benchmark data loaded from load_all()
    metrics : list
        List of metrics to include in the plot (e.g., ['cpu_cycles', 'Energy (Wh)', 'total_bytes'])
    scenario : str
        The scenario to use ('A' or 'C')
    security_modes : list
        List of security modes to include ('pki', 'psk', 'nosec')
    algorithms : list or None
        List of algorithms to include (default: None, include all)
    certificates : list or None
        List of certificate types to include (default: None, include all)
    include_legend : bool
        Whether to include a legend (default: True)
    normalization : str
        Normalization method ('min_max', 'max_percent', 'z_score', 'log', 'baseline_relative')
    baseline_network : str
        Network to use as the baseline for normalization (default: 'fiducial')
    apply_metric_inversion : bool
        If True, invert metrics where lower is better (so higher values are always better)
    max_security_configs : int
        Maximum number of security configurations to include (default: 9)
    output_dir : str
        Directory to save the output plot
    """
    try:
        # Extract networks
        all_networks = set()
        for item in all_data:
            all_networks.add(item['network'])
        
        networks = sorted(list(n for n in all_networks if n != baseline_network))
        
        if not networks:
            print(f"Warning: No non-baseline networks found. Cannot create spider plot.")
            return None
            
        # Define which metrics should be inverted (lower is better -> higher is better)
        invert_metrics = set(['cpu_cycles', 'Energy (Wh)', 'Power (W)', 'Max Power (W)', 
                             'duration', 'total_bytes', 'total_frames'])
        
        # Collect all raw data that matches our criteria
        raw_data = {}  # (alg, cert, mode, network) -> {metric_name: value}
        available_metrics = set()
        
        for item in all_data:
            # Skip if scenario doesn't match
            if item['scenario'] != scenario:
                continue
            
            # Get metric name as string (important for category objects)
            metric_name = str(item['metric'])
            
            # Skip if not in requested metrics
            if metric_name not in metrics:
                continue
                
            # Add to available metrics set
            available_metrics.add(metric_name)
                
            # Skip if security mode doesn't match
            if item['mode'] not in security_modes:
                continue
                
            # For PKI, filter by algorithms and certificates
            if item['mode'] == 'pki':
                if algorithms and item['alg'] not in algorithms:
                    continue
                if certificates and item['cert'] not in certificates:
                    continue
            # For PSK, filter only by algorithms
            elif item['mode'] == 'psk':
                if algorithms and item['alg'] not in algorithms:
                    continue
                    
            # Create configuration key
            config_key = (item['alg'], item['cert'], item['mode'], item['network'])
            
            # Initialize dictionary for this config if needed
            if config_key not in raw_data:
                raw_data[config_key] = {}
                
            # Store the value
            raw_data[config_key][metric_name] = item['mean']
            
            # Apply metric inversion if needed
            if apply_metric_inversion and metric_name in invert_metrics:
                if raw_data[config_key][metric_name] != 0:  # Avoid division by zero
                    raw_data[config_key][metric_name] = 1.0 / raw_data[config_key][metric_name]
        
        # Check what metrics we actually have data for
        metrics_to_use = sorted(list(available_metrics))
        
        if len(metrics_to_use) < 3:
            print(f"Error: Not enough metrics with data. Found: {metrics_to_use}")
            return None
            
        print(f"Using metrics: {metrics_to_use}")
        
        # Group raw data by security configuration
        sec_configs = {}  # (alg, cert, mode) -> set of networks
        
        for config_key in raw_data:
            alg, cert, mode, network = config_key
            sec_key = (alg, cert, mode)
            
            if sec_key not in sec_configs:
                sec_configs[sec_key] = set()
                
            sec_configs[sec_key].add(network)
        
        # Find security configurations that have data for all networks and all metrics
        valid_configs = []
        
        for sec_key, networks_set in sec_configs.items():
            if baseline_network not in networks_set:
                continue  # Skip if missing baseline
                
            if not all(net in networks_set for net in networks):
                continue  # Skip if missing any network
                
            # Check if we have all metrics for all networks
            all_metrics_present = True
            for network in networks_set:
                config_key = sec_key + (network,)
                if config_key not in raw_data:
                    all_metrics_present = False
                    break
                    
                if not all(m in raw_data[config_key] for m in metrics_to_use):
                    all_metrics_present = False
                    break
                    
            if all_metrics_present:
                valid_configs.append(sec_key)
        
        # If we still don't have any valid configs, we need to relax our requirements
        if not valid_configs:
            print("Warning: No configurations have complete data. Using partial data instead.")
            
            # Just use configs that have the baseline and at least one non-baseline network
            for sec_key, networks_set in sec_configs.items():
                if baseline_network not in networks_set:
                    continue  # Must have baseline
                    
                if not any(net in networks_set for net in networks):
                    continue  # Must have at least one other network
                    
                valid_configs.append(sec_key)
        
        # Sort configs to ensure consistent order
        valid_configs.sort()
        
        # Limit to max number of configs
        if len(valid_configs) > max_security_configs:
            print(f"Limiting from {len(valid_configs)} to {max_security_configs} configurations")
            valid_configs = valid_configs[:max_security_configs]
            
        if not valid_configs:
            print("Error: No valid configurations found.")
            return None
            
        print(f"Using {len(valid_configs)} security configurations")
        
        # Normalize the data with a much more robust approach
        norm_data = {}
        
        # For each metric, collect all available values first
        metric_values = {}
        for metric in metrics_to_use:
            metric_values[metric] = []
            
            for config_key, metrics_dict in raw_data.items():
                if metric in metrics_dict:
                    if pd.notna(metrics_dict[metric]):  # Skip NaN values
                        metric_values[metric].append(metrics_dict[metric])
        
        # Print information about available values
        for metric in metrics_to_use:
            if metric_values[metric]:
                print(f"Metric {metric}: {len(metric_values[metric])} values, "
                    f"range: {min(metric_values[metric]):.6g} to {max(metric_values[metric]):.6g}")
            else:
                print(f"Warning: No values found for metric {metric}")
        
        # Apply normalizations - with detailed error messages and fallbacks
        if normalization == 'min_max':
            print("Using min-max normalization")
            for metric in metrics_to_use:
                norm_data[metric] = {}
                
                if not metric_values[metric]:
                    continue
                    
                try:
                    min_val = min(metric_values[metric])
                    max_val = max(metric_values[metric])
                    range_val = max_val - min_val
                    
                    if range_val <= 0:
                        print(f"Warning: Zero range for metric {metric} - using constant value")
                        # Use constant 0.5 for all points
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
                        continue
                    
                    # Apply min-max normalization
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                norm_val = (metrics_dict[metric] - min_val) / range_val
                                # Sometimes small floating point errors can push outside [0,1]
                                norm_val = max(0.0, min(1.0, norm_val))
                                norm_data[metric][config_key] = norm_val
                            except Exception as e:
                                print(f"Error normalizing {metric} for {config_key}: {e}")
                                # Use 0.5 as fallback
                                norm_data[metric][config_key] = 0.5
                except Exception as e:
                    print(f"Error in min-max normalization for {metric}: {e}")
                    # Use raw values scaled to [0,1] as fallback
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                if max(metric_values[metric]) > 0:
                                    norm_data[metric][config_key] = metrics_dict[metric] / max(metric_values[metric])
                                else:
                                    norm_data[metric][config_key] = 0.5
                            except:
                                norm_data[metric][config_key] = 0.5
        
        elif normalization == 'max_percent':
            print("Using max-percent normalization")
            for metric in metrics_to_use:
                norm_data[metric] = {}
                
                if not metric_values[metric]:
                    continue
                    
                try:
                    max_val = max(metric_values[metric])
                    
                    if max_val <= 0:
                        print(f"Warning: Zero maximum for metric {metric} - using constant value")
                        # Use constant 0.5 for all points
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
                        continue
                    
                    # Apply max-percent normalization
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                norm_val = metrics_dict[metric] / max_val
                                # Sometimes small floating point errors can push outside [0,1]
                                norm_val = max(0.0, min(1.0, norm_val))
                                norm_data[metric][config_key] = norm_val
                            except Exception as e:
                                print(f"Error normalizing {metric} for {config_key}: {e}")
                                # Use 0.5 as fallback
                                norm_data[metric][config_key] = 0.5
                except Exception as e:
                    print(f"Error in max-percent normalization for {metric}: {e}")
                    # Use 0.5 as fallback for all values
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            norm_data[metric][config_key] = 0.5
        
        elif normalization == 'log':
            print("Using log normalization")
            for metric in metrics_to_use:
                norm_data[metric] = {}
                
                if not metric_values[metric]:
                    continue
                    
                try:
                    min_val = min(metric_values[metric])
                    max_val = max(metric_values[metric])
                    
                    # Handle zero/negative values with an offset
                    offset = 0
                    if min_val <= 0:
                        offset = abs(min_val) + 1e-6
                    
                    # Handle cases where all values are the same
                    if max_val + offset == min_val + offset:
                        print(f"Warning: All values the same for {metric} after log offset")
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
                        continue
                    
                    # Apply log transformation
                    log_min = np.log(min_val + offset)
                    log_max = np.log(max_val + offset)
                    log_range = log_max - log_min
                    
                    if log_range <= 0:
                        print(f"Warning: Zero log range for metric {metric} - using constant value")
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
                        continue
                    
                    # Apply log normalization
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                log_val = np.log(metrics_dict[metric] + offset)
                                norm_val = (log_val - log_min) / log_range
                                # Sometimes small floating point errors can push outside [0,1]
                                norm_val = max(0.0, min(1.0, norm_val))
                                norm_data[metric][config_key] = norm_val
                            except Exception as e:
                                print(f"Error log-normalizing {metric} for {config_key}: {e}")
                                # Use 0.5 as fallback
                                norm_data[metric][config_key] = 0.5
                except Exception as e:
                    print(f"Error in log normalization for {metric}: {e}")
                    # Use raw values scaled to [0,1] as fallback
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                if max(metric_values[metric]) > 0:
                                    norm_data[metric][config_key] = metrics_dict[metric] / max(metric_values[metric])
                                else:
                                    norm_data[metric][config_key] = 0.5
                            except:
                                norm_data[metric][config_key] = 0.5
        
        elif normalization == 'z_score':
            print("Using z-score normalization")
            for metric in metrics_to_use:
                norm_data[metric] = {}
                
                if not metric_values[metric]:
                    continue
                    
                if len(metric_values[metric]) < 2:
                    print(f"Warning: Not enough values for z-score of {metric} - using constant")
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            norm_data[metric][config_key] = 0.5
                    continue
                    
                try:
                    mean_val = np.mean(metric_values[metric])
                    std_val = np.std(metric_values[metric])
                    
                    if std_val <= 0:
                        print(f"Warning: Zero std dev for metric {metric} - using constant value")
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
                        continue
                    
                    # Apply z-score normalization
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                z = (metrics_dict[metric] - mean_val) / std_val
                                # Map z-scores to [0,1], assuming most values fall in [-3,3]
                                norm_val = (z + 3) / 6
                                # Clamp to [0,1]
                                norm_val = max(0.0, min(1.0, norm_val))
                                norm_data[metric][config_key] = norm_val
                            except Exception as e:
                                print(f"Error z-score normalizing {metric} for {config_key}: {e}")
                                # Use 0.5 as fallback
                                norm_data[metric][config_key] = 0.5
                except Exception as e:
                    print(f"Error in z-score normalization for {metric}: {e}")
                    # Use raw values scaled to [0,1] as fallback
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            try:
                                if max(metric_values[metric]) > 0:
                                    norm_data[metric][config_key] = metrics_dict[metric] / max(metric_values[metric])
                                else:
                                    norm_data[metric][config_key] = 0.5
                            except:
                                norm_data[metric][config_key] = 0.5
                                
        elif normalization == 'baseline_relative':
            print("Using baseline-relative normalization")
            for metric in metrics_to_use:
                norm_data[metric] = {}
                
                if not metric_values[metric]:
                    continue
                
                # Count how many configs we normalized successfully 
                successful_normalizations = 0
                
                # For each valid configuration
                for sec_key in valid_configs:
                    # Get baseline key
                    baseline_key = sec_key + (baseline_network,)
                    
                    # Skip if baseline doesn't exist
                    if baseline_key not in raw_data or metric not in raw_data[baseline_key]:
                        continue
                        
                    baseline_val = raw_data[baseline_key][metric]
                    
                    # Skip if baseline is zero
                    if baseline_val == 0:
                        continue
                    
                     # Process baseline network specifically
                    config_key = baseline_key
                    if config_key in raw_data and metric in raw_data[config_key]:
                        # Set baseline to 1.0 
                        norm_data[metric][config_key] = 1.0
                        successful_normalizations += 1
                    
                    # Apply normalization for all networks with this config
                    for network in networks:
                        config_key = sec_key + (network,)
                        
                        if config_key not in raw_data or metric not in raw_data[config_key]:
                            continue
                        
                        try:
                            # Get ratio to baseline
                            val = raw_data[config_key][metric]
                            ratio = val / baseline_val
                            
                            # Scale to [0,1.1] with 1.0 as the baseline point
                            if ratio < 1:
                                # Worse than baseline: scale to [0, 1.0)
                                norm_data[metric][config_key] = ratio
                            elif ratio > 1:
                                # Better than baseline: scale to (1.0, 1.1]
                                # Cap improvement at 10% beyond baseline
                                norm_data[metric][config_key] = min(ratio, 1.1)
                            else:
                                # Equal to baseline
                                norm_data[metric][config_key] = 1.0
                                
                            successful_normalizations += 1
                        except Exception as e:
                            print(f"Error in baseline relative normalization for {metric}, {config_key}: {e}")
                            # Use 0.5 as fallback
                            norm_data[metric][config_key] = 0.5
                
                print(f"Baseline-relative normalization for {metric}: {successful_normalizations} successful normalizations")
                
                # If baseline normalization failed for everything, fall back to min-max
                if successful_normalizations == 0:
                    print(f"Warning: No successful baseline normalizations for {metric}. Falling back to min-max.")
                    
                    try:
                        min_val = min(metric_values[metric])
                        max_val = max(metric_values[metric])
                        range_val = max_val - min_val
                        
                        if range_val <= 0:
                            # All values the same
                            for config_key, metrics_dict in raw_data.items():
                                if metric in metrics_dict:
                                    norm_data[metric][config_key] = 0.5
                        else:
                            # Apply min-max normalization as fallback
                            for config_key, metrics_dict in raw_data.items():
                                if metric in metrics_dict:
                                    try:
                                        norm_val = (metrics_dict[metric] - min_val) / range_val
                                        norm_data[metric][config_key] = norm_val
                                    except:
                                        norm_data[metric][config_key] = 0.5
                    except Exception as e:
                        print(f"Error in fallback normalization for {metric}: {e}")
                        # Just use constants as last resort
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
        else:
            print(f"Unknown normalization method: {normalization}, using max percent as fallback")
            # Fall back to max percent normalization
            for metric in metrics_to_use:
                norm_data[metric] = {}
                
                if not metric_values[metric]:
                    continue
                    
                try:
                    max_val = max(metric_values[metric])
                    
                    if max_val <= 0:
                        # Use constant
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                norm_data[metric][config_key] = 0.5
                    else:
                        # Use max percent
                        for config_key, metrics_dict in raw_data.items():
                            if metric in metrics_dict:
                                try:
                                    norm_data[metric][config_key] = metrics_dict[metric] / max_val
                                except:
                                    norm_data[metric][config_key] = 0.5
                except Exception as e:
                    print(f"Error in fallback normalization for {metric}: {e}")
                    # Use constants
                    for config_key, metrics_dict in raw_data.items():
                        if metric in metrics_dict:
                            norm_data[metric][config_key] = 0.5
                            
        # Final safety check - ensure EVERY metric and config has a value
        for metric in metrics_to_use:
            if metric not in norm_data:
                norm_data[metric] = {}
            
            for config_key, metrics_dict in raw_data.items():
                if metric in metrics_dict and config_key not in norm_data[metric]:
                    print(f"Warning: Missing normalized value for {metric}, {config_key}. Using 0.5")
                    norm_data[metric][config_key] = 0.5
        
        # Setup the plot - FIX FOR POLAR PLOT ISSUES
        fig = plt.figure(figsize=(14, 10))
        
        # Number of metrics
        N = len(metrics_to_use)
        
        # Calculate angles for each metric (evenly spaced)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # *** NO ROTATION - METRICS ON AXES ***
        # We won't rotate the polar coordinates as that seems to cause issues
        
        # Create subplot using standard axes - IMPORTANT CHANGE
        ax = fig.add_subplot(111, projection='polar')
        
        # Set direction to clockwise
        ax.set_theta_direction(-1)
        
        # Start at 90 degrees (top)
        ax.set_theta_offset(np.pi/2)
        
        # Draw axis lines for each variable
        for i in range(N):
            ax.plot([theta[i], theta[i]], [0, 1.1], 'k-', linewidth=0.5)
        
        # Add rings at different levels
        if normalization == 'baseline_relative':
            # Add rings at 0.25, 0.5, 0.75 and 1.0 (baseline)
            for level in [0.25, 0.5, 0.75]:
                ax.plot(np.linspace(0, 2*np.pi, 100), [level]*100, 'k--', alpha=0.3)
            # Add solid line for baseline
            ax.plot(np.linspace(0, 2*np.pi, 100), [1.0]*100, 'g-', alpha=0.5)
        else:
            # Standard rings for other normalizations
            for level in [0.25, 0.5, 0.75]:
                ax.plot(np.linspace(0, 2*np.pi, 100), [level]*100, 'k--', alpha=0.3)
        
        # Set limits - ensure the plot has the full range
        if normalization == 'baseline_relative':
            ax.set_ylim(0, 1.1)  # Allow space for values up to 10% better than baseline
        else:
            ax.set_ylim(0, 1)  # Standard range for other normalizations
        
        # Set ticks and labels for each variable
        ax.set_xticks(theta)
        ax.set_xticklabels([format_labels(m) for m in metrics_to_use])
        # Move all labels outward
        ax.tick_params(axis='x', pad=30)  # Increase this number for more spacing
                
        # Define colors and styles
        network_colors = {
            'fiducial': 'forestgreen',
            'smarthome': 'royalblue',
            'smartfactory': 'darkorchid',
            'publictransport': 'crimson'
        }
        
        # Define line styles for different security configurations
        line_styles = ['-', '--', '-.', ':']
        marker_styles = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X']
        
        # Plot the data
        legend_handles = []
        legend_labels = []
        
        # Determine which networks to plot (skip baseline if using baseline_relative)
        plot_networks = networks if normalization == 'baseline_relative' else networks + [baseline_network]
        
        # For each valid configuration
        for i, sec_key in enumerate(valid_configs):
            alg, cert, mode = sec_key
            
            # Choose line style
            line_style = line_styles[i % len(line_styles)]
            marker_style = marker_styles[i % len(marker_styles)]
            
            # Get readable config name
            if mode == 'pki':
                config_label = f"{alg} + {cert}"
            elif mode == 'psk':
                config_label = f"{alg} (PSK)"
            else:
                config_label = "NOSEC"
            
            # For each network
            for network in plot_networks:
                # Create full key
                full_key = sec_key + (network,)
                
                # Skip if any metric is missing for this config
                skip = False
                values = []
                
                for metric in metrics_to_use:
                    if metric not in norm_data or full_key not in norm_data[metric]:
                        skip = True
                        break
                    
                    values.append(norm_data[metric][full_key])
                
                if skip:
                    continue
                
                # *** IMPORTANT FIX: Ensure values form a complete loop ***
                # Close the loop by appending first value
                values_for_plot = np.concatenate((values, [values[0]]))
                theta_for_plot = np.concatenate((theta, [theta[0]]))
                
                # Plot this configuration
                color = network_colors.get(network, 'gray')
                
                # Plot the line
                line = ax.plot(theta_for_plot, values_for_plot, linestyle=line_style, marker=marker_style,
                         color=color, linewidth=2, alpha=0.7, markersize=8)[0]
                
                # Fill area
                ax.fill(theta_for_plot, values_for_plot, color=color, alpha=0.1)
                
                # Add to legend
                legend_handles.append(line)
                legend_labels.append(f"{network} - {config_label}")
        
        # Add title
        title = f"Impact of Network Conditions on {security_modes} - Scenario {scenario}"
        ax.set_title(title, pad=25)
        
        # Add legend with better placement
        if include_legend:
            fig.subplots_adjust(right=0.7)  # Make room for legend
            ax.legend(legend_handles, legend_labels, loc='best',
                     bbox_to_anchor=(1.1, 0.5))
            
        # Add explanation for baseline_relative
        #if normalization == 'baseline_relative':
        #    plt.figtext(0.5, 0.01, f"Values are relative to {baseline_network} baseline\n"
        #            f"1.0 = Equal to baseline, <1.0 = Worse, >1.0 = Better",
        #            ha='center', va='center', fontsize=10)
        
        # Generate output filename
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics string for filename
        metrics_str = "_".join(m.replace(" ", "").replace("(", "").replace(")", "") for m in metrics_to_use)
        if len(metrics_str) > 50:
            metrics_str = f"{len(metrics_to_use)}_metrics"
            
        security_str = "_".join(security_modes)
        
        filename = f"{output_dir}/spider_{scenario}_{security_str}_{normalization}_{metrics_str}.pdf"
        
        # Save figure
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved spider plot to: {filename}")
        
        return filename
    
    except Exception as e:
        import traceback
        print(f"Error in create_spider_plot: {str(e)}")
        traceback.print_exc()
        plt.close()
        return None
    
def create_network_difference_plot(all_data, metric, scenario, 
                                  baseline_network='fiducial',
                                  comparison_networks=None,
                                  sample_size=15,
                                  security_modes=['pki', 'psk', 'nosec'],
                                  algorithms=None, certificates=None,
                                  alpha=0.05, output_dir='./bench-plots-compare'):
    """
    Create difference plots showing network impact on energy consumption
    with proper error propagation and statistical significance testing.
    
    This function calculates the difference between baseline network (fiducial)
    and other networks for each security configuration, providing a clean
    way to analyze network impact while handling outliers properly.
    
    Parameters:
    -----------
    all_data : list
        The benchmark data loaded from load_all()
    metric : str
        Metric to analyze (e.g., 'Energy (Wh)', 'duration', 'cpu_cycles')
    scenario : str
        Scenario identifier ('A' or 'C')
    baseline_network : str
        Network to use as baseline (default: 'fiducial')
    comparison_networks : list or None
        Networks to compare against baseline (default: ['smarthome'])
    security_modes : list
        Security modes to include ('pki', 'psk', 'nosec')
    algorithms : list or None
        Algorithms to include (default: all from global list)
    certificates : list or None
        Certificate types to include (default: all from global list)
    alpha : float
        Significance level for statistical tests (default: 0.05)
    output_dir : str
        Directory to save plots and reports
    
    Returns:
    --------
    dict : Dictionary containing difference statistics and file paths
    """
    
    # Set defaults
    if comparison_networks is None:
        comparison_networks = ['smarthome']
    if algorithms is None:
        algorithms = ['KYBER_LEVEL1', 'KYBER_LEVEL3', 'KYBER_LEVEL5']
    if certificates is None:
        certificates = ['RSA_2048', 'EC_P256', 'EC_ED25519', 
                       'DILITHIUM_LEVEL2', 'DILITHIUM_LEVEL3', 'DILITHIUM_LEVEL5',
                       'FALCON_LEVEL1', 'FALCON_LEVEL5']
    
    print(f"Creating network difference plot for {metric} (Scenario {scenario})")
    print(f"Baseline: {baseline_network}")
    print(f"Comparisons: {comparison_networks}")
    
    # Convert data to easier format for processing
    data_by_config = {}
    
    for item in all_data:
        if (item['scenario'] != scenario or 
            item['metric'] != metric or
            item['network'] not in [baseline_network] + comparison_networks):
            continue
        
        # Create configuration key
        config_key = (item['alg'], item['cert'], item['mode'])
        
        # Initialize if needed
        if config_key not in data_by_config:
            data_by_config[config_key] = {}
        
        # Store data by network
        data_by_config[config_key][item['network']] = {
            'mean': item['mean'],
            'std': item['std']
        }
    
    # Calculate differences for each comparison network
    all_differences = []
    
    for comp_network in comparison_networks:
        print(f"\nProcessing differences: {comp_network} - {baseline_network}")
        
        network_differences = []
        
        for config_key, network_data in data_by_config.items():
            alg, cert, mode = config_key
            
            # Skip if not in requested security modes
            if mode not in security_modes:
                continue
                
            # Skip if specific algorithms/certificates requested and not in list
            if mode == 'pki':
                if (algorithms and alg not in algorithms) or (certificates and cert not in certificates):
                    continue
            elif mode == 'psk':
                if algorithms and alg not in algorithms:
                    continue
            # nosec always included if in security_modes
            
            # Check if we have data for both networks
            if baseline_network not in network_data or comp_network not in network_data:
                continue
            
            baseline_data = network_data[baseline_network]
            comparison_data = network_data[comp_network]
            
            # Calculate difference with error propagation
            diff_stats = calculate_network_difference_statistics(
                comparison_data, baseline_data, sample_size, config_key, comp_network, baseline_network
            )
            
            if diff_stats:
                network_differences.append(diff_stats)
        
        if network_differences:
            all_differences.extend(network_differences)
    
    if not all_differences:
        print("No valid differences found for plotting")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort differences by configuration for consistent plotting
    all_differences.sort(key=lambda x: (x['mode'], x['alg'] or 'zzz', x['cert'] or 'zzz'))
    
    # Prepare plot data
    positions = np.arange(len(all_differences))
    differences = [d['mean_diff'] for d in all_differences]
    ci_lower = [d['ci_lower'] for d in all_differences]
    ci_upper = [d['ci_upper'] for d in all_differences]
    p_values = [d['p_value'] for d in all_differences]
    labels = [d['config_label'] for d in all_differences]
    
    # Color based on statistical significance and magnitude
    colors = []
    for diff, p_val in zip(differences, p_values):
        if p_val < alpha:
            # Significant: red for positive (network penalty), blue for negative (network benefit)
            colors.append('#d32f2f' if diff > 0 else '#1976d2')
        else:
            # Not significant: gray
            colors.append('#757575')
    
    # Create bar plot
    bars = ax.bar(positions, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add confidence intervals
    err_lower = np.array(differences) - np.array(ci_lower)
    err_upper = np.array(ci_upper) - np.array(differences)
    ax.errorbar(positions, differences, yerr=[err_lower, err_upper],
                fmt='none', color='black', capsize=3, capthick=1)
    
    # Add horizontal line at zero (no difference)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    # Add significance indicators
    max_diff = max(ci_upper) if ci_upper else max(differences) if differences else 1
    min_diff = min(ci_lower) if ci_lower else min(differences) if differences else -1
    y_range = max_diff - min_diff
    
    for i, (pos, diff, p_val, ci_u) in enumerate(zip(positions, differences, p_values, ci_upper)):
        if p_val < alpha:
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            # Position significance marker above error bar
            y_pos = ci_u + y_range * 0.02
            ax.text(pos, y_pos, significance, ha='center', va='bottom', 
                   fontweight='bold', color='red', fontsize=8)
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'Network Impact on {format_labels(metric)} (Comparison - Baseline)')
    ax.set_xlabel('Security Configuration')
    
    #ax.set_yscale('symlog')
    
    # Create title
    comp_networks_str = ' vs '.join(comparison_networks)
    title = f'Network Impact: {comp_networks_str} vs {baseline_network}\n'
    title += f'{format_labels(metric)} (Scenario {scenario})'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#d32f2f', alpha=0.7, label=f'Significant increase (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#1976d2', alpha=0.7, label=f'Significant decrease (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#757575', alpha=0.7, label=f'No significant difference (p $\\geq$ {alpha})')
    ]
    ax.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1.0, 1.0))
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Generate filename
    os.makedirs(output_dir, exist_ok=True)
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    comp_str = '_'.join(comparison_networks)
    filename = f"{output_dir}/network_difference_{clean_metric}_{security_modes[:]}_{scenario}_{comp_str}_vs_{baseline_network}.pdf"
    
    # Save plot
    plt.savefig(filename, bbox_inches='tight')
    print(f"Network difference plot saved to: {filename}")
    
    # Generate statistical report
    report = generate_network_difference_report(all_differences, metric, scenario, 
                                              baseline_network, comparison_networks, sample_size, alpha)
    
    # Save report
    report_file = filename.replace('.pdf', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Statistical report saved to: {report_file}")
    
    # Show plot
    plt.show()
    
    # Return results
    return {
        'differences': all_differences,
        'plot_file': filename,
        'report_file': report_file,
        'significant_differences': [d for d in all_differences if d['p_value'] < alpha],
        'total_comparisons': len(all_differences),
        'alpha': alpha
    }

def calculate_network_difference_statistics(comparison_data, baseline_data, sample_size, config_key, comp_network, baseline_network):
    """
    Calculate statistical difference between network conditions with proper error propagation.
    
    Parameters:
    -----------
    comparison_data : dict
        Data from comparison network {'mean': float, 'std': float}
    baseline_data : dict
        Data from baseline network {'mean': float, 'std': float}
    config_key : tuple
        (algorithm, certificate, mode) configuration identifier
    comp_network : str
        Name of comparison network
    baseline_network : str
        Name of baseline network
    
    Returns:
    --------
    dict : Statistical difference analysis
    """
    
    try:
        alg, cert, mode = config_key
        
        # Extract values
        mean_comp = float(comparison_data['mean'])
        std_comp = float(comparison_data['std'])
        mean_base = float(baseline_data['mean'])
        std_base = float(baseline_data['std'])
        
        # Calculate difference
        mean_diff = mean_comp - mean_base
        
        # Error propagation for difference (assuming independent measurements)
        # For difference A - B: σ_diff = √(σ_A² + σ_B²)
        std_diff = np.sqrt(std_comp**2 + std_base**2)
        
        # Assume sample size (you might want to make this configurable)
        n = sample_size  # Based on your experimental setup
        
        # Standard error of the difference
        se_diff = std_diff / np.sqrt(n)
        
        # t-statistic for testing if difference is significantly different from zero
        t_stat = mean_diff / se_diff if se_diff > 0 else 0
        
        # Degrees of freedom (conservative estimate)
        df = n - 1
        
        # p-value (two-tailed test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if se_diff > 0 else 1.0
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, df)
        margin_error = t_critical * se_diff
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        # Effect size (Cohen's d - standardized difference)
        pooled_std = np.sqrt((std_comp**2 + std_base**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Create readable configuration label
        if mode == 'nosec':
            config_label = 'NoSec'
        elif mode == 'psk':
            config_label = f'{alg} (PSK)' if alg else 'PSK'
        elif mode == 'pki':
            if alg and cert:
                # Shorten labels for readability
                alg_short = alg.replace('KYBER_LEVEL', 'K').replace('_', '')
                cert_short = cert.replace('DILITHIUM_LEVEL', 'D').replace('FALCON_LEVEL', 'F').replace('_', '')
                config_label = f'{alg_short}+{cert_short}'
            else:
                config_label = 'PKI'
        else:
            config_label = f'{alg or ""}+{cert or ""}'
        
        return {
            'config_key': config_key,
            'alg': alg,
            'cert': cert,
            'mode': mode,
            'config_label': config_label,
            'comparison_network': comp_network,
            'baseline_network': baseline_network,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'se_diff': se_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'cohens_d': cohens_d,
            'mean_comparison': mean_comp,
            'mean_baseline': mean_base,
            'std_comparison': std_comp,
            'std_baseline': std_base
        }
        
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error calculating difference statistics for {config_key}: {e}")
        return None

def generate_network_difference_report(differences, metric, scenario, baseline_network, comparison_networks, sample_size, alpha):
    """Generate comprehensive text report for network difference analysis."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""NETWORK DIFFERENCE ANALYSIS REPORT
{'=' * 70}
Metric: {format_labels(metric)}
Scenario: {scenario}
Baseline Network: {baseline_network}
Comparison Networks: {', '.join(comparison_networks)}
Significance Level: α = {alpha}
Generated: {timestamp}

SUMMARY STATISTICS
{'-' * 40}
"""
    
    significant_diffs = [d for d in differences if d['p_value'] < alpha]
    positive_diffs = [d for d in significant_diffs if d['mean_diff'] > 0]
    negative_diffs = [d for d in significant_diffs if d['mean_diff'] < 0]
    
    report += f"""Total comparisons: {len(differences)}
Significant differences: {len(significant_diffs)} ({len(significant_diffs)/len(differences)*100:.1f}%)
  - Network penalties (positive): {len(positive_diffs)}
  - Network benefits (negative): {len(negative_diffs)}
Non-significant differences: {len(differences) - len(significant_diffs)}

"""
    
    if positive_diffs:
        report += f"""SIGNIFICANT NETWORK PENALTIES (p < {alpha})
{'-' * 60}
These configurations show significantly higher energy consumption
under network stress compared to baseline conditions.

"""
        
        for diff in sorted(positive_diffs, key=lambda x: x['mean_diff'], reverse=True):
            effect_size = interpret_cohens_d(diff['cohens_d'])
            report += f"""
{diff['config_label']} ({diff['comparison_network']} vs {diff['baseline_network']}):
  Network penalty: {diff['mean_diff']:.4f} ± {diff['se_diff']:.4f}
  95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
  t-statistic: {diff['t_statistic']:.3f} (df = {diff['degrees_freedom']})
  p-value: {diff['p_value']:.6f}
  Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
  Baseline: {diff['mean_baseline']:.4f}, Network: {diff['mean_comparison']:.4f}
"""
    
    if negative_diffs:
        report += f"""
SIGNIFICANT NETWORK BENEFITS (p < {alpha})
{'-' * 60}
These configurations show significantly lower energy consumption
under network stress (unexpected but possible).

"""
        
        for diff in sorted(negative_diffs, key=lambda x: x['mean_diff']):
            effect_size = interpret_cohens_d(diff['cohens_d'])
            report += f"""
{diff['config_label']} ({diff['comparison_network']} vs {diff['baseline_network']}):
  Network benefit: {diff['mean_diff']:.4f} ± {diff['se_diff']:.4f}
  95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
  t-statistic: {diff['t_statistic']:.3f} (df = {diff['degrees_freedom']})
  p-value: {diff['p_value']:.6f}
  Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
  Baseline: {diff['mean_baseline']:.4f}, Network: {diff['mean_comparison']:.4f}
"""
    
    if not significant_diffs:
        report += f"""NO SIGNIFICANT NETWORK EFFECTS FOUND
{'-' * 60}
All configurations show no statistically significant difference
between baseline and network conditions. This could indicate:
1. Network conditions have minimal impact on energy consumption
2. Measurement variability is too high to detect differences
3. The chosen network conditions are too mild to cause measurable effects
"""
    
    # Non-significant differences summary
    non_significant = [d for d in differences if d['p_value'] >= alpha]
    if non_significant:
        report += f"""
NON-SIGNIFICANT DIFFERENCES (p ≥ {alpha})
{'-' * 60}
"""
        for diff in sorted(non_significant, key=lambda x: x['p_value']):
            report += f"{diff['config_label']}: Δ = {diff['mean_diff']:.4f}, p = {diff['p_value']:.3f}\n"
    
    report += f"""
INTERPRETATION GUIDELINES
{'-' * 40}
Statistical Significance (p-value):
  p < 0.001: Very strong evidence of network effect
  p < 0.01:  Strong evidence of network effect
  p < 0.05:  Moderate evidence of network effect
  p ≥ 0.05:  No significant network effect detected

Effect Size (Cohen's d):
  |d| < 0.2:  Small effect
  |d| < 0.5:  Medium effect
  |d| ≥ 0.8:  Large effect

Network Impact Interpretation:
  Positive differences: Network conditions increase energy consumption
  Negative differences: Network conditions decrease energy consumption
  Zero differences: No measurable network impact

METHODOLOGICAL NOTES
{'-' * 40}
• Error propagation: σ_diff = √(σ_baseline² + σ_comparison²)
• Independence assumption: Measurements assumed independent
• Sample size: n = {sample_size} per configuration
• Two-tailed t-test: Tests if difference is significantly different from zero
• Confidence intervals: 95% confidence level
• Multiple comparisons: Consider Bonferroni correction for family-wise error rate

RECOMMENDATIONS
{'-' * 40}
"""
    
    if len(positive_diffs) > len(differences) * 0.5:
        report += "• HIGH NETWORK SENSITIVITY: Most configurations show significant network penalties\n"
        report += "• Consider network-aware deployment strategies\n"
        report += "• Investigate network resilience improvements\n"
    elif len(significant_diffs) > 0:
        report += "• MODERATE NETWORK SENSITIVITY: Some configurations affected by network conditions\n"
        report += "• Focus on robust configurations for deployment\n"
    else:
        report += "• LOW NETWORK SENSITIVITY: Configurations appear network-resilient\n"
        report += "• Current network conditions have minimal impact\n"
    
    if len(significant_diffs) > 1:
        report += f"• MULTIPLE COMPARISONS: Consider Bonferroni correction (α = {alpha/len(differences):.4f})\n"
    
    large_effects = [d for d in significant_diffs if abs(d['cohens_d']) >= 0.8]
    if large_effects:
        report += f"• LARGE EFFECTS DETECTED: {len(large_effects)} configurations show substantial network impact\n"
    
    return report

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
    
def create_algorithm_difference_plot(all_data, metric, scenario,
                                    baseline_algorithm='KYBER_LEVEL1',
                                    comparison_algorithms=None,
                                    sample_size=15,
                                    networks=None,
                                    security_modes=['pki', 'psk'],
                                    certificates=None,
                                    alpha=0.05, output_dir='./bench-plots-compare'):
    """
    Create difference plots showing algorithm scaling impact on performance
    with proper error propagation and statistical significance testing.
    
    This function calculates the difference between baseline algorithm (KYBER_LEVEL1)
    and other algorithms within the same network and security configuration.
    
    Parameters:
    -----------
    all_data : list
        The benchmark data loaded from load_all()
    metric : str
        Metric to analyze (e.g., 'Energy (Wh)', 'duration', 'cpu_cycles')
    scenario : str
        Scenario identifier ('A' or 'C')
    baseline_algorithm : str
        Algorithm to use as baseline (default: 'KYBER_LEVEL1')
    comparison_algorithms : list or None
        Algorithms to compare against baseline (default: ['KYBER_LEVEL3', 'KYBER_LEVEL5'])
    networks : list or None
        Networks to include (default: ['fiducial', 'smarthome'])
    security_modes : list
        Security modes to include ('pki', 'psk', 'nosec')
    certificates : list or None
        Certificate types to include for PKI mode (default: all)
    alpha : float
        Significance level for statistical tests (default: 0.05)
    output_dir : str
        Directory to save plots and reports
    
    Returns:
    --------
    dict : Dictionary containing difference statistics and file paths
    """
    
    # Set defaults
    if comparison_algorithms is None:
        comparison_algorithms = ['KYBER_LEVEL3', 'KYBER_LEVEL5']
    if networks is None:
        networks = ['fiducial', 'smarthome']
    if certificates is None:
        certificates = ['RSA_2048', 'EC_P256', 'EC_ED25519', 
                       'DILITHIUM_LEVEL2', 'DILITHIUM_LEVEL3', 'DILITHIUM_LEVEL5',
                       'FALCON_LEVEL1', 'FALCON_LEVEL5']
    
    print(f"Creating corrected algorithm difference plot for {metric} (Scenario {scenario})")
    print(f"Baseline: {baseline_algorithm}")
    print(f"Comparisons: {comparison_algorithms}")
    print(f"Networks: {networks}")
    
    # Step 1: Extract and organize data by (network, mode, cert) -> {algorithm: data}
    configurations = {}
    
    for item in all_data:
        if (item['scenario'] != scenario or 
            item['metric'] != metric or
            item['network'] not in networks):
            continue
        
        network = item['network']
        mode = item['mode']
        cert = item['cert']  # Will be None for PSK and NoSec
        alg = item['alg']    # Will be None for NoSec
        
        # Skip NoSec mode (no algorithm to compare)
        if mode == 'nosec':
            continue
            
        # Filter by security modes
        if mode not in security_modes:
            continue
            
        # Filter certificates for PKI mode
        if mode == 'pki' and certificates and cert not in certificates:
            continue
            
        # Filter algorithms - must have baseline and at least one comparison
        if alg not in [baseline_algorithm] + comparison_algorithms:
            continue
        
        # Create configuration key
        # For PKI: (network, mode, cert)
        # For PSK: (network, mode, None) - since cert is None for PSK
        config_key = (network, mode, cert)
        
        if config_key not in configurations:
            configurations[config_key] = {}
        
        # Store data by algorithm
        configurations[config_key][alg] = {
            'mean': item['mean'],
            'std': item['std']
        }
    
    print(f"Found {len(configurations)} unique configurations")
    
    # Debug: Show what configurations we found
    for config_key, alg_data in configurations.items():
        network, mode, cert = config_key
        algs = list(alg_data.keys())
        cert_str = cert if cert else "None"
        print(f"  {network} + {mode} + {cert_str}: algorithms {algs}")
    
    # Step 2: Calculate differences for each configuration and comparison algorithm
    all_differences = []
    
    for config_key, algorithm_data in configurations.items():
        network, mode, cert = config_key
        
        # Check if we have baseline algorithm data
        if baseline_algorithm not in algorithm_data:
            print(f"Warning: No {baseline_algorithm} data for {config_key}")
            continue
        
        baseline_data = algorithm_data[baseline_algorithm]
        
        # Calculate differences for each comparison algorithm
        for comp_alg in comparison_algorithms:
            if comp_alg not in algorithm_data:
                continue
                
            comparison_data = algorithm_data[comp_alg]
            
            # Calculate the difference: comparison - baseline
            mean_diff = comparison_data['mean'] - baseline_data['mean']
            
            # Error propagation: sqrt(std_comp^2 + std_base^2)
            std_diff = np.sqrt(comparison_data['std']**2 + baseline_data['std']**2)
            
            # Statistical test (assuming n=15)
            n = sample_size
            se_diff = std_diff / np.sqrt(n)
            
            if se_diff > 0:
                t_stat = mean_diff / se_diff
                df = n - 1
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                # Confidence interval
                t_critical = stats.t.ppf(0.975, df)
                margin_error = t_critical * se_diff
                ci_lower = mean_diff - margin_error
                ci_upper = mean_diff + margin_error
            else:
                t_stat = 0
                p_value = 1.0
                ci_lower = mean_diff
                ci_upper = mean_diff
            
            # Effect size
            pooled_std = np.sqrt((comparison_data['std']**2 + baseline_data['std']**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Create label based on mode
            if mode == 'pki':
                config_label = f"{network} ({cert})"
            elif mode == 'psk':
                config_label = f"{network} (PSK)"
            else:
                config_label = f"{network} ({mode})"
            
            # Store difference data
            all_differences.append({
                'config_key': config_key,
                'network': network,
                'mode': mode,
                'cert': cert,
                'config_label': config_label,
                'comparison_algorithm': comp_alg,
                'baseline_algorithm': baseline_algorithm,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'se_diff': se_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                't_statistic': t_stat,
                'p_value': p_value,
                'degrees_freedom': df,
                'cohens_d': cohens_d,
                'mean_comparison': comparison_data['mean'],
                'mean_baseline': baseline_data['mean'],
                'std_comparison': comparison_data['std'],
                'std_baseline': baseline_data['std']
            })
    
    if not all_differences:
        print("No valid differences found for plotting")
        return None
    
    print(f"Calculated {len(all_differences)} differences")
    
    # Debug: Show first few differences
    for i, diff in enumerate(all_differences[:3]):
        print(f"Difference {i}: {diff['config_label']} {diff['comparison_algorithm']} vs {diff['baseline_algorithm']}")
        print(f"  Mean diff: {diff['mean_diff']:.6f}, p-value: {diff['p_value']:.4f}")
    
    # Step 3: Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort by network, then mode, then cert, then comparison algorithm
    all_differences.sort(key=lambda x: (x['network'], x['mode'], x['cert'] or 'zzz', x['comparison_algorithm']))
    
    # Prepare plot data
    positions = np.arange(len(all_differences))
    differences = [d['mean_diff'] for d in all_differences]
    ci_lower = [d['ci_lower'] for d in all_differences]
    ci_upper = [d['ci_upper'] for d in all_differences]
    p_values = [d['p_value'] for d in all_differences]
    labels = [f"{d['config_label']} ({d['comparison_algorithm']})" for d in all_differences]
    
    # Color based on statistical significance
    colors = []
    for diff, p_val in zip(differences, p_values):
        if p_val < alpha:
            colors.append('#d32f2f' if diff > 0 else '#1976d2')
        else:
            colors.append('#757575')
    
    # Create bar plot
    bars = ax.bar(positions, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add confidence intervals
    err_lower = np.array(differences) - np.array(ci_lower)
    err_upper = np.array(ci_upper) - np.array(differences)
    ax.errorbar(positions, differences, yerr=[err_lower, err_upper],
                fmt='none', color='black', capsize=3, capthick=1)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    # Add significance indicators
    if differences:
        max_diff = max(ci_upper) if ci_upper else max(differences)
        min_diff = min(ci_lower) if ci_lower else min(differences)
        y_range = max_diff - min_diff if max_diff != min_diff else abs(max_diff) * 0.1
        
        for i, (pos, diff, p_val, ci_u) in enumerate(zip(positions, differences, p_values, ci_upper)):
            if p_val < alpha:
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                y_pos = ci_u + y_range * 0.02
                ax.text(pos, y_pos, significance, ha='center', va='bottom', 
                       fontweight='bold', color='red', fontsize=8)
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Algorithm Impact on {format_labels(metric)} (Comparison - Baseline)')
    ax.set_xlabel('Network + Security Configuration + Algorithm')
    
    # Title
    comp_algorithms_str = ' vs '.join(comparison_algorithms)
    title = f'Algorithm Scaling: {comp_algorithms_str} vs {baseline_algorithm}\n'
    title += f'{format_labels(metric)} (Scenario {scenario})'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#d32f2f', alpha=0.7, label=f'Significant increase (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#1976d2', alpha=0.7, label=f'Significant decrease (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#757575', alpha=0.7, label=f'No significant difference (p $\\geq$ {alpha})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    comp_str = '_'.join(comparison_algorithms)
    networks_str = '_'.join(networks)
    filename = f"{output_dir}/algorithm_difference_corrected_{clean_metric}_{scenario}_{comp_str}_vs_{baseline_algorithm}_{networks_str}.pdf"
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Corrected algorithm difference plot saved to: {filename}")
    
    # Generate statistical report
    report = generate_algorithm_difference_report(all_differences, metric, scenario, 
                                                baseline_algorithm, comparison_algorithms, sample_size, alpha)
    
    # Save report
    report_file = filename.replace('.pdf', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Statistical report saved to: {report_file}")
    
    plt.show()
    
    # Return results
    return {
        'differences': all_differences,
        'plot_file': filename,
        'report_file': report_file,
        'significant_differences': [d for d in all_differences if d['p_value'] < alpha],
        'total_comparisons': len(all_differences),
        'alpha': alpha
    }


def calculate_algorithm_difference_statistics(comparison_data, baseline_data, sample_size, config_key, comp_algorithm, baseline_algorithm):
    """
    Calculate statistical difference between algorithms with proper error propagation.
    Similar to network version but adapted for algorithm comparisons.
    
    Parameters:
    -----------
    comparison_data : dict
        Data from comparison algorithm {'mean': float, 'std': float}
    baseline_data : dict
        Data from baseline algorithm {'mean': float, 'std': float}
    config_key : tuple
        (network, certificate, mode) configuration identifier
    comp_algorithm : str
        Name of comparison algorithm
    baseline_algorithm : str
        Name of baseline algorithm
    
    Returns:
    --------
    dict : Statistical difference analysis
    """
    
    try:
        network, cert, mode = config_key
        
        # Extract values
        mean_comp = float(comparison_data['mean'])
        std_comp = float(comparison_data['std'])
        mean_base = float(baseline_data['mean'])
        std_base = float(baseline_data['std'])
        
        # Calculate difference
        mean_diff = mean_comp - mean_base
        
        # Error propagation for difference (assuming independent measurements)
        std_diff = np.sqrt(std_comp**2 + std_base**2)
        
        # Assume sample size (make this configurable if needed)
        n = sample_size  # Based on your experimental setup
        
        # Standard error of the difference
        se_diff = std_diff / np.sqrt(n)
        
        # t-statistic for testing if difference is significantly different from zero
        t_stat = mean_diff / se_diff if se_diff > 0 else 0
        
        # Degrees of freedom (conservative estimate)
        df = n - 1
        
        # p-value (two-tailed test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if se_diff > 0 else 1.0
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, df)
        margin_error = t_critical * se_diff
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        # Effect size (Cohen's d - standardized difference)
        pooled_std = np.sqrt((std_comp**2 + std_base**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Create readable configuration label
        if mode == 'nosec':
            config_label = f'{network} (NoSec)'
        elif mode == 'psk':
            config_label = f'{network} (PSK)'
        elif mode == 'pki':
            if cert:
                # Shorten labels for readability
                cert_short = cert.replace('DILITHIUM_LEVEL', 'D').replace('FALCON_LEVEL', 'F').replace('_', '')
                config_label = f'{network} ({cert_short})'
            else:
                config_label = f'{network} (PKI)'
        else:
            config_label = f'{network} ({mode})'
        
        return {
            'config_key': config_key,
            'network': network,
            'cert': cert,
            'mode': mode,
            'config_label': config_label,
            'comparison_algorithm': comp_algorithm,
            'baseline_algorithm': baseline_algorithm,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'se_diff': se_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'cohens_d': cohens_d,
            'mean_comparison': mean_comp,
            'mean_baseline': mean_base,
            'std_comparison': std_comp,
            'std_baseline': std_base
        }
        
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error calculating algorithm difference statistics for {config_key}: {e}")
        return None


def generate_algorithm_difference_report(differences, metric, scenario, baseline_algorithm, comparison_algorithms, sample_size, alpha):
    """Generate comprehensive text report for algorithm difference analysis."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""ALGORITHM DIFFERENCE ANALYSIS REPORT
{'=' * 70}
Metric: {format_labels(metric)}
Scenario: {scenario}
Baseline Algorithm: {baseline_algorithm}
Comparison Algorithms: {', '.join(comparison_algorithms)}
Sample size: {sample_size}
Significance Level: α = {alpha}
Generated: {timestamp}

SUMMARY STATISTICS
{'-' * 40}
"""
    
    significant_diffs = [d for d in differences if d['p_value'] < alpha]
    positive_diffs = [d for d in significant_diffs if d['mean_diff'] > 0]
    negative_diffs = [d for d in significant_diffs if d['mean_diff'] < 0]
    
    report += f"""Total comparisons: {len(differences)}
Significant differences: {len(significant_diffs)} ({len(significant_diffs)/len(differences)*100:.1f}%)
  - Algorithm penalties (positive): {len(positive_diffs)}
  - Algorithm benefits (negative): {len(negative_diffs)}
Non-significant differences: {len(differences) - len(significant_diffs)}

"""
    
    if positive_diffs:
        report += f"""SIGNIFICANT ALGORITHM PENALTIES (p < {alpha})
{'-' * 60}
These configurations show significantly higher resource consumption
with larger algorithms compared to {baseline_algorithm}.

"""
        
        for diff in sorted(positive_diffs, key=lambda x: x['mean_diff'], reverse=True):
            effect_size = interpret_cohens_d(diff['cohens_d'])
            report += f"""
{diff['config_label']} ({diff['comparison_algorithm']} vs {diff['baseline_algorithm']}):
  Algorithm penalty: {diff['mean_diff']:.4f} ± {diff['se_diff']:.4f}
  95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
  t-statistic: {diff['t_statistic']:.3f} (df = {diff['degrees_freedom']})
  p-value: {diff['p_value']:.6f}
  Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
  {baseline_algorithm}: {diff['mean_baseline']:.4f}, {diff['comparison_algorithm']}: {diff['mean_comparison']:.4f}
"""
    
    if negative_diffs:
        report += f"""
SIGNIFICANT ALGORITHM BENEFITS (p < {alpha})
{'-' * 60}
These configurations show significantly lower resource consumption
with larger algorithms (unexpected but possible due to optimization effects).

"""
        
        for diff in sorted(negative_diffs, key=lambda x: x['mean_diff']):
            effect_size = interpret_cohens_d(diff['cohens_d'])
            report += f"""
{diff['config_label']} ({diff['comparison_algorithm']} vs {diff['baseline_algorithm']}):
  Algorithm benefit: {diff['mean_diff']:.4f} ± {diff['se_diff']:.4f}
  95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
  t-statistic: {diff['t_statistic']:.3f} (df = {diff['degrees_freedom']})
  p-value: {diff['p_value']:.6f}
  Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
  {baseline_algorithm}: {diff['mean_baseline']:.4f}, {diff['comparison_algorithm']}: {diff['mean_comparison']:.4f}
"""
    
    if not significant_diffs:
        report += f"""NO SIGNIFICANT ALGORITHM SCALING EFFECTS FOUND
{'-' * 60}
All configurations show no statistically significant difference
between {baseline_algorithm} and larger algorithms. This could indicate:
1. Algorithm scaling has minimal impact on the measured metric
2. Measurement variability is too high to detect differences
3. Optimizations in larger algorithms compensate for increased complexity
"""
    
    report += f"""
INTERPRETATION GUIDELINES
{'-' * 40}
Algorithm Scaling Analysis:
  Positive differences: Larger algorithms consume more resources
  Negative differences: Larger algorithms consume fewer resources (optimization effects)
  Zero differences: No measurable scaling impact

This analysis helps identify:
• Which security configurations are sensitive to algorithm scaling
• Whether the complexity increase in KYBER_LEVEL3/5 translates to proportional resource usage
• Optimal algorithm choices for resource-constrained deployments

RECOMMENDATIONS
{'-' * 40}
"""
    
    if len(positive_diffs) > len(differences) * 0.7:
        report += "• HIGH SCALING SENSITIVITY: Most configurations show significant penalties with larger algorithms\n"
        report += "• Consider using KYBER_LEVEL1 for resource-constrained environments\n"
        report += "• Evaluate if the security benefits of higher levels justify the resource costs\n"
    elif len(positive_diffs) > len(differences) * 0.3:
        report += "• MODERATE SCALING SENSITIVITY: Some configurations affected by algorithm size\n"
        report += "• Consider configuration-specific algorithm selection\n"
    else:
        report += "• LOW SCALING SENSITIVITY: Algorithm size has minimal resource impact\n"
        report += "• Higher security levels can be used without significant resource penalty\n"
    
    if len(negative_diffs) > 0:
        report += f"• OPTIMIZATION EFFECTS: {len(negative_diffs)} configurations show benefits with larger algorithms\n"
        report += "• This may indicate implementation optimizations or measurement artifacts\n"
    
    return report

def analyze_data_structure(all_data, metric='Energy (Wh)', scenario='A'):
    """
    Analyze how the data is structured to understand the grouping needed.
    """
    print("=== DATA STRUCTURE ANALYSIS ===")
    
    # Sample a few records to understand structure
    sample_records = [item for item in all_data[:10] if item['metric'] == metric and item['scenario'] == scenario]
    
    print("Sample records:")
    for i, record in enumerate(sample_records):
        print(f"Record {i}:")
        for key, value in record.items():
            print(f"  {key}: {value}")
        print()
    
    # Analyze unique combinations
    combinations = set()
    algorithms = set()
    networks = set()
    modes = set()
    certs = set()
    
    for item in all_data:
        if item['metric'] == metric and item['scenario'] == scenario:
            combinations.add((item['network'], item['mode'], item['cert'], item['alg']))
            if item['alg'] is not None:
                algorithms.add(item['alg'])
            if item['network'] is not None:
                networks.add(item['network'])
            if item['mode'] is not None:
                modes.add(item['mode'])
            if item['cert'] is not None:
                certs.add(item['cert'])
    
    print(f"Total combinations found: {len(combinations)}")
    print(f"Networks: {sorted(networks)}")
    print(f"Algorithms: {sorted(algorithms)}")
    print(f"Modes: {sorted(modes)}")
    print(f"Certificates: {sorted([c for c in certs if c])}")
    print()
    
    # Show structure by mode
    print("=== STRUCTURE BY MODE ===")
    for mode in sorted(modes):
        print(f"\n{mode.upper()} mode:")
        mode_combinations = [(net, cert, alg) for net, m, cert, alg in combinations if m == mode]
        mode_combinations.sort()
        
        for net, cert, alg in mode_combinations[:25]:  # Show first 25
            print(f"  {net} + {cert or 'N/A'} + {alg}")
        if len(mode_combinations) > 25:
            print(f"  ... and {len(mode_combinations) - 25} more")

# MAIN
if __name__ == '__main__':
    print("Loading data from all networks and security modes...")
    all_data = load_all()
    
    setup_matplotlib_style(use_latex=True)
    
    print(f"Loaded {len(all_data)} data records")
    
    # Generate summary for continuous metrics
    if not os.path.exists('summary_cts.csv'):
        print("\nCreating summary for continuous metrics...")
        summary_cts = summarize(all_data, metrics=METRIC_CTS)
        summary_cts.to_csv('summary_cts.csv')
        print("Wrote summary_cts.csv with continuous metrics comparison.")
    
    # Generate summary for discrete metrics
    if not os.path.exists('summary_dsc.csv'):
        print("\nCreating summary for discrete metrics...")
        summary_dsc = summarize(all_data, metrics=METRIC_DSC)
        summary_dsc.to_csv('summary_dsc.csv')
        print("Wrote summary_dsc.csv with discrete metrics comparison.")
    
    # Verification
    print("\nVerification - KYBER_LEVEL1 + DILITHIUM_LEVEL2 (PKI, Scenario A, Fiducial):")
    try:
        # Verify continuous
        if os.path.exists('summary_cts.csv'):
            summary_cts = pd.read_csv('summary_cts.csv')
            summary_cts = summary_cts.set_index(['alg', 'cert', 'mode', 'scenario'])
            sample = summary_cts.loc[('KYBER_LEVEL1', 'DILITHIUM_LEVEL2', 'pki', 'A')]
            cpu_mean = sample["('cpu_cycles', 'mean', 'fiducial')"]
            print(f"CPU cycles mean (continuous): {cpu_mean:.0f}")
    except Exception as e:
        print(f"Could not find sample data: {e}")
    
    try:
        # Verify discrete
        if os.path.exists('summary_dsc.csv'):
            summary_dsc = pd.read_csv('summary_dsc.csv')
            summary_dsc = summary_dsc.set_index(['alg', 'cert', 'mode', 'scenario'])
            sample = summary_dsc.loc[('KYBER_LEVEL1', 'DILITHIUM_LEVEL2', 'pki', 'A')]
            frames_mode = sample["('total_frames', 'mode', 'fiducial')"]
            print(f"Total frames mode (discrete): {frames_mode:.0f}")
    except Exception as e:
        print(f"Could not find sample data: {e}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Trade-off plots
    for scen in SCENARIOS:
        #plot_tradeoff(all_data, 'cpu_cycles', 'Energy (Wh)', scen, normalize=True, output_dir=OUT_DIR)
        #plot_tradeoff(all_data, 'total_bytes', 'Energy (Wh)', scen, normalize=True, output_dir=OUT_DIR)
        
        #create_waterfall_plot(all_data, 'Energy (Wh)', scen, plot_type='network', output_dir=OUT_DIR)
        #create_waterfall_plot(all_data, 'Energy (Wh)', scen, plot_type='algorithm', output_dir=OUT_DIR)

        # Generate network comparison plots for different security modes
        #create_network_comparison_plot(all_data, 'Energy (Wh)', scen, 'pki', output_dir=OUT_DIR)
        #create_network_comparison_plot(all_data, 'Energy (Wh)', scen, 'psk', output_dir=OUT_DIR)
        
        pass
            
    # Spider plots
    #for scen in SCENARIOS:
    #    for norm in ['log']:
    #        create_spider_plot(
    #        all_data,
    #            metrics=['cpu_cycles', 'Energy (Wh)', 'Power (W)', 'Max Power (W)', 'bytes_sent', 'bytes_received'],
    #            scenario=scen,
    #            security_modes=['psk'],
    #            algorithms=['KYBER_LEVEL1'],  # Specify algorithm(s)
    #            certificates=['RSA_2048', 'DILITHIUM_LEVEL2', 'FALCON_LEVEL1'],    # Specify certificate(s)
    #            normalization=norm,
    #            output_dir=OUT_DIR
    #        )
            
    for scen in SCENARIOS:
    #    create_network_difference_plot(
    #        all_data=all_data,
    #        metric='Energy (Wh)',
    #        scenario=scen,
    #        sample_size=15,
    #        comparison_networks=['smarthome'],
    #        security_modes=['pki','psk','nosec'],
    #        #algorithms=['KYBER_LEVEL1'],
    #        alpha=0.05,
    #        output_dir=OUT_DIR
    #    )
        
        create_algorithm_difference_plot(
            all_data=all_data,
            metric='Energy (Wh)',
            scenario=scen,
            baseline_algorithm='KYBER_LEVEL1',
            comparison_algorithms=['KYBER_LEVEL3'],
            sample_size=15,
            networks=['smarthome'],
            security_modes=['pki', 'psk'],
            #certificates=['RSA_2048', 'DILITHIUM_LEVEL2'],  # Subset for clarity
            output_dir=OUT_DIR
        )
    
    #analyze_data_structure(all_data=all_data, metric='Energy (Wh)', scenario='A')
    
    print("All plots generated successfully!")