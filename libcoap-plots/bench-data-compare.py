import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import re
from adjustText import adjust_text 

# 1. CONFIGURATION
ROOT_DIR = '.'  # where bench-data-fiducial/ etc. live
NETWORKS = ['fiducial', 'smarthome', 'smartfactory' , 'publictransport']

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
METRIC_CTS = ['cpu_cycles', 'Energy (Wh)', 'Power (W)', 'Max Power (W)']
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
        "font.size": 11,           # Base font size
        "axes.titlesize": 12,       # Title font size
        "axes.labelsize": 11,       # Axis label font size
        "xtick.labelsize": 10,      # x-tick label font size
        "ytick.labelsize": 10,      # y-tick label font size
        "legend.fontsize": 10,      # Legend font size
        "figure.titlesize": 14      # Figure title font size
    })
    
    # Improve figure aesthetics
    plt.rcParams.update({
        "figure.figsize": (10, 6),  # Default figure size
        "figure.dpi": 100,          # Figure resolution
        "savefig.dpi": 300,         # Saved figure resolution
        "savefig.format": "pdf",    # Default save format
        "savefig.bbox": "tight",    # Tight bounding box
        "savefig.pad_inches": 0.1   # Padding
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
        "xtick.major.size": 3.5,    # Major tick size
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2,      # Minor tick size
        "ytick.minor.size": 2,
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
        return r'CPU Cycles ($\times 10^9$)'
    elif "energy" in metric.lower() and "wh" in metric.lower():
        return r'Energy (Wh)'
    elif "power" in metric.lower() and "max" in metric.lower():
        return r'Maximum Power (W)'
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
        nosec_re = re.compile(r"^udp_rasp_conv_stats_n25_nosec_scenario(?P<scen>[AC])\.csv$")
        m = nosec_re.match(base)
        if m:
            return None, None, 'nosec', m.group('scen')
    
    # Handle psk case - algorithm but no cert
    psk_re = re.compile(
        r"^udp_rasp_conv_stats_"
        r"(?P<alg>[A-Z0-9_]+)"
        r"_n25_psk_scenario(?P<scen>[AC])\.csv$"
    )
    m = psk_re.match(base)
    if m:
        return m.group('alg'), None, 'psk', m.group('scen')
    
    # Handle pki case - both algorithm and cert
    pki_re = re.compile(
        r"^udp_rasp_conv_stats_"
        r"(?P<alg>[A-Z0-9_]+)"
        r"_(?P<cert>RSA_2048|EC_P256|EC_ED25519|DILITHIUM_LEVEL[235]|FALCON_LEVEL[15])"
        r"_n25_pki_scenario(?P<scen>[AC])\.csv$"
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

def plot_tradeoff(all_data, metric_x, metric_y, scenario, include_nosec=False, normalize=False, log_scale=False):
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
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)
    plt.grid(True, ls='--', alpha=0.3, which='both')
    plt.tight_layout()
    
    # Adjust filename to indicate if nosec is excluded and if normalized
    suffix = ''
    if normalize:
        suffix += '_normalized'
    plt.savefig(f'tradeoff_{metric_x}_{metric_y}_{scenario}{suffix}.pdf', bbox_inches='tight')
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
    
    # Calculate incremental changes between networks
    values = []
    labels = []
    colors = []
    pct_labels = []
    
    # Add baseline (always 100%)
    values.append(1.0)  # Normalized baseline is always 1.0
    labels.append('Fiducial\n(Baseline)')
    colors.append('#4CAF50')  # Green for baseline
    pct_labels.append('100%')
    
    # Track the running total for incremental changes
    running_total = 1.0
    prev_network = 'fiducial'
    
    # Process each network after fiducial
    for _, row in df[df['network'] != 'fiducial'].iterrows():
        # Calculate the incremental change from previous network
        incremental_change = row['normalized'] - running_total
        
        # Add data for this network
        values.append(incremental_change)
        labels.append(row['network'].capitalize())
        
        # Determine color based on change direction
        if incremental_change < 0:
            colors.append('#4CAF50')  # Green for improvement
        else:
            colors.append('#F44336')  # Red for degradation
        
        # Format percentage change label
        network_to_fiducial_pct = f"{row['pct_change']:.1f}%"
        incremental_pct = f"{(incremental_change / running_total * 100):.1f}%"
        pct_labels.append(f"{incremental_pct}\n(Total: {network_to_fiducial_pct})")
        
        # Update running total
        running_total += incremental_change
        prev_network = row['network']
    
    # Add the final total bar
    values.append(running_total)
    labels.append('Total')
    colors.append('#2196F3')  # Blue for total
    pct_labels.append(f"{((running_total - 1) * 100):.1f}%")
    
    # Create the waterfall chart
    # First bar (baseline)
    ax.bar(0, values[0], color=colors[0], edgecolor='black')
    
    # Calculate positions for all bars except baseline and total
    bottoms = np.zeros(len(values) - 2)
    running_sum = values[0]
    
    for i in range(len(bottoms)):
        bottoms[i] = running_sum
        running_sum += values[i + 1]
    
    # Intermediate bars (incremental changes)
    for i in range(len(bottoms)):
        ax.bar(i + 1, values[i + 1], bottom=bottoms[i], color=colors[i + 1], edgecolor='black')
    
    # Final bar (total)
    ax.bar(len(values) - 1, values[-1], color=colors[-1], edgecolor='black')
    
    # Add connecting lines
    for i in range(len(values) - 2):
        start_x = i
        end_x = i + 1
        start_y = bottoms[i] if i > 0 else values[0]
        end_y = bottoms[i]
        
        ax.plot([start_x, end_x], [start_y, end_y], 'k--', alpha=0.5)
    
    # Add percentage labels
    for i, (v, pct) in enumerate(zip(values, pct_labels)):
        if i == 0:  # Baseline
            ax.text(i, v + 0.05, pct, ha='center', va='bottom', fontweight='bold')
        elif i == len(values) - 1:  # Total
            ax.text(i, v + 0.05, pct, ha='center', va='bottom', fontweight='bold')
        else:  # Incremental bars
            # Position based on whether change is positive or negative
            if v < 0:
                ax.text(i, bottoms[i-1] + v - 0.05, pct, ha='center', va='top', fontweight='bold')
            else:
                ax.text(i, bottoms[i-1] + v + 0.05, pct, ha='center', va='bottom', fontweight='bold')
    
    # Set axis labels and title
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f"Normalized {format_labels(metric)} (Fiducial = 1.0)")
    
    # Construct title based on configuration
    if security_mode == 'pki':
        title = f"Network Impact on {format_labels(metric)}\n{format_labels(algorithm)} with {format_labels(cert_type)} (Scenario {scenario})"
    elif security_mode == 'psk':
        title = f"Network Impact on {format_labels(metric)}\n{format_labels(algorithm)} with PSK (Scenario {scenario})"
    else:  # nosec
        title = f"Network Impact on {format_labels(metric)}\nNo Security (Scenario {scenario})"
    
    ax.set_title(title)
    
    # Set y-axis limits with padding
    max_value = max(running_total, 1.0)
    y_padding = max_value * 0.2
    ax.set_ylim(0, max_value + y_padding)
    
    # Add grid for readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=1.0 (baseline reference)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

def create_algorithm_waterfall(all_data, ax, metric, scenario, security_mode='pki',
                           cert_type=None, network='fiducial'):
    """
    Helper function to create normalized algorithm scaling waterfall plot
    
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
    
    # Calculate incremental changes between algorithm levels
    values = []
    labels = []
    colors = []
    pct_labels = []
    
    # Add baseline (always 100%)
    values.append(1.0)  # Normalized baseline is always 1.0
    labels.append(f'{base_alg}\n(Baseline)')
    colors.append('#4CAF50')  # Green for baseline
    pct_labels.append('100%')
    
    # Track the running total for incremental changes
    running_total = 1.0
    prev_alg = base_alg
    
    # Process each algorithm level after baseline
    for _, row in df[df['algorithm'] != base_alg].iterrows():
        # Calculate the incremental change from previous algorithm
        incremental_change = row['normalized'] - running_total
        
        # Add data for this algorithm
        values.append(incremental_change)
        labels.append(row['algorithm'])
        
        # Determine color based on change direction
        if incremental_change < 0:
            colors.append('#4CAF50')  # Green for improvement
        else:
            colors.append('#F44336')  # Red for degradation
        
        # Format percentage change label
        alg_to_baseline_pct = f"{row['pct_change']:.1f}%"
        incremental_pct = f"{(incremental_change / running_total * 100):.1f}%"
        pct_labels.append(f"{incremental_pct}\n(Total: {alg_to_baseline_pct})")
        
        # Update running total
        running_total += incremental_change
        prev_alg = row['algorithm']
    
    # Add the final total bar
    values.append(running_total)
    labels.append('Total')
    colors.append('#2196F3')  # Blue for total
    pct_labels.append(f"{((running_total - 1) * 100):.1f}%")
    
    # Create the waterfall chart
    # First bar (baseline)
    ax.bar(0, values[0], color=colors[0], edgecolor='black')
    
    # Calculate positions for all bars except baseline and total
    bottoms = np.zeros(len(values) - 2)
    running_sum = values[0]
    
    for i in range(len(bottoms)):
        bottoms[i] = running_sum
        running_sum += values[i + 1]
    
    # Intermediate bars (incremental changes)
    for i in range(len(bottoms)):
        ax.bar(i + 1, values[i + 1], bottom=bottoms[i], color=colors[i + 1], edgecolor='black')
    
    # Final bar (total)
    ax.bar(len(values) - 1, values[-1], color=colors[-1], edgecolor='black')
    
    # Add connecting lines
    for i in range(len(values) - 2):
        start_x = i
        end_x = i + 1
        start_y = bottoms[i] if i > 0 else values[0]
        end_y = bottoms[i]
        
        ax.plot([start_x, end_x], [start_y, end_y], 'k--', alpha=0.5)
    
    # Add percentage labels
    for i, (v, pct) in enumerate(zip(values, pct_labels)):
        if i == 0:  # Baseline
            ax.text(i, v + 0.05, pct, ha='center', va='bottom', fontweight='bold')
        elif i == len(values) - 1:  # Total
            ax.text(i, v + 0.05, pct, ha='center', va='bottom', fontweight='bold')
        else:  # Incremental bars
            # Position based on whether change is positive or negative
            if v < 0:
                ax.text(i, bottoms[i-1] + v - 0.05, pct, ha='center', va='top', fontweight='bold')
            else:
                ax.text(i, bottoms[i-1] + v + 0.05, pct, ha='center', va='bottom', fontweight='bold')
    
    # Set axis labels and title
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f"Normalized {format_labels(metric)} (KYBER_LEVEL1 = 1.0)")
    
    # Construct title based on configuration
    if security_mode == 'pki':
        title = f"Algorithm Scaling Impact on {format_labels(metric)}\n{format_labels(cert_type)} Certificate (Scenario {scenario}, {network.capitalize()})"
    else:  # psk
        title = f"Algorithm Scaling Impact on {format_labels(metric)}\nPSK (Scenario {scenario}, {network.capitalize()})"
    
    ax.set_title(title)
    
    # Set y-axis limits with padding
    max_value = max(running_total, 1.0)
    y_padding = max_value * 0.2
    ax.set_ylim(0, max_value + y_padding)
    
    # Add grid for readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=1.0 (baseline reference)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
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
                        label_text = f"+{pct_change:.1f}%"
                    else:
                        label_text = f"{pct_change:.1f}%"
                    
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
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)
    
    # Set axis labels and title
    ax.set_ylabel(f"Change from Fiducial Baseline (%)")
    
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
        plot_tradeoff(all_data, 'cpu_cycles', 'Energy (Wh)', scen, normalize=True)
        plot_tradeoff(all_data, 'total_bytes', 'Energy (Wh)', scen, normalize=True)
        
        create_waterfall_plot(all_data, 'Energy (Wh)', scen, plot_type='network')
        create_waterfall_plot(all_data, 'Energy (Wh)', scen, plot_type='algorithm')

        # Generate network comparison plots for different security modes
        create_network_comparison_plot(all_data, 'Energy (Wh)', scen, 'pki')
        create_network_comparison_plot(all_data, 'Energy (Wh)', scen, 'psk')
    
    print("All plots generated successfully!")