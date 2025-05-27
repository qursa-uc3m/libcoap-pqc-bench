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
    """
    Parse filename to extract configuration details, now supporting filtered files.
    
    Args:
        fpath (str): File path to parse
        
    Returns:
        tuple: (algorithm, certificate, mode, scenario, is_filtered)
    """
    base = os.path.basename(fpath)
    
    # Check if this is a filtered file
    is_filtered = '_filtered' in base
    
    # Remove the filtered suffix for parsing
    if is_filtered:
        base = base.replace('_filtered', '')
    
    # Handle nosec case - no algorithm or cert
    if 'nosec' in base:
        nosec_re = re.compile(r"^udp_rasp_conv_stats_n25(_parallel)?_nosec_scenario(?P<scen>[AC])\.csv$")
        m = nosec_re.match(base)
        if m:
            return None, None, 'nosec', m.group('scen'), is_filtered
    
    # Handle psk case - algorithm but no cert
    psk_re = re.compile(
        r"^udp_rasp_conv_stats_"
        r"(?P<alg>[A-Z0-9_]+)"
        r"_n25(_parallel)?_psk_scenario(?P<scen>[AC])\.csv$"
    )
    m = psk_re.match(base)
    if m:
        return m.group('alg'), None, 'psk', m.group('scen'), is_filtered
    
    # Handle pki case - both algorithm and cert
    pki_re = re.compile(
        r"^udp_rasp_conv_stats_"
        r"(?P<alg>[A-Z0-9_]+)"
        r"_(?P<cert>RSA_2048|EC_P256|EC_ED25519|DILITHIUM_LEVEL[235]|FALCON_LEVEL[15])"
        r"_n25(_parallel)?_pki_scenario(?P<scen>[AC])\.csv$"
    )
    m = pki_re.match(base)
    if m:
        return m.group('alg'), m.group('cert'), 'pki', m.group('scen'), is_filtered
    
    return None, None, None, None, is_filtered

# 3. DATA PARSING AND LOADING
def parse_benchmark_file(file_path):
    """
    Parse a benchmark file with iteration blocks and global summary.
    Enhanced to handle different sample sizes in filtered files.
    """
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
    
    # Calculate actual sample size from iterations
    actual_sample_size = len(iterations)
    
    return {
        'iterations': iterations,
        'global_mean': global_mean,
        'global_std': global_std,
        'global_mode': global_mode,
        'global_below': global_below,
        'global_above': global_above,
        'global_min': global_min,
        'global_max': global_max,
        'global_range': global_range,
        'actual_sample_size': actual_sample_size  # NEW: Track actual sample size
    }

def load_all(use_filtered=True):
    """
    Load all data using pre-calculated summary statistics.
    
    Args:
        use_filtered (bool): Whether to use filtered files when available.
                            Falls back to original files when filtered don't exist (e.g., fiducial network).
        
    Returns:
        list: All data records with sample size information
    """
    all_data = []
    file_stats = {'original': 0, 'filtered': 0, 'missing': 0}
    
    for net in NETWORKS:
        data_dir = os.path.join(ROOT_DIR, f'bench-data-{net}')
        if not os.path.isdir(data_dir):
            print(f"Warning: '{data_dir}' not found, skipping.")
            continue
            
        # Get all CSV files in the directory (both original and filtered)
        all_files = glob.glob(os.path.join(data_dir, 'udp_rasp_conv_stats_*.csv'))
        
        # Group files by their base configuration to avoid duplicates
        file_configs = {}
        for f in all_files:
            alg, cert, mode, scen, is_filtered = parse_filename(f)
            if mode not in security_modes or scen not in SCENARIOS:
                continue
            
            # Create a config key without the filtered flag
            config_key = (alg, cert, mode, scen, net)
            
            if config_key not in file_configs:
                file_configs[config_key] = {'original': None, 'filtered': None}
            
            if is_filtered:
                file_configs[config_key]['filtered'] = f
            else:
                file_configs[config_key]['original'] = f
        
        # Select the appropriate file for each configuration
        for config_key, files in file_configs.items():
            alg, cert, mode, scen, net = config_key
            
            # Choose file based on use_filtered preference
            selected_file = None
            is_using_filtered = False
            
            if use_filtered and files['filtered'] is not None:
                selected_file = files['filtered']
                is_using_filtered = True
                file_stats['filtered'] += 1
            elif files['original'] is not None:
                selected_file = files['original']
                is_using_filtered = False
                file_stats['original'] += 1
            else:
                print(f"Warning: No suitable file found for {config_key}")
                file_stats['missing'] += 1
                continue
            
            # Parse the selected file
            try:
                parsed_data = parse_benchmark_file(selected_file)
            except Exception as e:
                print(f"Error parsing {selected_file}: {e}")
                file_stats['missing'] += 1
                continue
            
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
                            'file': selected_file,
                            'is_filtered': is_using_filtered,
                            'actual_sample_size': parsed_data['actual_sample_size']
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
                            'file': selected_file,
                            'is_filtered': is_using_filtered,
                            'actual_sample_size': parsed_data['actual_sample_size']
                        }
                        all_data.append(record)
                    
            for item in all_data:
                if 'metric' in item:
                    item['metric'] = pd.Categorical([item['metric']], 
                                                categories=METRIC_CTS + METRIC_DSC, 
                                                ordered=True)[0]
    
    # Print file statistics
    print(f"File Statistics:")
    print(f"  Original files used: {file_stats['original']}")
    print(f"  Filtered files used: {file_stats['filtered']}")
    print(f"  Files with errors: {file_stats['missing']}")
    
    # Check for sample size consistency
    sample_sizes_by_network = {}
    for item in all_data:
        net = item['network']
        size = item['actual_sample_size']
        if net not in sample_sizes_by_network:
            sample_sizes_by_network[net] = set()
        sample_sizes_by_network[net].add(size)
    
    # Report sample size summary
    print(f"Sample Size Summary by Network:")
    for net, sizes in sample_sizes_by_network.items():
        if len(sizes) == 1:
            print(f"  {net}: {list(sizes)[0]} samples (consistent)")
        else:
            print(f"  {net}: {sorted(sizes)} samples (variable - expected for filtered data)")
    
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
   Plot mean metric_y vs metric_x per configuration, colored by network for a given scenario.
   Handles both original and filtered data automatically.
   """
   
   # Detect if filtered data is being used
   use_filtered = any(item.get('is_filtered', False) for item in all_data)
   
   print(f"Creating tradeoff plot using {'filtered' if use_filtered else 'original'} data")
   
   plt.figure(figsize=(12, 10))
   ax = plt.gca()
   
   # Convert data to DataFrame for easier filtering
   records = []
   sample_sizes = {}  # Track sample sizes
   
   for item in all_data:
       if 'mean' in item:
           # Filter out nosec unless explicitly included
           if item['mode'] == 'nosec' and not include_nosec:
               continue
           
           # Track sample sizes
           key = (item['network'], item['mode'], item['alg'], item['cert'])
           sample_sizes[key] = item.get('actual_sample_size', 15)
           
           records.append({
               'alg': item['alg'] or 'none',
               'cert': item['cert'] or 'none',
               'mode': item['mode'],
               'scenario': item['scenario'],
               'network': item['network'],
               'metric': item['metric'],
               'mean': item['mean']
           })
   
   # Report sample size information
   unique_sizes = set(sample_sizes.values())
   if len(unique_sizes) > 1:
       print(f"Sample size variation in tradeoff plot: {sorted(unique_sizes)}")
   
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
   if use_filtered:
       title += " - Filtered Data"
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
   
   # Adjust filename to indicate if nosec is excluded and if normalized/filtered
   suffix = ''
   if normalize:
       suffix += '_normalized'
   if use_filtered:
       suffix += '_filtered'
   
   # Ensure output directory exists
   os.makedirs(output_dir, exist_ok=True)
   plt.savefig(f'{output_dir}/tradeoff_{metric_x}_{metric_y}_{scenario}{suffix}.pdf', bbox_inches='tight')
   plt.show()
    
def create_spider_plot(all_data, metrics, scenario, security_modes=['pki', 'psk', 'nosec'], 
                     algorithms=None, certificates=None, include_legend=True,
                     normalization='min_max', baseline_network='fiducial', 
                     apply_metric_inversion=True, max_security_configs=9,
                     output_dir='./bench-plots-compare'):
   """
   Create a spider (radar) plot comparing metrics across different networks.
   Handles both original and filtered data automatically.
   """
   try:
       # Detect if filtered data is being used
       use_filtered = any(item.get('is_filtered', False) for item in all_data)
       
       print(f"Creating spider plot using {'filtered' if use_filtered else 'original'} data")
       
       # Extract networks
       all_networks = set()
       sample_sizes = {}  # Track sample sizes
       
       for item in all_data:
           all_networks.add(item['network'])
           # Track sample sizes
           key = (item['network'], item['mode'], item['alg'], item['cert'], item['scenario'])
           sample_sizes[key] = item.get('actual_sample_size', 15)
       
       networks = sorted(list(n for n in all_networks if n != baseline_network))
       
       if not networks:
           print(f"Warning: No non-baseline networks found. Cannot create spider plot.")
           return None
       
       # Report sample size information
       unique_sizes = set(sample_sizes.values())
       if len(unique_sizes) > 1:
           print(f"Sample size variation in spider plot data: {sorted(unique_sizes)}")
           size_distribution = {}
           for size in sample_sizes.values():
               size_distribution[size] = size_distribution.get(size, 0) + 1
           print(f"Sample size counts: {size_distribution}")
       
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
       
       # Group data by security configuration
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
       
       # Normalize the data
       norm_data = {}
       
       # For each metric, collect all available values first
       metric_values = {}
       for metric in metrics_to_use:
           metric_values[metric] = []
           
           for config_key, metrics_dict in raw_data.items():
               if metric in metrics_dict:
                   if pd.notna(metrics_dict[metric]):  # Skip NaN values
                       metric_values[metric].append(metrics_dict[metric])
       
       # Apply normalization - COMPLETE VERSION WITH ALL METHODS
       if normalization == 'min_max':
           for metric in metrics_to_use:
               norm_data[metric] = {}
               
               if not metric_values[metric]:
                   continue
                   
               min_val = min(metric_values[metric])
               max_val = max(metric_values[metric])
               range_val = max_val - min_val
               
               if range_val <= 0:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
                   continue
               
               for config_key, metrics_dict in raw_data.items():
                   if metric in metrics_dict:
                       norm_val = (metrics_dict[metric] - min_val) / range_val
                       norm_val = max(0.0, min(1.0, norm_val))
                       norm_data[metric][config_key] = norm_val

       elif normalization == 'max_percent':
           for metric in metrics_to_use:
               norm_data[metric] = {}
               
               if not metric_values[metric]:
                   continue
                   
               max_val = max(metric_values[metric])
               
               if max_val <= 0:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
                   continue
               
               for config_key, metrics_dict in raw_data.items():
                   if metric in metrics_dict:
                       norm_val = metrics_dict[metric] / max_val
                       norm_val = max(0.0, min(1.0, norm_val))
                       norm_data[metric][config_key] = norm_val

       elif normalization == 'z_score':
           for metric in metrics_to_use:
               norm_data[metric] = {}
               
               if not metric_values[metric] or len(metric_values[metric]) < 2:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
                   continue
                   
               mean_val = np.mean(metric_values[metric])
               std_val = np.std(metric_values[metric])
               
               if std_val <= 0:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
                   continue
               
               for config_key, metrics_dict in raw_data.items():
                   if metric in metrics_dict:
                       z = (metrics_dict[metric] - mean_val) / std_val
                       norm_val = (z + 3) / 6
                       norm_val = max(0.0, min(1.0, norm_val))
                       norm_data[metric][config_key] = norm_val

       elif normalization == 'log':
           for metric in metrics_to_use:
               norm_data[metric] = {}
               
               if not metric_values[metric]:
                   continue
                   
               min_val = min(metric_values[metric])
               max_val = max(metric_values[metric])
               
               offset = 0
               if min_val <= 0:
                   offset = abs(min_val) + 1e-6
               
               if max_val + offset == min_val + offset:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
                   continue
               
               log_min = np.log(min_val + offset)
               log_max = np.log(max_val + offset)
               log_range = log_max - log_min
               
               if log_range <= 0:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
                   continue
               
               for config_key, metrics_dict in raw_data.items():
                   if metric in metrics_dict:
                       log_val = np.log(metrics_dict[metric] + offset)
                       norm_val = (log_val - log_min) / log_range
                       norm_val = max(0.0, min(1.0, norm_val))
                       norm_data[metric][config_key] = norm_val

       elif normalization == 'baseline_relative':
           for metric in metrics_to_use:
               norm_data[metric] = {}
               
               if not metric_values[metric]:
                   continue
               
               for sec_key in valid_configs:
                   baseline_key = sec_key + (baseline_network,)
                   
                   if baseline_key not in raw_data or metric not in raw_data[baseline_key]:
                       continue
                       
                   baseline_val = raw_data[baseline_key][metric]
                   
                   if baseline_val == 0:
                       continue
                   
                   norm_data[metric][baseline_key] = 1.0
                   
                   for network in networks:
                       config_key = sec_key + (network,)
                       
                       if config_key not in raw_data or metric not in raw_data[config_key]:
                           continue
                       
                       val = raw_data[config_key][metric]
                       ratio = val / baseline_val
                       
                       if ratio < 1:
                           norm_data[metric][config_key] = ratio
                       elif ratio > 1:
                           norm_data[metric][config_key] = min(ratio, 1.1)
                       else:
                           norm_data[metric][config_key] = 1.0

       else:
           for metric in metrics_to_use:
               norm_data[metric] = {}
               
               if not metric_values[metric]:
                   continue
                   
               max_val = max(metric_values[metric])
               
               if max_val <= 0:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = 0.5
               else:
                   for config_key, metrics_dict in raw_data.items():
                       if metric in metrics_dict:
                           norm_data[metric][config_key] = metrics_dict[metric] / max_val
       
       # Setup the plot
       fig = plt.figure(figsize=(14, 10))
       
       # Number of metrics
       N = len(metrics_to_use)
       
       # Calculate angles for each metric (evenly spaced)
       theta = np.linspace(0, 2*np.pi, N, endpoint=False)
       
       # Create subplot using polar projection
       ax = fig.add_subplot(111, projection='polar')
       
       # Set direction to clockwise
       ax.set_theta_direction(-1)
       
       # Start at 90 degrees (top)
       ax.set_theta_offset(np.pi/2)
       
       # Draw axis lines for each variable
       for i in range(N):
           ax.plot([theta[i], theta[i]], [0, 1.1], 'k-', linewidth=0.5)
       
       # Add rings at different levels
       for level in [0.25, 0.5, 0.75]:
           ax.plot(np.linspace(0, 2*np.pi, 100), [level]*100, 'k--', alpha=0.3)
       
       # Set limits
       ax.set_ylim(0, 1)
       
       # Set ticks and labels for each variable
       ax.set_xticks(theta)
       ax.set_xticklabels([format_labels(m) for m in metrics_to_use])
       # Move all labels outward
       ax.tick_params(axis='x', pad=30)
               
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
       
       # Plot networks (including baseline for comparison)
       plot_networks = networks + [baseline_network]
       
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
       security_str = "+".join(security_modes)
       title = f"Impact of Network Conditions on {security_str} - Scenario {scenario}"
       if use_filtered:
           title += " (Filtered Data)"
       ax.set_title(title, pad=25)
       
       # Add legend with better placement
       if include_legend:
           fig.subplots_adjust(right=0.7)  # Make room for legend
           ax.legend(legend_handles, legend_labels, loc='best',
                    bbox_to_anchor=(1.1, 0.5))
       
       # Generate output filename
       os.makedirs(output_dir, exist_ok=True)
       
       # Create metrics string for filename
       metrics_str = "_".join(m.replace(" ", "").replace("(", "").replace(")", "") for m in metrics_to_use)
       if len(metrics_str) > 50:
           metrics_str = f"{len(metrics_to_use)}_metrics"
           
       security_str = "_".join(security_modes)
       filtered_suffix = '_filtered' if use_filtered else ''
       
       filename = f"{output_dir}/spider_{scenario}_{security_str}_{normalization}_{metrics_str}{filtered_suffix}.pdf"
       
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
   

    
def calculate_network_difference_statistics(comparison_data, baseline_data, config_key, comp_network, baseline_network):
    """
    Calculate statistical difference between networks with enhanced sample size handling.
    
    Args:
        comparison_data (dict): Data from comparison network 
        baseline_data (dict): Data from baseline network
        config_key (tuple): Configuration identifier
        comp_network (str): Name of comparison network
        baseline_network (str): Name of baseline network
        
    Returns:
        dict: Statistical difference analysis with sample size handling
    """
    
    try:
        alg, cert, mode = config_key
        
        # Extract values with sample size information (backward compatible)
        mean_comp = float(comparison_data['mean'])
        std_comp = float(comparison_data['std'])
        n_comp = comparison_data.get('actual_sample_size', 15)  # Default to 15 for backward compatibility
        
        mean_base = float(baseline_data['mean'])
        std_base = float(baseline_data['std'])
        n_base = baseline_data.get('actual_sample_size', 15)  # Default to 15 for backward compatibility
        
        # Calculate difference
        mean_diff = mean_comp - mean_base
        
        # Enhanced error propagation considering different sample sizes
        se_comp = std_comp / np.sqrt(n_comp) if n_comp > 0 else 0
        se_base = std_base / np.sqrt(n_base) if n_base > 0 else 0
        se_diff = np.sqrt(se_comp**2 + se_base**2)
        
        # Use Welch's t-test for unequal sample sizes (backward compatible)
        if se_diff > 0:
            t_stat = mean_diff / se_diff
            
            # Welch's degrees of freedom approximation
            if n_comp > 1 and n_base > 1:
                var_comp = std_comp**2
                var_base = std_base**2
                
                numerator = (var_comp/n_comp + var_base/n_base)**2
                denominator = (var_comp/n_comp)**2/(n_comp-1) + (var_base/n_base)**2/(n_base-1)
                df = numerator / denominator if denominator > 0 else min(n_comp, n_base) - 1
            else:
                df = min(n_comp, n_base) - 1
                
            df = max(1, df)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            t_stat = 0
            df = min(n_comp, n_base) - 1
            p_value = 1.0
        
        # 95% confidence interval
        if df > 0:
            t_critical = stats.t.ppf(0.975, df)
            margin_error = t_critical * se_diff
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
        else:
            ci_lower = mean_diff
            ci_upper = mean_diff
        
        # Effect size (Cohen's d)
        if n_comp > 1 and n_base > 1:
            pooled_var = ((n_comp-1)*std_comp**2 + (n_base-1)*std_base**2) / (n_comp + n_base - 2)
            pooled_std = np.sqrt(pooled_var)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        else:
            pooled_std = np.sqrt((std_comp**2 + std_base**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Create readable configuration label
        if mode == 'nosec':
            config_label = 'NoSec'
        elif mode == 'psk':
            config_label = f'{alg} (PSK)' if alg else 'PSK'
        elif mode == 'pki':
            if alg and cert:
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
            'std_baseline': std_base,
            'n_comparison': n_comp,
            'n_baseline': n_base,
            'sample_size_warning': n_comp != n_base  # NEW: Flag for different sample sizes
        }
        
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error calculating difference statistics for {config_key}: {e}")
        return None
    
def create_network_difference_plot(all_data, metric, scenario, 
                                baseline_network='fiducial',
                                comparison_networks=None,
                                security_modes=['pki', 'psk', 'nosec'],
                                algorithms=None, certificates=None,
                                alpha=0.05, output_dir='./bench-plots-compare'):
    """
    Create network difference plot that handles both original and filtered data.
    
    Args:
        all_data (list): The benchmark data loaded from load_all()
        metric (str): Metric to analyze
        scenario (str): Scenario identifier 
        baseline_network (str): Network to use as baseline
        comparison_networks (list): Networks to compare against baseline
        security_modes (list): Security modes to include
        algorithms (list): Algorithms to include
        certificates (list): Certificate types to include
        alpha (float): Significance level
        output_dir (str): Directory to save plots
        
    Returns:
        dict: Results including difference statistics and file paths
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
    
    # Detect if filtered data is being used
    use_filtered = any(item.get('is_filtered', False) for item in all_data)
    
    print(f"Creating network difference plot for {metric} (Scenario {scenario})")
    print(f"Baseline: {baseline_network}")
    print(f"Comparisons: {comparison_networks}")
    if use_filtered:
        print("Detected filtered data in dataset")
    
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
        
        # Store data by network with sample size info
        data_by_config[config_key][item['network']] = {
            'mean': item['mean'],
            'std': item['std'],
            'actual_sample_size': item.get('actual_sample_size', 15),  # Backward compatible
            'is_filtered': item.get('is_filtered', False)              # Backward compatible
        }
    
    # Calculate differences for each comparison network
    all_differences = []
    sample_size_warnings = []
    
    for comp_network in comparison_networks:
        print(f"\nProcessing differences: {comp_network} - {baseline_network}")
        
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
            
            # Check if we have data for both networks
            if baseline_network not in network_data or comp_network not in network_data:
                continue
            
            baseline_data = network_data[baseline_network]
            comparison_data = network_data[comp_network]
            
            # Calculate difference with enhanced error propagation
            diff_stats = calculate_network_difference_statistics(
                comparison_data, baseline_data, config_key, comp_network, baseline_network
            )
            
            if diff_stats:
                # Add the missing fields for sorting
                diff_stats['alg'] = alg
                diff_stats['cert'] = cert
                all_differences.append(diff_stats)
                if diff_stats.get('sample_size_warning', False):
                    sample_size_warnings.append({
                        'config': diff_stats['config_label'],
                        'baseline_n': diff_stats.get('n_baseline', 15),
                        'comparison_n': diff_stats.get('n_comparison', 15)
                    })
    
    if not all_differences:
        print("No valid differences found for plotting")
        return None
    
    # Report sample size warnings if any
    if sample_size_warnings:
        print(f"\nSample Size Warnings ({len(sample_size_warnings)} configurations):")
        for warning in sample_size_warnings[:5]:  # Show first 5
            print(f"  {warning['config']}: baseline={warning['baseline_n']}, comparison={warning['comparison_n']}")
        if len(sample_size_warnings) > 5:
            print(f"  ... and {len(sample_size_warnings) - 5} more")
    
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
    sample_warnings = [d.get('sample_size_warning', False) for d in all_differences]
    
    # Color based on statistical significance and sample size warnings
    colors = []
    for diff, p_val, has_warning in zip(differences, p_values, sample_warnings):
        if p_val < alpha:
            base_color = '#d32f2f' if diff > 0 else '#1976d2'
            colors.append(base_color)
        else:
            colors.append('#757575')
    
    # Create bar plot
    bars = ax.bar(positions, differences, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add different edge styles for sample size warnings
    for i, (bar, has_warning) in enumerate(zip(bars, sample_warnings)):
        if has_warning:
            bar.set_linestyle('--')
            bar.set_linewidth(2)
            bar.set_alpha(0.6)
    
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
            y_pos = ci_u + y_range * 0.02
            ax.text(pos, y_pos, significance, ha='center', va='bottom', 
                   fontweight='bold', color='red', fontsize=8)
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'Network Impact on {format_labels(metric)} (Comparison - Baseline)')
    ax.set_xlabel('Security Configuration')
    
    # Create title
    comp_networks_str = ' vs '.join(comparison_networks)
    title = f'Network Impact: {comp_networks_str} vs {baseline_network}\n'
    title += f'{format_labels(metric)} (Scenario {scenario})'
    if use_filtered:
        title += ' - Using Filtered Data'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Enhanced legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#d32f2f', alpha=0.8, label=f'Significant increase (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#1976d2', alpha=0.8, label=f'Significant decrease (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#757575', alpha=0.8, label=f'No significant difference (p $\\geq$ {alpha})'),
    ]
    
    if sample_size_warnings:
        legend_elements.append(
            plt.Rectangle((0,0),1,1, color='gray', alpha=0.6, linestyle='--', 
                         label='Different sample sizes')
        )
    
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
    filtered_suffix = '_filtered' if use_filtered else ''
    filename = f"{output_dir}/network_difference_{clean_metric}_{scenario}_{comp_str}_vs_{baseline_network}{filtered_suffix}.pdf"
    
    # Save plot
    plt.savefig(filename, bbox_inches='tight')
    print(f"Network difference plot saved to: {filename}")
    
    # Generate statistical report
    report = generate_network_difference_report(
        all_differences, metric, scenario, baseline_network, comparison_networks, alpha, use_filtered
    )
    
    # Save report
    report_file = filename.replace('.pdf', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Statistical report saved to: {report_file}")
    
    # Show plot
    plt.show()
    
    # Return results with enhanced information
    return {
        'differences': all_differences,
        'plot_file': filename,
        'report_file': report_file,
        'significant_differences': [d for d in all_differences if d['p_value'] < alpha],
        'total_comparisons': len(all_differences),
        'sample_size_warnings': sample_size_warnings,
        'using_filtered_data': use_filtered,
        'alpha': alpha
    }

def generate_network_difference_report(differences, metric, scenario, baseline_network, 
                                     comparison_networks, alpha, use_filtered):
    """Generate comprehensive text report for network difference analysis."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""NETWORK DIFFERENCE ANALYSIS REPORT
{'=' * 70}
Metric: {format_labels(metric)}
Scenario: {scenario}
Baseline Network: {baseline_network}
Comparison Networks: {', '.join(comparison_networks)}
Data Type: {'Filtered (outliers removed)' if use_filtered else 'Original'}
Significance Level: α = {alpha}
Generated: {timestamp}

SUMMARY STATISTICS
{'-' * 40}
"""
    
    significant_diffs = [d for d in differences if d['p_value'] < alpha]
    positive_diffs = [d for d in significant_diffs if d['mean_diff'] > 0]
    negative_diffs = [d for d in significant_diffs if d['mean_diff'] < 0]
    sample_warnings = [d for d in differences if d.get('sample_size_warning', False)]
    
    report += f"""Total comparisons: {len(differences)}
Significant differences: {len(significant_diffs)} ({len(significant_diffs)/len(differences)*100:.1f}%)
  - Network penalties (positive): {len(positive_diffs)}
  - Network benefits (negative): {len(negative_diffs)}
Non-significant differences: {len(differences) - len(significant_diffs)}
Configurations with different sample sizes: {len(sample_warnings)}

"""
    
    # Sample size analysis
    if sample_warnings:
        report += f"""SAMPLE SIZE ANALYSIS
{'-' * 40}
The following configurations have different sample sizes between baseline and comparison networks.
This typically occurs when filtered data is used and different networks had different amounts of outliers.

"""
        sample_size_summary = {}
        for diff in differences:
            n_base = diff.get('n_baseline', 15)
            n_comp = diff.get('n_comparison', 15)
            key = f"{n_base}-{n_comp}"
            if key not in sample_size_summary:
                sample_size_summary[key] = 0
            sample_size_summary[key] += 1
        
        report += "Sample size combinations (baseline-comparison): count\n"
        for combo, count in sorted(sample_size_summary.items()):
            report += f"  {combo}: {count} configurations\n"
        report += "\n"
    
    if positive_diffs:
        report += f"""SIGNIFICANT NETWORK PENALTIES (p < {alpha})
{'-' * 60}
These configurations show significantly higher resource consumption
under network stress compared to baseline conditions.

"""
        
        for diff in sorted(positive_diffs, key=lambda x: x['mean_diff'], reverse=True):
            effect_size = interpret_cohens_d(diff['cohens_d'])
            sample_info = ""
            if diff.get('sample_size_warning', False):
                n_base = diff.get('n_baseline', 15)
                n_comp = diff.get('n_comparison', 15)
                sample_info = f" (n_base={n_base}, n_comp={n_comp})"
            
            report += f"""
{diff['config_label']} ({diff['comparison_network']} vs {diff['baseline_network']}){sample_info}:
  Network penalty: {diff['mean_diff']:.4f} ± {diff.get('se_diff', 0):.4f}
  95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
  t-statistic: {diff['t_statistic']:.3f} (df = {diff.get('degrees_freedom', 0):.1f})
  p-value: {diff['p_value']:.6f}
  Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
  Baseline: {diff['mean_baseline']:.4f} ± {diff.get('std_baseline', 0):.4f}
  Comparison: {diff['mean_comparison']:.4f} ± {diff.get('std_comparison', 0):.4f}
"""
    
    if negative_diffs:
        report += f"""
SIGNIFICANT NETWORK BENEFITS (p < {alpha})
{'-' * 60}
These configurations show significantly lower resource consumption
under network stress (unexpected but possible).

"""
        
        for diff in sorted(negative_diffs, key=lambda x: x['mean_diff']):
            effect_size = interpret_cohens_d(diff['cohens_d'])
            sample_info = ""
            if diff.get('sample_size_warning', False):
                n_base = diff.get('n_baseline', 15)
                n_comp = diff.get('n_comparison', 15)
                sample_info = f" (n_base={n_base}, n_comp={n_comp})"
            
            report += f"""
{diff['config_label']} ({diff['comparison_network']} vs {diff['baseline_network']}){sample_info}:
  Network benefit: {diff['mean_diff']:.4f} ± {diff.get('se_diff', 0):.4f}
  95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
  t-statistic: {diff['t_statistic']:.3f} (df = {diff.get('degrees_freedom', 0):.1f})
  p-value: {diff['p_value']:.6f}
  Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
  Baseline: {diff['mean_baseline']:.4f} ± {diff.get('std_baseline', 0):.4f}
  Comparison: {diff['mean_comparison']:.4f} ± {diff.get('std_comparison', 0):.4f}
"""
    
    if not significant_diffs:
        report += f"""NO SIGNIFICANT NETWORK EFFECTS FOUND
{'-' * 60}
All configurations show no statistically significant difference
between baseline and network conditions.
"""
    
    report += f"""
METHODOLOGICAL NOTES
{'-' * 40}
• Statistical Test: Welch's t-test (handles unequal sample sizes and variances)
• Error Propagation: SE_diff = √(SE_baseline² + SE_comparison²)
• Degrees of Freedom: Welch's approximation for unequal samples
• Effect Size: Cohen's d using pooled standard deviation
• Data Type: {'Filtered dataset (outliers removed)' if use_filtered else 'Original dataset'}

RECOMMENDATIONS
{'-' * 40}
"""
    
    if use_filtered and sample_warnings:
        report += f"""• UNEQUAL SAMPLE SIZES: {len(sample_warnings)} configurations have different sample sizes
  - This is expected when using filtered data
  - Welch's t-test properly accounts for unequal sample sizes
  - Results remain statistically valid

"""
    
    if len(positive_diffs) > len(differences) * 0.5:
        report += "• HIGH NETWORK SENSITIVITY: Most configurations show significant network penalties\n"
    elif len(significant_diffs) > 0:
        report += "• MODERATE NETWORK SENSITIVITY: Some configurations affected by network conditions\n"
    else:
        report += "• LOW NETWORK SENSITIVITY: Configurations appear network-resilient\n"
    
    if use_filtered:
        report += f"""
FILTERED DATA CONSIDERATIONS
{'-' * 40}
• Outlier removal may have affected sample sizes differently across networks
• Baseline network (fiducial) typically has fewer outliers than stressed networks
• Statistical tests account for unequal sample sizes using Welch's method
• Results represent typical behavior after removing extreme outliers
"""
    
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
                                    networks=None,
                                    security_modes=['pki', 'psk'],
                                    certificates=None,
                                    alpha=0.05, output_dir='./bench-plots-compare'):
    """
    Create algorithm difference plot that handles both original and filtered data.
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
    
    # Detect if filtered data is being used
    use_filtered = any(item.get('is_filtered', False) for item in all_data)
    
    print(f"Creating algorithm difference plot for {metric} (Scenario {scenario})")
    if use_filtered:
        print("Detected filtered data in dataset")
    
    # Extract and organize data by (network, mode, cert) -> {algorithm: data}
    configurations = {}
    
    for item in all_data:
        if (item['scenario'] != scenario or 
            item['metric'] != metric or
            item['network'] not in networks or
            item['mode'] == 'nosec' or
            item['mode'] not in security_modes):
            continue
        
        if item['mode'] == 'pki' and certificates and item['cert'] not in certificates:
            continue
            
        if item['alg'] not in [baseline_algorithm] + comparison_algorithms:
            continue
        
        config_key = (item['network'], item['mode'], item['cert'])
        
        if config_key not in configurations:
            configurations[config_key] = {}
        
        configurations[config_key][item['alg']] = {
            'mean': item['mean'],
            'std': item['std'],
            'actual_sample_size': item.get('actual_sample_size', 15),
            'is_filtered': item.get('is_filtered', False)
        }
    
    # Calculate differences
    all_differences = []
    sample_size_warnings = []
    
    for config_key, algorithm_data in configurations.items():
        if baseline_algorithm not in algorithm_data:
            continue
        
        baseline_data = algorithm_data[baseline_algorithm]
        
        for comp_alg in comparison_algorithms:
            if comp_alg not in algorithm_data:
                continue
                
            comparison_data = algorithm_data[comp_alg]
            
            diff_stats = calculate_network_difference_statistics(
                comparison_data, baseline_data, config_key, comp_alg, baseline_algorithm
            )
            
            if diff_stats:
                # Add the missing fields for sorting
                diff_stats['network'] = config_key[0]     # network
                diff_stats['alg'] = comp_alg              # comparison algorithm 
                diff_stats['cert'] = config_key[2]       # cert
                diff_stats['comparison_algorithm'] = comp_alg
                all_differences.append(diff_stats)
                if diff_stats.get('sample_size_warning', False):
                    sample_size_warnings.append({
                        'config': diff_stats['config_label'],
                        'baseline_n': diff_stats.get('n_baseline', 15),
                        'comparison_n': diff_stats.get('n_comparison', 15)
                    })
    
    if not all_differences:
        print("No valid differences found for plotting")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort by network, then mode, then cert, then comparison algorithm
    all_differences.sort(key=lambda x: (
        str(x.get('network', '')), 
        str(x.get('mode', '')), 
        str(x.get('cert', '') or 'zzz'), 
        str(x.get('comparison_algorithm', ''))
    ))
    
    positions = np.arange(len(all_differences))
    differences = [d['mean_diff'] for d in all_differences]
    ci_lower = [d['ci_lower'] for d in all_differences]
    ci_upper = [d['ci_upper'] for d in all_differences]
    p_values = [d['p_value'] for d in all_differences]
    labels = [f"{d['config_label']} ({d['comparison_network']})" for d in all_differences]
    sample_warnings = [d.get('sample_size_warning', False) for d in all_differences]
    
    # Color and styling
    colors = []
    for diff, p_val in zip(differences, p_values):
        if p_val < alpha:
            colors.append('#d32f2f' if diff > 0 else '#1976d2')
        else:
            colors.append('#757575')
    
    bars = ax.bar(positions, differences, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Different styling for sample size warnings
    for i, (bar, has_warning) in enumerate(zip(bars, sample_warnings)):
        if has_warning:
            bar.set_linestyle('--')
            bar.set_linewidth(2)
            bar.set_alpha(0.6)
    
    # Add confidence intervals and formatting
    err_lower = np.array(differences) - np.array(ci_lower)
    err_upper = np.array(ci_upper) - np.array(differences)
    ax.errorbar(positions, differences, yerr=[err_lower, err_upper],
                fmt='none', color='black', capsize=3, capthick=1)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    # Significance indicators
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
    
    # Labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Algorithm Impact on {format_labels(metric)} (Comparison - Baseline)')
    ax.set_xlabel('Network + Security Configuration')
    
    comp_algorithms_str = ' vs '.join(comparison_algorithms)
    title = f'Algorithm Scaling: {comp_algorithms_str} vs {baseline_algorithm}\n'
    title += f'{format_labels(metric)} (Scenario {scenario})'
    if use_filtered:
        title += ' - Using Filtered Data'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#d32f2f', alpha=0.8, label=f'Significant increase (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#1976d2', alpha=0.8, label=f'Significant decrease (p $<$ {alpha})'),
        plt.Rectangle((0,0),1,1, color='#757575', alpha=0.8, label=f'No significant difference (p $\\geq$ {alpha})'),
    ]
    
    if sample_size_warnings:
        legend_elements.append(
            plt.Rectangle((0,0),1,1, color='gray', alpha=0.6, linestyle='--', 
                         label='Different sample sizes')
        )
    
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    comp_str = '_'.join(comparison_algorithms)
    networks_str = '_'.join(networks)
    filtered_suffix = '_filtered' if use_filtered else ''
    filename = f"{output_dir}/algorithm_difference_{clean_metric}_{scenario}_{comp_str}_vs_{baseline_algorithm}_{networks_str}{filtered_suffix}.pdf"
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Algorithm difference plot saved to: {filename}")
    
    plt.show()
    
    return {
        'differences': all_differences,
        'plot_file': filename,
        'significant_differences': [d for d in all_differences if d['p_value'] < alpha],
        'total_comparisons': len(all_differences),
        'sample_size_warnings': sample_size_warnings,
        'using_filtered_data': use_filtered,
        'alpha': alpha
    }


def calculate_algorithm_difference_statistics(comparison_data, baseline_data, config_key, comp_algorithm, baseline_algorithm):
   """
   Calculate statistical difference between algorithms with enhanced sample size handling.
   
   Parameters:
   -----------
   comparison_data : dict
       Data from comparison algorithm {'mean': float, 'std': float, 'actual_sample_size': int, ...}
   baseline_data : dict
       Data from baseline algorithm {'mean': float, 'std': float, 'actual_sample_size': int, ...}
   config_key : tuple
       (network, mode, certificate) configuration identifier
   comp_algorithm : str
       Name of comparison algorithm
   baseline_algorithm : str
       Name of baseline algorithm
   
   Returns:
   --------
   dict : Statistical difference analysis with sample size handling
   """
   
   try:
       network, mode, cert = config_key
       
       # Extract values with sample size information (backward compatible)
       mean_comp = float(comparison_data['mean'])
       std_comp = float(comparison_data['std'])
       n_comp = comparison_data.get('actual_sample_size', 15)  # Default to 15 for backward compatibility
       
       mean_base = float(baseline_data['mean'])
       std_base = float(baseline_data['std'])
       n_base = baseline_data.get('actual_sample_size', 15)  # Default to 15 for backward compatibility
       
       # Calculate difference
       mean_diff = mean_comp - mean_base
       
       # Enhanced error propagation considering different sample sizes
       se_comp = std_comp / np.sqrt(n_comp) if n_comp > 0 else 0
       se_base = std_base / np.sqrt(n_base) if n_base > 0 else 0
       se_diff = np.sqrt(se_comp**2 + se_base**2)
       
       # Use Welch's t-test for unequal sample sizes
       if se_diff > 0:
           t_stat = mean_diff / se_diff
           
           # Welch's degrees of freedom approximation
           if n_comp > 1 and n_base > 1:
               var_comp = std_comp**2
               var_base = std_base**2
               
               numerator = (var_comp/n_comp + var_base/n_base)**2
               denominator = (var_comp/n_comp)**2/(n_comp-1) + (var_base/n_base)**2/(n_base-1)
               df = numerator / denominator if denominator > 0 else min(n_comp, n_base) - 1
           else:
               df = min(n_comp, n_base) - 1
               
           df = max(1, df)
           p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
       else:
           t_stat = 0
           df = min(n_comp, n_base) - 1
           p_value = 1.0
       
       # 95% confidence interval
       if df > 0:
           t_critical = stats.t.ppf(0.975, df)
           margin_error = t_critical * se_diff
           ci_lower = mean_diff - margin_error
           ci_upper = mean_diff + margin_error
       else:
           ci_lower = mean_diff
           ci_upper = mean_diff
       
       # Effect size (Cohen's d)
       if n_comp > 1 and n_base > 1:
           pooled_var = ((n_comp-1)*std_comp**2 + (n_base-1)*std_base**2) / (n_comp + n_base - 2)
           pooled_std = np.sqrt(pooled_var)
           cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
       else:
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
           'comparison_network': comp_algorithm,  # Using same field name for compatibility
           'baseline_network': baseline_algorithm,  # Using same field name for compatibility
           'comparison_algorithm': comp_algorithm,
           'baseline_algorithm': baseline_algorithm,
           'mean_diff': mean_diff,
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
           'std_baseline': std_base,
           'n_comparison': n_comp,
           'n_baseline': n_base,
           'sample_size_warning': n_comp != n_base  # Flag for different sample sizes
       }
       
   except (ValueError, TypeError, KeyError) as e:
       print(f"Error calculating algorithm difference statistics for {config_key}: {e}")
       return None


def generate_algorithm_difference_report(differences, metric, scenario, baseline_algorithm, comparison_algorithms, alpha, use_filtered):
   """Generate comprehensive text report for algorithm difference analysis."""
   
   timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   
   report = f"""ALGORITHM DIFFERENCE ANALYSIS REPORT
{'=' * 60}
Metric: {format_labels(metric)}
Scenario: {scenario}
Baseline Algorithm: {baseline_algorithm}
Comparison Algorithms: {', '.join(comparison_algorithms)}
Data Type: {'Filtered (outliers removed)' if use_filtered else 'Original'}
Significance Level: α = {alpha}
Generated: {timestamp}

SUMMARY STATISTICS
{'-' * 30}
"""
   
   significant_diffs = [d for d in differences if d['p_value'] < alpha]
   positive_diffs = [d for d in significant_diffs if d['mean_diff'] > 0]
   negative_diffs = [d for d in significant_diffs if d['mean_diff'] < 0]
   sample_warnings = [d for d in differences if d.get('sample_size_warning', False)]
   
   report += f"""Total comparisons: {len(differences)}
Significant differences: {len(significant_diffs)} ({len(significant_diffs)/len(differences)*100:.1f}%)
 - Algorithm penalties (positive): {len(positive_diffs)}
 - Algorithm benefits (negative): {len(negative_diffs)}
Non-significant differences: {len(differences) - len(significant_diffs)}
Configurations with different sample sizes: {len(sample_warnings)}

"""
   
   # Sample size analysis
   if sample_warnings:
       report += f"""SAMPLE SIZE ANALYSIS
{'-' * 30}
Configurations with different sample sizes between baseline and comparison algorithms.

"""
       sample_size_summary = {}
       for diff in differences:
           n_base = diff.get('n_baseline', 15)
           n_comp = diff.get('n_comparison', 15)
           key = f"{n_base}-{n_comp}"
           if key not in sample_size_summary:
               sample_size_summary[key] = 0
           sample_size_summary[key] += 1
       
       report += "Sample size combinations (baseline-comparison): count\n"
       for combo, count in sorted(sample_size_summary.items()):
           report += f"  {combo}: {count} configurations\n"
       report += "\n"
   
   if positive_diffs:
       report += f"""SIGNIFICANT ALGORITHM PENALTIES (p < {alpha})
{'-' * 50}
These configurations show significantly higher resource consumption
with larger algorithms compared to {baseline_algorithm}.

"""
       
       for diff in sorted(positive_diffs, key=lambda x: x['mean_diff'], reverse=True):
           effect_size = interpret_cohens_d(diff['cohens_d'])
           sample_info = ""
           if diff.get('sample_size_warning', False):
               n_base = diff.get('n_baseline', 15)
               n_comp = diff.get('n_comparison', 15)
               sample_info = f" (n_base={n_base}, n_comp={n_comp})"
           
           report += f"""
{diff['config_label']} ({diff['comparison_algorithm']} vs {diff['baseline_algorithm']}){sample_info}:
 Algorithm penalty: {diff['mean_diff']:.4f} ± {diff.get('se_diff', 0):.4f}
 95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
 t-statistic: {diff['t_statistic']:.3f} (df = {diff.get('degrees_freedom', 0):.1f})
 p-value: {diff['p_value']:.6f}
 Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
 {diff['baseline_algorithm']}: {diff['mean_baseline']:.4f} ± {diff.get('std_baseline', 0):.4f}
 {diff['comparison_algorithm']}: {diff['mean_comparison']:.4f} ± {diff.get('std_comparison', 0):.4f}
"""
   
   if negative_diffs:
       report += f"""
SIGNIFICANT ALGORITHM BENEFITS (p < {alpha})
{'-' * 50}
These configurations show significantly lower resource consumption
with larger algorithms (unexpected but possible due to optimization effects).

"""
       
       for diff in sorted(negative_diffs, key=lambda x: x['mean_diff']):
           effect_size = interpret_cohens_d(diff['cohens_d'])
           sample_info = ""
           if diff.get('sample_size_warning', False):
               n_base = diff.get('n_baseline', 15)  
               n_comp = diff.get('n_comparison', 15)
               sample_info = f" (n_base={n_base}, n_comp={n_comp})"
           
           report += f"""
{diff['config_label']} ({diff['comparison_algorithm']} vs {diff['baseline_algorithm']}){sample_info}:
 Algorithm benefit: {diff['mean_diff']:.4f} ± {diff.get('se_diff', 0):.4f}
 95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}]
 t-statistic: {diff['t_statistic']:.3f} (df = {diff.get('degrees_freedom', 0):.1f})
 p-value: {diff['p_value']:.6f}
 Effect size (Cohen's d): {diff['cohens_d']:.3f} ({effect_size})
 {diff['baseline_algorithm']}: {diff['mean_baseline']:.4f} ± {diff.get('std_baseline', 0):.4f}
 {diff['comparison_algorithm']}: {diff['mean_comparison']:.4f} ± {diff.get('std_comparison', 0):.4f}
"""
   
   if not significant_diffs:
       report += f"""NO SIGNIFICANT ALGORITHM SCALING EFFECTS FOUND
{'-' * 50}
All configurations show no statistically significant difference
between {baseline_algorithm} and larger algorithms.
"""
   
   report += f"""
METHODOLOGICAL NOTES
{'-' * 30}
- Statistical Test: Welch's t-test (handles unequal sample sizes and variances)
- Error Propagation: SE_diff = √(SE_baseline² + SE_comparison²)
- Effect Size: Cohen's d using pooled standard deviation
- Data Type: {'Filtered dataset (outliers removed)' if use_filtered else 'Original dataset'}

RECOMMENDATIONS
{'-' * 30}
"""
   
   if len(positive_diffs) > len(differences) * 0.7:
       report += "• HIGH SCALING SENSITIVITY: Most configurations show significant penalties with larger algorithms\n"
       report += f"• Consider using {baseline_algorithm} for resource-constrained environments\n"
   elif len(positive_diffs) > len(differences) * 0.3:
       report += "• MODERATE SCALING SENSITIVITY: Some configurations affected by algorithm size\n"
   else:
       report += "• LOW SCALING SENSITIVITY: Algorithm size has minimal resource impact\n"
   
   if len(negative_diffs) > 0:
       report += f"• OPTIMIZATION EFFECTS: {len(negative_diffs)} configurations show benefits with larger algorithms\n"
   
   if use_filtered and sample_warnings:
       report += f"""
FILTERED DATA CONSIDERATIONS
{'-' * 30}
- {len(sample_warnings)} configurations have different sample sizes
- Statistical tests account for unequal sample sizes using Welch's method
- Results represent typical behavior after removing extreme outliers
"""
   
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
    print("Enhanced CoAP Benchmark Comparison Tool")
    print("Loading data with filtered file support...")
    
    # Simple control: use filtered files when available, fall back to original
    USE_FILTERED = True  # Set this to False to use only original files
    
    if USE_FILTERED:
        print("Using filtered data files when available (falling back to original for fiducial network)...")
    else:
        print("Using original data files only...")
    
    all_data = load_all(use_filtered=USE_FILTERED)
    
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
        NORMALIZE = False
        plot_tradeoff(all_data, 'duration', 'Energy (Wh)', scen, normalize=NORMALIZE, output_dir=OUT_DIR)
        plot_tradeoff(all_data, 'cpu_cycles', 'Energy (Wh)', scen, normalize=NORMALIZE, output_dir=OUT_DIR)
        plot_tradeoff(all_data, 'duration', 'cpu_cycles', scen, normalize=NORMALIZE, output_dir=OUT_DIR)

    # Spider plots
    for scen in SCENARIOS:
        for norm in ['baseline_relative']:
            print(f"\nCreating Spider plot with {norm}-normalization for scenario {scen}...")
            create_spider_plot(
            all_data,
                metrics=['duration', 'cpu_cycles', 'Energy (Wh)', 'Power (W)', 'Max Power (W)'],
                scenario=scen,
                security_modes=['psk'],
                algorithms=['KYBER_LEVEL1', 'KYBER_LEVEL3', 'KYBER_LEVEL5'],  # Specify algorithm(s)
                certificates=None,    # Specify certificate(s)
                normalization=norm,
                output_dir=OUT_DIR
            )
            
    for scen in SCENARIOS:
        # Network difference plot (now enhanced but with same interface)
        print(f"\nCreating network difference plot for scenario {scen}...")
        create_network_difference_plot(
            all_data=all_data,
            metric='Energy (Wh)',
            scenario=scen,
            #sample_size=15,
            comparison_networks=['smarthome'],
            security_modes=['pki','psk','nosec'],
            algorithms=['KYBER_LEVEL1', 'KYBER_LEVEL3', 'KYBER_LEVEL5'],
            alpha=0.05,
            output_dir=OUT_DIR
        )
        # Network difference plot (now enhanced but with same interface)
        print(f"\nCreating algorithm difference plot for scenario {scen}...")
        create_algorithm_difference_plot(
            all_data=all_data,
            metric='Energy (Wh)',
            scenario=scen,
            baseline_algorithm='KYBER_LEVEL1',
            comparison_algorithms=['KYBER_LEVEL5'],
            #sample_size=15,
            networks=['fiducial'],
            security_modes=['pki', 'psk'],
            #certificates=['RSA_2048', 'DILITHIUM_LEVEL2'],  # Subset for clarity
            alpha = 0.05,
            output_dir=OUT_DIR
        )
    
    #analyze_data_structure(all_data=all_data, metric='Energy (Wh)', scenario='A')
    
    print("All plots generated successfully!")