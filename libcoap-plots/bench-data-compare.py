import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
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
        xlabel = f'{metric_x} (normalized)'
        ylabel = f'{metric_y} (normalized)'
    else:
        x_data_plot = x_data
        y_data_plot = y_data
        xlabel = metric_x
        ylabel = metric_y
    
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
            if not normalize:
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
    title = f"{metric_y} vs {metric_x} (Scenario {scenario})"
    
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

# MAIN
if __name__ == '__main__':
    print("Loading data from all networks and security modes...")
    all_data = load_all()
    
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
        plot_tradeoff(all_data, 'cpu_cycles', 'Energy (Wh)', scen)
        plot_tradeoff(all_data, 'cpu_cycles', 'Energy (Wh)', scen, normalize=True)
    
    print("All plots generated successfully!")