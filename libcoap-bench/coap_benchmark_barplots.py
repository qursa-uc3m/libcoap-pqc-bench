import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import glob

# Generate the color palette
color_palette = list(mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(np.linspace(0, 1, 9)))
security_mode_colors = {
    "pki": color_palette[0],
    "psk": color_palette[3],
    "nosec": color_palette[6]
}

def read_csv(file_path, metric_column):
    """
    Read CSV file and extract the metric and standard deviation values from specified columns.
    
    Args:
        file_path (str): Path to the CSV file.
        metric_column (str): Column name for the metric value.

    Returns:
        tuple: Metric value and standard deviation value, or None, None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, sep=';')
        metric_value = float(df.iloc[-2][metric_column])
        std_dev_value = float(df.iloc[-1][metric_column])
        return metric_value, std_dev_value
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None, None
    except KeyError:
        print(f"Error: Column '{metric_column}' not found in {file_path}")
        return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def find_matching_files(base_dir, pattern):
    """
    Find all files in the directory that match the given pattern.
    
    Args:
        base_dir (str): Base directory to search in.
        pattern (str): File pattern to search for.
        
    Returns:
        list: List of matching file paths.
    """
    try:
        return glob.glob(os.path.join(base_dir, pattern))
    except Exception as e:
        print(f"Error finding files: {e}")
        return []

def create_bar_plot(metric, algorithms_list, cert_types_list, n, scenarios, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None):
    """
    Create bar plot for the specified metric and algorithms under different security modes and scenarios.

    Args:
        metric (str): The metric to be plotted.
        algorithms_list (list): List of algorithm names.
        cert_types_list (list): List of certificate types to include.
        n (int): Number of repetitions.
        scenarios (list): List of scenarios to consider.
        rasp (bool): If True, use Rasp dataset.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter.
        data_dir (str): Directory containing the data files.
        custom_suffix (str): Optional suffix for data and plot directories.
    """
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Handle custom directory naming
    if custom_suffix:
        data_dir = f"bench-data-{custom_suffix}"
        plots_dir = f"bench-plots-{custom_suffix}"
    else:
        plots_dir = "bench-plots"
    
    data_dir_path = os.path.join(script_directory, data_dir)
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate the number of types of bars we'll have
    num_bar_types = 1 + len(cert_types_list)  # PSK + all certificate types (PKI)
    
    # Calculate the number of bar groups (algorithm + scenario combinations)
    num_algorithms = len(algorithms_list)
    num_scenarios = len(scenarios)
    num_groups = num_algorithms * num_scenarios
    
    # Define the spacing between groups and individual bars
    group_width = num_bar_types * 1.2  # Total width for a group of bars
    bar_width = 1.0  # Width of each bar
    group_spacing = 2.0  # Space between groups
    
    # Calculate positions for each group
    group_positions = np.arange(num_groups) * (group_width + group_spacing)
    
    # Calculate position for nosec bars (separate section)
    nosec_start = group_positions[-1] + group_width + group_spacing if num_groups > 0 else 0
    nosec_positions = np.array([nosec_start + i * 1.5 for i in range(num_scenarios)])
    
    # Create a dictionary to store bar positions
    bar_positions = {}
    x_labels = []
    
    # Generate colors for different certificate types
    cert_colors = {}
    pki_base_color = security_mode_colors["pki"]
    num_cert_types = len(cert_types_list)
    
    if num_cert_types > 1:
        # Generate a gradient of colors for multiple certificate types
        cert_colors_list = list(mcolors.LinearSegmentedColormap.from_list(
            "", [pki_base_color, "#006400"])(np.linspace(0, 1, num_cert_types)))
        for i, cert_type in enumerate(cert_types_list):
            cert_colors[cert_type] = cert_colors_list[i]
    else:
        # Just use the base PKI color if there's only one certificate type
        for cert_type in cert_types_list:
            cert_colors[cert_type] = pki_base_color
    
    # Common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    rasp_prefix = "_rasp" if rasp else ""
    
    # Dictionary to store legend entries
    legend_entries = {}
    
    # Process and plot each algorithm and scenario combination
    group_idx = 0
    for alg_idx, algorithm in enumerate(algorithms_list):
        for scen_idx, scenario in enumerate(scenarios):
            scenario_suffix = f"_scenario{scenario}"
            group_center = group_positions[group_idx]
            bar_idx = 0
            
            # Set the x label for this group
            x_labels.append((group_center, f"{algorithm}-{scenario}"))
            
            # Process each certificate type (PKI)
            for cert_idx, cert_type in enumerate(cert_types_list):
                # PKI file pattern
                pki_pattern = f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki{scenario_suffix}.csv"
                pki_files = find_matching_files(data_dir_path, pki_pattern)
                print(f"Searching for PKI files with pattern: {pki_pattern}")
                print(f"Files found: {pki_files}")
                
                # Try with client-auth if needed
                if not pki_files:
                    pki_pattern = f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki_client-auth{scenario_suffix}.csv"
                    pki_files = find_matching_files(data_dir_path, pki_pattern)
                
                if pki_files:
                    file_path = pki_files[0]
                    metric_value, std_dev_value = read_csv(file_path, metric)
                    
                    if metric_value is not None and std_dev_value is not None:
                        bar_pos = group_center + (bar_idx - (num_bar_types - 1) / 2) * 1.2
                        
                        # Add error bars
                        (_, caps, _) = ax.errorbar(bar_pos, metric_value, yerr=std_dev_value, 
                                                  capsize=5, fmt='o', color='black', markersize=5)
                        for cap in caps:
                            cap.set_markeredgewidth(1)
                        
                        # Add the bar
                        ax.bar(bar_pos, metric_value, width=bar_width, 
                               color=cert_colors[cert_type], edgecolor='black', linewidth=1)
                        
                        # Add to legend entries
                        legend_name = f'PKI ({cert_type})'
                        legend_entries[legend_name] = cert_colors[cert_type]
                        
                        # Store the bar position
                        key = (algorithm, scenario, f"PKI ({cert_type})")
                        bar_positions[key] = bar_pos
                
                bar_idx += 1
            
            # Process PSK
            psk_pattern = f"udp{rasp_prefix}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_psk{scenario_suffix}.csv"
            psk_files = find_matching_files(data_dir_path, psk_pattern)
            print(f"Searching for PKI files with pattern: {psk_pattern}")
            print(f"Files found: {psk_files}")
            
            if psk_files:
                file_path = psk_files[0]
                metric_value, std_dev_value = read_csv(file_path, metric)
                
                if metric_value is not None and std_dev_value is not None:
                    bar_pos = group_center + (bar_idx - (num_bar_types - 1) / 2) * 1.2
                    
                    # Add error bars
                    (_, caps, _) = ax.errorbar(bar_pos, metric_value, yerr=std_dev_value, 
                                              capsize=5, fmt='o', color='black', markersize=5)
                    for cap in caps:
                        cap.set_markeredgewidth(1)
                    
                    # Add the bar
                    ax.bar(bar_pos, metric_value, width=bar_width, 
                           color=security_mode_colors['psk'], edgecolor='black', linewidth=1)
                    
                    # Add to legend entries
                    legend_entries['PSK'] = security_mode_colors['psk']
                    
                    # Store the bar position
                    key = (algorithm, scenario, 'PSK')
                    bar_positions[key] = bar_pos
            
            group_idx += 1
    
    # Process and plot nosec for each scenario
    for scen_idx, scenario in enumerate(scenarios):
        scenario_suffix = f"_scenario{scenario}"
        
        # NoSec file pattern
        nosec_pattern = f"udp{rasp_prefix}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}.csv"
        nosec_files = find_matching_files(data_dir_path, nosec_pattern)
        
        # Try a more flexible pattern if needed
        if not nosec_files:
            nosec_pattern = f"udp{rasp_prefix}_conv_stats*n{n}*_nosec*{scenario_suffix}.csv"
            nosec_files = find_matching_files(data_dir_path, nosec_pattern)
        
        if nosec_files:
            file_path = nosec_files[0]
            metric_value, std_dev_value = read_csv(file_path, metric)
            
            if metric_value is not None and std_dev_value is not None:
                bar_pos = nosec_positions[scen_idx]
                
                # Add error bars
                (_, caps, _) = ax.errorbar(bar_pos, metric_value, yerr=std_dev_value, 
                                          capsize=5, fmt='o', color='black', markersize=5)
                for cap in caps:
                    cap.set_markeredgewidth(1)
                
                # Add the bar
                ax.bar(bar_pos, metric_value, width=bar_width, 
                       color=security_mode_colors['nosec'], edgecolor='black', linewidth=1)
                
                # Add to legend entries
                legend_entries['NoSec'] = security_mode_colors['nosec']
                
                # Add nosec label
                x_labels.append((bar_pos, f"NoSec-{scenario}"))
    
    # Set the x-ticks and labels
    xtick_positions, xtick_labels = zip(*x_labels)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=10)

    # Set axis labels and title
    ax.set_xlabel('Algorithm - Scenario - Mode')
    ax.set_ylabel(metric)
    
    title = f'{metric} by algorithm, security mode, and scenario - n={n}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', p={p}'
    ax.set_title(title)
    
    # Create a custom legend
    handles = [Patch(facecolor=color, label=label) for label, color in legend_entries.items()]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)

    plt.tight_layout()
    
    # Generate the output filename
    algorithms_str = "_".join(algorithms_list)
    scenarios_str = "_".join(scenarios)
    cert_types_str = "_".join([cert_type.replace('_', '') for cert_type in cert_types_list])
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the plots directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/barplot_{"rasp_" if rasp else ""}{clean_metric}_n{n}_{s if s else ""}_{p if p else ""}_{algorithms_str}_{cert_types_str}_{scenarios_str}.png'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    # Updated usage instructions
    if len(sys.argv) < 7:
        print("Usage: python3 coap_benchmark_barplots.py <metric> <algorithms_list> <cert_types_list> <n> <rasp> <scenarios_list> [s] [p] [custom_suffix]")
        print("\nExample: python3 coap_benchmark_barplots.py 'duration' 'KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5' 'DILITHIUM_LEVEL2,RSA_2048' 50 true 'A,B,C'")
        print("\nFor energy metrics: python3 coap_benchmark_barplots.py 'Energy (Wh)' 'KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5' 'DILITHIUM_LEVEL2' 10 true 'A,C'")
        print("\nWith custom directory: python3 coap_benchmark_barplots.py 'duration' 'KYBER_LEVEL1' 'DILITHIUM_LEVEL2' 10 true 'A' 10 background test1")
        print("  This reads from bench-data-test1 and outputs to bench-plots-test1")
        sys.exit(1)

    # Parse the standard arguments
    metric = sys.argv[1]
    algorithms_list = [alg.strip() for alg in sys.argv[2].split(',')]
    cert_types_list = [cert.strip() for cert in sys.argv[3].split(',')]
    n = int(sys.argv[4])
    rasp = sys.argv[5].lower() == "true"
    scenarios_list = [scenario.strip() for scenario in sys.argv[6].split(',')]

    # Process optional arguments
    s, p, custom_suffix = None, None, None
    remaining_args = sys.argv[7:]
    
    # Parse s and p parameters (if provided)
    for i, arg in enumerate(remaining_args):
        if i == 0 and arg.isdigit():
            s = int(arg)
        elif i == 0 and not arg.isdigit():
            p = arg
        elif i == 1 and s is not None and not arg.isdigit():
            p = arg
        elif i == 1 and p is not None and arg.isdigit():
            s = int(arg)
        elif i == 2 or (i == 1 and s is None and p is None):
            custom_suffix = arg
    
    # Call the function with all parameters
    create_bar_plot(metric, algorithms_list, cert_types_list, n, scenarios_list, 
                   rasp, s, p, custom_suffix=custom_suffix)