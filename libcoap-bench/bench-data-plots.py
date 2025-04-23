import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse

# Generate the color palette
color_palette = list(mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(np.linspace(0, 1, 9)))
security_mode_colors = {
    "pki": color_palette[0],
    "psk": color_palette[3],
    "nosec": color_palette[6]
}

def read_csv(file_path, metric_column):
    """
    Read CSV file and extract the metric value and standard deviation.
    
    Args:
        file_path (str): Path to the CSV file.
        metric_column (str): Name of the column to extract.
        
    Returns:
        tuple: (metric_value, std_dev_value) or (None, None) if file not found or other error.
    """
    try:
        # Read CSV file with semicolon delimiter
        df = pd.read_csv(file_path, sep=';')
        
        # Get the last but one row (mean value) and the specified metric column
        metric_value = float(df.iloc[-2][metric_column])
        # Get the last row (standard deviation) and the specified metric column
        std_dev_value = float(df.iloc[-1][metric_column])
        
        return metric_value, std_dev_value
    except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def find_files(base_dir, pattern):
    """
    Find files matching a pattern in the base directory
    
    Args:
        base_dir (str): Base directory to search in.
        pattern (str): File pattern to search for.
        
    Returns:
        list: List of matching file paths.
    """
    try:
        matching_files = glob.glob(os.path.join(base_dir, pattern))
        return matching_files
    except Exception as e:
        print(f"Error searching for files: {e}")
        return []

def get_certificate_colors(cert_types_list):
    """
    Generate colors for different certificate types.
    
    Args:
        cert_types_list (list): List of certificate types.
        
    Returns:
        dict: Dictionary mapping certificate types to colors.
    """
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
    
    return cert_colors

def get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp=False):
    """
    Generate file patterns for various security modes.
    
    Args:
        algorithm (str): Algorithm name.
        cert_type (str): Certificate type.
        n (int): Number of repetitions.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter.
        scenario (str): Scenario identifier.
        rasp (bool): Whether the data is from a Raspberry Pi.
        
    Returns:
        dict: Dictionary containing file patterns for different security modes.
    """
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    rasp_prefix = "_rasp" if rasp else ""
    
    patterns = {
        'pki': f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki{scenario_suffix}.csv",
        'pki_client_auth': f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki_client-auth{scenario_suffix}.csv",
        'psk': f"udp{rasp_prefix}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_psk{scenario_suffix}.csv",
        'nosec': f"udp{rasp_prefix}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}.csv",
        'nosec_flexible': f"udp{rasp_prefix}_conv_stats*n{n}*_nosec*{scenario_suffix}.csv"
    }
    
    return patterns

def setup_output_dirs(custom_suffix=None):
    """
    Setup input and output directories based on custom suffix.
    
    Args:
        custom_suffix (str or None): Optional suffix for data and plot directories.
        
    Returns:
        tuple: (data_dir, plots_dir) directory names.
    """
    if custom_suffix:
        data_dir = f"bench-data-{custom_suffix}"
        plots_dir = f"bench-plots-{custom_suffix}"
    else:
        data_dir = "bench-data"
        plots_dir = "bench-plots"
    
    return data_dir, plots_dir

def create_scatter_plot(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None):
    """
    Create scatter plots for the specified metric and algorithms under different security modes.
    
    Args:
        metric (str): The metric to be plotted.
        algorithms_list (list): List of algorithm names.
        cert_types_list (list): List of certificate types to include.
        n (int): Number of repetitions.
        scenario (str): Scenario identifier (A, B, or C).
        rasp (bool): Whether the data is from a Raspberry Pi.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter (parallelization mode).
        data_dir (str): Directory containing the data files.
        custom_suffix (str): Optional suffix for data and plot directories.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(custom_suffix)
    data_dir_path = os.path.join(script_directory, data_dir_)
    
    print(f"Data directory: {data_dir_path}")
    print(f"Plot directory: {plots_dir}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set colors for different modes
    pki_base_color = color_palette[0]
    psk_color = color_palette[1]
    nosec_color = color_palette[2]
    
    # Generate additional colors for different certificate types
    cert_colors = get_certificate_colors(cert_types_list)

    # Construct the common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    rasp_prefix = "rasp" if rasp else ""

    # Data structures to hold values for each algorithm and certificate type
    data = {
        'psk': {
            'values': [],
            'std_devs': [],
            'algorithms': []
        },
        'nosec': {
            'value': None,
            'std_dev': None
        }
    }
    
    # Add entries for each certificate type
    for cert_type in cert_types_list:
        data[cert_type] = {
            'values': [],
            'std_devs': [],
            'algorithms': []
        }

    # Get PSK data for each algorithm
    for algorithm in algorithms_list:
        # Get file patterns
        patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp)
        
        # PSK file pattern
        psk_pattern = patterns['psk']
        psk_files = find_files(data_dir_path, psk_pattern)
        
        if psk_files:
            psk_file_path = psk_files[0]
            metric_value_psk, std_dev_psk = read_csv(psk_file_path, metric)
            if metric_value_psk is not None:
                data['psk']['values'].append(metric_value_psk)
                data['psk']['std_devs'].append(std_dev_psk)
                data['psk']['algorithms'].append(algorithm)
           #     print(f"Found PSK file for {algorithm}: {psk_file_path}")
           # else:
           #     print(f"Warning: Could not read PSK data from {psk_file_path}")
        else:
            print(f"Warning: Could not find PSK file for algorithm {algorithm}")

    # Get PKI data for each algorithm and certificate type
    for cert_type in cert_types_list:
        for algorithm in algorithms_list:
            # Get file patterns
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp)
            
            # PKI file pattern
            pki_pattern = patterns['pki']
            pki_files = find_files(data_dir_path, pki_pattern)
            
            # Try with client-auth suffix if needed
            if not pki_files:
                pki_pattern = patterns['pki_client_auth']
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                pki_file_path = pki_files[0]
                metric_value_pki, std_dev_pki = read_csv(pki_file_path, metric)
                if metric_value_pki is not None:
                    data[cert_type]['values'].append(metric_value_pki)
                    data[cert_type]['std_devs'].append(std_dev_pki)
                    data[cert_type]['algorithms'].append(algorithm)
                #    print(f"Found PKI file for {algorithm} with {cert_type}: {pki_file_path}")
                #else:
                #    print(f"Warning: Could not read PKI data from {pki_file_path}")
            else:
                print(f"Warning: Could not find PKI file for algorithm {algorithm} with {cert_type}")
    
    # Get nosec data
    patterns = get_file_patterns("", "", n, s, p, scenario, rasp)
    nosec_pattern = patterns['nosec']
    nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if not nosec_files:
        # Try a more flexible pattern
        nosec_pattern = patterns['nosec_flexible']
        nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if nosec_files:
        nosec_file_path = nosec_files[0]
        metric_value_horizontal, std_dev_horizontal = read_csv(nosec_file_path, metric)
        if metric_value_horizontal is not None:
            data['nosec']['value'] = metric_value_horizontal
            data['nosec']['std_dev'] = std_dev_horizontal
        #    print(f"Using nosec file: {nosec_file_path}")
        #else:
        #    print(f"Warning: Could not read nosec data from {nosec_file_path}")
    else:
        print(f"Warning: Could not find nosec file matching pattern {nosec_pattern}")

    # Plot data for each certificate type
    for cert_type in cert_types_list:
        if data[cert_type]['values']:
            ax.errorbar(
                data[cert_type]['algorithms'], 
                data[cert_type]['values'], 
                yerr=data[cert_type]['std_devs'], 
                label=f'PKI ({cert_type})', 
                fmt='o', 
                capsize=5, 
                color=cert_colors[cert_type]
            )
            ax.plot(
                data[cert_type]['algorithms'], 
                data[cert_type]['values'], 
                '--', 
                color=cert_colors[cert_type]
            )
    
    # Plot PSK data
    if data['psk']['values']:
        ax.errorbar(
            data['psk']['algorithms'], 
            data['psk']['values'], 
            yerr=data['psk']['std_devs'], 
            label='PSK', 
            fmt='o', 
            capsize=5, 
            color=psk_color
        )
        ax.plot(
            data['psk']['algorithms'], 
            data['psk']['values'], 
            '--', 
            color=psk_color
        )

    # Plot nosec data
    if data['nosec']['value'] is not None:
        ax.errorbar(
            algorithms_list, 
            [data['nosec']['value']] * len(algorithms_list), 
            yerr=data['nosec']['std_dev'], 
            label='NoSec', 
            fmt='o', 
            capsize=5, 
            color=nosec_color
        )
        ax.plot(
            algorithms_list, 
            [data['nosec']['value']] * len(algorithms_list), 
            '--', 
            color=nosec_color
        )

    # Set labels and title
    ax.set_xlabel('Algorithms')
    ax.set_ylabel(metric)
    title = f'{metric} by algorithm and security mode - n={n}, scenario={scenario}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', p={p}'
    ax.set_title(title)
    
    # Calculate the minimum and maximum values considering error bars
    values_to_consider = []
    std_devs_to_consider = []
    
    # Include all certificate types
    for cert_type in cert_types_list:
        if data[cert_type]['values']:
            values_to_consider.extend(data[cert_type]['values'])
            std_devs_to_consider.extend(data[cert_type]['std_devs'])
    
    # Include PSK
    if data['psk']['values']:
        values_to_consider.extend(data['psk']['values'])
        std_devs_to_consider.extend(data['psk']['std_devs'])
    
    # Include NoSec
    if data['nosec']['value'] is not None:
        values_to_consider.append(data['nosec']['value'])
        std_devs_to_consider.append(data['nosec']['std_dev'])
    
    if values_to_consider and std_devs_to_consider:
        min_value = min([v - s for v, s in zip(values_to_consider, std_devs_to_consider)])
        max_value = max([v + s for v, s in zip(values_to_consider, std_devs_to_consider)])
        
        # Calculate the range based on the maximum absolute value
        value_range = 0.1 * max(abs(min_value), abs(max_value))
        
        # Set y-axis limits based on the calculated range
        ax.set_ylim(min_value - value_range, max_value + value_range)

    # Show legend
    ax.legend()

    # Save the plot
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/scatter_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}.pdf'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

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
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(custom_suffix)
    data_dir_path = os.path.join(script_directory, data_dir_)
    
    print(f"Data directory: {data_dir_path}")
    print(f"Plot directory: {plots_dir}")
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate colors for different certificate types
    cert_colors = get_certificate_colors(cert_types_list)

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
    
    # Common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    rasp_prefix = "rasp" if rasp else ""
    
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
                # Get file patterns
                patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp)
                
                # PKI file pattern
                pki_pattern = patterns['pki']
                pki_files = find_files(data_dir_path, pki_pattern)
                
                # Try with client-auth if needed
                if not pki_files:
                    pki_pattern = patterns['pki_client_auth']
                    pki_files = find_files(data_dir_path, pki_pattern)
                
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
            patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp)
            psk_pattern = patterns['psk']
            psk_files = find_files(data_dir_path, psk_pattern)
            
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
        
        # Get file patterns
        patterns = get_file_patterns("", "", n, s, p, scenario, rasp)
        
        # NoSec file pattern
        nosec_pattern = patterns['nosec']
        nosec_files = find_files(data_dir_path, nosec_pattern)
        
        # Try a more flexible pattern if needed
        if not nosec_files:
            nosec_pattern = patterns['nosec_flexible']
            nosec_files = find_files(data_dir_path, nosec_pattern)
        
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
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    scenarios_str = "_".join(scenarios)
    
    # Create the plots directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/barplot_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}.pdf'

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()
    
def create_heat_map(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None):
    """
    Create a heat map visualization of a metric across algorithms and certificate types.
    
    Args:
        metric (str): The metric to be visualized (e.g., 'duration', 'Energy (Wh)').
        algorithms_list (list): List of algorithm names.
        cert_types_list (list): List of certificate types to include.
        n (int): Number of repetitions.
        scenario (str): Scenario identifier (A, B, or C).
        rasp (bool): Whether the data is from a Raspberry Pi.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter (parallelization mode).
        data_dir (str): Directory containing the data files.
        custom_suffix (str): Optional suffix for data and plot directories.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(custom_suffix)
    data_dir_path = os.path.join(script_directory, data_dir_)
    
    print(f"Data directory: {data_dir_path}")
    print(f"Plot directory: {plots_dir}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a grid to store the metric values
    heat_data = np.zeros((len(algorithms_list), len(cert_types_list)))
    
    # Construct the common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    rasp_prefix = "rasp" if rasp else ""
    
    # Variables to track min and max values for color scaling
    valid_data_found = False
    min_value = float('inf')
    max_value = float('-inf')
    
    # Collect data for the heat map
    for alg_idx, algorithm in enumerate(algorithms_list):
        for cert_idx, cert_type in enumerate(cert_types_list):
            # Get file patterns
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp)
            
            # PKI file pattern
            pki_pattern = patterns['pki']
            pki_files = find_files(data_dir_path, pki_pattern)
            
            # Try with client-auth if needed
            if not pki_files:
                pki_pattern = patterns['pki_client_auth']
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                file_path = pki_files[0]
                metric_value, _ = read_csv(file_path, metric)
                
                if metric_value is not None:
                    heat_data[alg_idx, cert_idx] = metric_value
                    min_value = min(min_value, metric_value)
                    max_value = max(max_value, metric_value)
                    valid_data_found = True
                else:
                    # Use NaN for missing values
                    heat_data[alg_idx, cert_idx] = np.nan
            else:
                # Use NaN for missing files
                heat_data[alg_idx, cert_idx] = np.nan
    
    if not valid_data_found:
        print("No valid data found for heat map. Please check your parameters.")
        return
    
    # Create the heatmap
    cmap = plt.cm.YlOrRd  # Yellow-Orange-Red colormap (good for showing intensity)
    im = ax.imshow(heat_data, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(cert_types_list)))
    ax.set_yticks(np.arange(len(algorithms_list)))
    ax.set_xticklabels(cert_types_list)
    ax.set_yticklabels(algorithms_list)
    
    # Rotate the x labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations inside each cell showing the value
    for i in range(len(algorithms_list)):
        for j in range(len(cert_types_list)):
            if not np.isnan(heat_data[i, j]):
                # Format the text differently based on the value range
                if isinstance(heat_data[i, j], int):
                    text = ax.text(j, i, f"{heat_data[i, j]:.0f}",
                                  ha="center", va="center", color="black" if heat_data[i, j] < (max_value - min_value) * 0.7 + min_value else "white")
                else:
                    text = ax.text(j, i, f"{heat_data[i, j]:.4f}",
                                  ha="center", va="center", color="black" if heat_data[i, j] < (max_value - min_value) * 0.7 + min_value else "white")
    
    # Set labels and title
    ax.set_xlabel('Certificate Types')
    ax.set_ylabel('Algorithms')
    title = f'Heat Map of {metric} - n={n}, scenario={scenario}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', p={p}'
    ax.set_title(title)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/heatmap_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}.pdf'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()
    
def create_box_plot(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None):
    """
    Create box plots to visualize variability in metrics across configurations.
    
    Args:
        metric (str): The metric to be visualized (e.g., 'duration', 'Energy (Wh)').
        algorithms_list (list): List of algorithm names.
        cert_types_list (list): List of certificate types to include.
        n (int): Number of repetitions.
        scenario (str): Scenario identifier (A, B, or C).
        rasp (bool): Whether the data is from a Raspberry Pi.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter (parallelization mode).
        data_dir (str): Directory containing the data files.
        custom_suffix (str): Optional suffix for data and plot directories.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(custom_suffix)
    data_dir_path = os.path.join(script_directory, data_dir_)
    
    print(f"Data directory: {data_dir_path}")
    print(f"Plot directory: {plots_dir}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generate colors for different certificate types
    cert_colors = get_certificate_colors(cert_types_list)
    
    # Dictionary to store raw data points for each configuration
    raw_data_dict = {}
    
    # Construct the common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    rasp_prefix = "rasp" if rasp else ""
    
    # Collect raw data for each configuration
    for algorithm in algorithms_list:
        for cert_type in cert_types_list:
            # Get file patterns
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp)
            
            # PKI file pattern
            pki_pattern = patterns['pki']
            pki_files = find_files(data_dir_path, pki_pattern)
            
            # Try with client-auth if needed
            if not pki_files:
                pki_pattern = patterns['pki_client_auth']
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                file_path = pki_files[0]
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path, sep=';')
                    
                    # Find the separator row
                    separator_idx = None
                    for i, row in df.iterrows():
                        # Check if this is a separator row
                        if isinstance(row.iloc[0], str) and '---' in row.iloc[0]:
                            separator_idx = i
                            break
                    
                    if separator_idx is not None:
                        # Get all data rows before the separator
                        data_rows = df.iloc[:separator_idx]
                        
                        # Extract the metric column
                        if metric in data_rows.columns:
                            # Filter out non-numeric values (if any)
                            raw_values = pd.to_numeric(data_rows[metric], errors='coerce').dropna().values
                            
                            # Store values in the dictionary
                            key = f"{algorithm}_{cert_type}"
                            raw_data_dict[key] = {
                                'values': raw_values,
                                'algorithm': algorithm,
                                'cert_type': cert_type,
                                'color': cert_colors[cert_type]
                            }
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    # Also get data for PSK and NoSec
    for algorithm in algorithms_list:
        # PSK data
        patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp)
        psk_pattern = patterns['psk']
        psk_files = find_files(data_dir_path, psk_pattern)
        
        if psk_files:
            file_path = psk_files[0]
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, sep=';')
                
                # Find the separator row
                separator_idx = None
                for i, row in df.iterrows():
                    # Check if this is a separator row
                    if isinstance(row.iloc[0], str) and '---' in row.iloc[0]:
                        separator_idx = i
                        break
                
                if separator_idx is not None:
                    # Get all data rows before the separator
                    data_rows = df.iloc[:separator_idx]
                    
                    # Extract the metric column
                    if metric in data_rows.columns:
                        # Filter out non-numeric values (if any)
                        raw_values = pd.to_numeric(data_rows[metric], errors='coerce').dropna().values
                        
                        # Store values in the dictionary
                        key = f"{algorithm}_PSK"
                        raw_data_dict[key] = {
                            'values': raw_values,
                            'algorithm': algorithm,
                            'cert_type': 'PSK',
                            'color': security_mode_colors['psk']
                        }
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    # NoSec data
    patterns = get_file_patterns("", "", n, s, p, scenario, rasp)
    nosec_pattern = patterns['nosec']
    nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if not nosec_files:
        # Try a more flexible pattern
        nosec_pattern = patterns['nosec_flexible']
        nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if nosec_files:
        file_path = nosec_files[0]
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, sep=';')
            
            # Find the separator row
            separator_idx = None
            for i, row in df.iterrows():
                # Check if this is a separator row
                if isinstance(row.iloc[0], str) and '---' in row.iloc[0]:
                    separator_idx = i
                    break
            
            if separator_idx is not None:
                # Get all data rows before the separator
                data_rows = df.iloc[:separator_idx]
                
                # Extract the metric column
                if metric in data_rows.columns:
                    # Filter out non-numeric values (if any)
                    raw_values = pd.to_numeric(data_rows[metric], errors='coerce').dropna().values
                    
                    # Store values in the dictionary
                    key = f"NoSec"
                    raw_data_dict[key] = {
                        'values': raw_values,
                        'algorithm': 'NoSec',
                        'cert_type': 'NoSec',
                        'color': security_mode_colors['nosec']
                    }
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    # If no data was found, exit
    if not raw_data_dict:
        print("No valid data found for box plots. Please check your parameters.")
        return
    
    # Prepare box plot data
    box_data = []
    box_colors = []
    labels = []
    
    # Sort by algorithm and certificate type for consistent ordering
    for key in sorted(raw_data_dict.keys()):
        data = raw_data_dict[key]
        if len(data['values']) > 0:
            box_data.append(data['values'])
            box_colors.append(data['color'])
            # Format the label
            if data['cert_type'] == 'PSK':
                labels.append(f"{data['algorithm']}\nPSK")
            elif data['cert_type'] == 'NoSec':
                labels.append("NoSec")
            else:
                labels.append(f"{data['algorithm']}\n{data['cert_type']}")
    
    # Create box plot
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True, medianprops={'color': 'black'})
    
    # Set colors for each box
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric)
    title = f'Variability of {metric} across configurations - n={n}, scenario={scenario}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', p={p}'
    ax.set_title(title)
    
    # Add grid for easier reading
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Create a custom legend
    handles = []
    legend_labels = []
    
    # Add PKI certificate types to legend
    for cert_type in cert_types_list:
        handles.append(Patch(facecolor=cert_colors[cert_type], alpha=0.7))
        legend_labels.append(f"PKI ({cert_type})")
    
    # Add PSK to legend
    handles.append(Patch(facecolor=security_mode_colors['psk'], alpha=0.7))
    legend_labels.append("PSK")
    
    # Add NoSec to legend if available
    if "NoSec" in [data['cert_type'] for data in raw_data_dict.values()]:
        handles.append(Patch(facecolor=security_mode_colors['nosec'], alpha=0.7))
        legend_labels.append("NoSec")
    
    # Add the legend
    ax.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    clean_metric = metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/boxplot_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}.pdf'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

def parse_args():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description='CoAP Benchmark Plotting Tool',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('metric', help='Metric to be plotted (e.g., "duration", "Energy (Wh)")')
    parser.add_argument('algorithms', help='Comma-separated list of algorithms (e.g., "KYBER_LEVEL1,KYBER_LEVEL3")')
    parser.add_argument('cert_types', help='Comma-separated list of certificate types (e.g., "DILITHIUM_LEVEL2,RSA_2048")')
    parser.add_argument('n', type=int, help='Number of repetitions')
    
    # Plot type selection
    plot_type = parser.add_mutually_exclusive_group(required=True)
    plot_type.add_argument('--scatter', action='store_true', help='Generate scatter plot')
    plot_type.add_argument('--barplot', action='store_true', help='Generate bar plot')
    plot_type.add_argument('--heatmap', action='store_true', help='Generate heat map')
    plot_type.add_argument('--boxplot', action='store_true', help='Generate box plot for variability analysis')
    
    # Scenarios
    parser.add_argument('--scenarios', default='A', 
                        help='Comma-separated list of scenarios (e.g., "A,B,C"). For scatter, heatmap, boxplot, radar, and efficiency plots, only the first scenario is used.')
    
    # Optional arguments
    parser.add_argument('--rasp', action='store_true', help='Use Raspberry Pi dataset')
    parser.add_argument('--s', type=int, help='Optional s parameter')
    parser.add_argument('--p', help='Optional p parameter (parallelization mode)')
    parser.add_argument('--data-dir', default='bench-data', help='Directory containing the data files')
    parser.add_argument('--custom-suffix', help='Suffix for data and plot directories')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process scenarios
    scenarios = [scenario.strip().upper() for scenario in args.scenarios.split(',')]
    
    # Validate scenarios
    for scenario in scenarios:
        if scenario not in ['A', 'B', 'C']:
            parser.error(f"Invalid scenario: {scenario}. Must be one of 'A', 'B', or 'C'.")
    
    # Process algorithms and certificate types
    algorithms = [alg.strip() for alg in args.algorithms.split(',')]
    cert_types = [cert.strip() for cert in args.cert_types.split(',')]
    
    return args, algorithms, cert_types, scenarios

def main():
    """Main function to parse arguments and generate plots."""
    args, algorithms, cert_types, scenarios = parse_args()
    
    # Use only the first scenario for plots that don't support multiple scenarios
    scenario = scenarios[0]
    
    if args.scatter:
        print(f"Generating scatter plot for scenario {scenario}...")
        create_scatter_plot(
            args.metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix
        )
    elif args.heatmap:
        print(f"Generating heat map for scenario {scenario}...")
        create_heat_map(
            args.metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix
        )
    elif args.boxplot:
        print(f"Generating box plot for scenario {scenario}...")
        create_box_plot(
            args.metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix
        )
    else:
        # Bar plot supports multiple scenarios
        print(f"Generating bar plot for scenarios {', '.join(scenarios)}...")
        create_bar_plot(
            args.metric, algorithms, cert_types, args.n, scenarios,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix
        )

if __name__ == "__main__":
    main()
    