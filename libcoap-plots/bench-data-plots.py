import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import argparse

# Generate the color palette
color_palette = list(mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(np.linspace(0, 1, 9)))
security_mode_colors = {
    "pki": color_palette[0],
    "psk": color_palette[3],
    "nosec": color_palette[6]
}

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
        "font.size": 14,           # Base font size
        "axes.titlesize": 16,       # Title font size
        "axes.labelsize": 14,       # Axis label font size
        "xtick.labelsize": 12,      # x-tick label font size
        "ytick.labelsize": 12,      # y-tick label font size
        "legend.fontsize": 12,      # Legend font size
        "figure.titlesize": 18      # Figure title font size
    })
    
    # Improve figure aesthetics
    plt.rcParams.update({
        "figure.figsize": (10, 6),  # Default figure size
        "figure.dpi": 200,          # Figure resolution
        "savefig.dpi": 300,         # Saved figure resolution
        "savefig.format": "pdf",    # Default save format
        "savefig.bbox": "tight",    # Tight bounding box
        "savefig.pad_inches": 0.1   # Padding
    })
    
    # Improve axes, grid, and ticks
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.4,
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

def parse_metric_with_unit(metric_str):
    """
    Parse a metric string to extract the base metric and any unit conversion.
    
    Examples:
    - "duration" -> ("duration", None)
    - "duration ms" -> ("duration", "ms")
    - "Energy (Wh)" -> ("Energy (Wh)", None)
    - "Energy (mWh)" -> ("Energy (Wh)", "mWh")
    
    Args:
        metric_str (str): The metric string from command line
        
    Returns:
        tuple: (base_metric, target_unit)
    """
    # Check for duration with milliseconds
    if metric_str.lower() == "duration ms":
        return "duration", "ms"
    
    # Check for energy with milliwatt-hours
    if metric_str.lower() == "energy (mwh)":
        return "Energy (Wh)", "mWh"
    
    # No conversion needed
    return metric_str, None

def convert_value_based_on_unit(value, metric, target_unit):
    """
    Convert a value based on the target unit if needed.
    
    Args:
        value: The value to convert
        metric (str): The base metric name
        target_unit (str or None): The target unit, if conversion is needed
        
    Returns:
        The converted value or original value if no conversion needed
    """
    if value is None:
        return None
        
    # Convert duration from seconds to milliseconds
    if metric == "duration" and target_unit == "ms":
        return value * 1000.0
        
    # Convert energy from Wh to mWh
    if metric == "Energy (Wh)" and target_unit == "mWh":
        return value * 1000.0
        
    # No conversion needed
    return value

def get_display_metric_label(metric, target_unit):
    """
    Get the display label for a metric based on the target unit.
    
    Args:
        metric (str): The base metric name
        target_unit (str or None): The target unit, if conversion is needed
        
    Returns:
        str: The formatted metric label for display
    """
    if metric == "duration":
        if target_unit == "ms":
            return "duration ms"
        return metric
        
    if metric == "Energy (Wh)":
        if target_unit == "mWh":
            return "Energy (mWh)"
        return metric
        
    return metric

def format_labels(metric, target_unit=None):
    """
    Format metric names for LaTeX with proper units and mathematical notation.
    
    Args:
        metric (str): The metric name to format
        target_unit (str or None): The target unit, if conversion is needed
        
    Returns:
        str: Properly formatted LaTeX string with underscores removed
    """
    if not metric:
        return ""
        
    # Format based on metric type
    if "cpu_cycles" in metric.lower():
        return r'CPU Cycles ($\times 10^9$)'
    elif metric.lower() == "energy (wh)":
        return r'Energy (mWh)' if target_unit == "mWh" else r'Energy (Wh)'
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
    elif metric.lower() == "duration":
        return r'Duration (ms)' if target_unit == "ms" else r'Duration (s)'
    else:
        # Generic case - REMOVE underscores (not escape them)
        return metric.replace("_", " ").title()

def read_csv(file_path, metric_column, n, target_unit=None):
    """
    Read CSV file and extract the metric and standard deviation values from specified columns,
    applying unit conversion if needed.
    
    Args:
        file_path (str): Path to the CSV file.
        metric_column (str): Column name for the metric value.
        n (int): Number of samples
        target_unit (str or None): Target unit for conversion, if any

    Returns:
        tuple: Metric value and standard deviation value, or None, None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path, sep=';')
        
        # Separator row with dashes (------------)
        sep_idx = None
        for i, row in df.iterrows():
            first_col = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
            if '------------' in first_col:
                sep_idx = i
                break
            
        if sep_idx is None:
            print(f"Warning: Could not find separator row in {file_path}")
            return None, None
        
        mean_idx = sep_idx + 1
        std_idx = sep_idx + 2
        mode_idx = sep_idx + 3 
        
        discrete_metrics = ['frames_sent', 'bytes_sent', 'frames_received', 
                          'bytes_received', 'total_frames', 'total_bytes']
        
        if metric_column in df.columns:
            # Extract mean and std
            metric_value = pd.to_numeric(df.iloc[mean_idx][metric_column])
            std_dev_value = pd.to_numeric(df.iloc[std_idx][metric_column])
            
            # Apply unit conversion if needed
            metric_value = convert_value_based_on_unit(metric_value, metric_column, target_unit)
            std_dev_value = convert_value_based_on_unit(std_dev_value, metric_column, target_unit)
            
            # Extract mode
            mode_value = df.iloc[mode_idx][metric_column]
            
            # Discrete metrics: mode is the primary value
            if not pd.isna(mode_value) and metric_column in discrete_metrics:
                metric_value = pd.to_numeric(mode_value, errors='coerce')
                
                # Apply unit conversion if needed (for discrete metrics as well)
                metric_value = convert_value_based_on_unit(metric_value, metric_column, target_unit)
                
                # Estimate std from below/above mode counts
                below_count = pd.to_numeric(df.iloc[mode_idx + 1][metric_column], errors='coerce')
                above_count = pd.to_numeric(df.iloc[mode_idx + 2][metric_column], errors='coerce')
                
                # Measure of dispersion based on counts
                total_count = n # Total number of samples
                
                if total_count > 0:
                    # Spread relative to the mode, scaled by the mode value
                    mode_freq = n - (below_count + above_count)
                    std_dev_value = (1 - mode_freq / n) * metric_value
            
            return metric_value, std_dev_value
        
        return None, None
    
    except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
        print(f"Error reading {file_path}: {e}")
        return None, None
    
def read_csv_discrete(file_path, metric_column, n, target_unit=None):
    """
    Read discrete metric data from CSV files with the new format,
    applying unit conversion if needed.
    
    Args:
        file_path (str): Path to the CSV file.
        metric_column (str): Column name for the metric value.
        n (int): Number of samples.
        target_unit (str or None): Target unit for conversion, if any

    Returns:
        dict: Dictionary with 'mean', 'std', 'mode', 'min', 'max' values
    """
    try:
        df = pd.read_csv(file_path, sep=';')
        
        # Find the first separator row with dashes (------------)
        first_sep_idx = None
        for i, row in df.iterrows():
            first_col = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
            if '------------' in first_col:
                first_sep_idx = i
                break
        
        if first_sep_idx is None:
            print(f"Warning: Could not find first separator row in {file_path}")
            return None
        
        # Find the equals separator row
        equals_sep_idx = None
        for i in range(first_sep_idx + 1, len(df)):
            first_col = str(df.iloc[i, 0]) if not pd.isna(df.iloc[i, 0]) else ""
            if '======' in first_col:
                equals_sep_idx = i
                break
        
        if equals_sep_idx is None:
            print(f"Warning: Could not find equals separator row in {file_path}")
            return None
        
        # Extract values from their expected positions
        result = {
            'mean': None,
            'std': None,
            'mode': None,
            'min': None,
            'max': None
        }
        
        # Mean and std are always after first separator
        mean_idx = first_sep_idx + 1
        std_idx = first_sep_idx + 2
        
        # Mode, below, above are at positions 3, 4, 5 after first separator
        mode_idx = first_sep_idx + 3
        below_idx = first_sep_idx + 4
        above_idx = first_sep_idx + 5
        
        # Min and max are after equals separator
        min_idx = equals_sep_idx + 1
        max_idx = equals_sep_idx + 2
        
        # Extract metric values if column exists
        if metric_column in df.columns:
            # Get mean
            if mean_idx < len(df):
                result['mean'] = pd.to_numeric(df.iloc[mean_idx][metric_column], errors='coerce')
            
            # Get std
            if std_idx < len(df):
                result['std'] = pd.to_numeric(df.iloc[std_idx][metric_column], errors='coerce')
            
            # Get mode
            if mode_idx < len(df):
                mode_val = df.iloc[mode_idx][metric_column]
                if not pd.isna(mode_val) and str(mode_val) != '--':
                    result['mode'] = pd.to_numeric(mode_val, errors='coerce')
            
            # Get min
            if min_idx < len(df):
                min_val = df.iloc[min_idx][metric_column]
                if not pd.isna(min_val) and str(min_val) != '--':
                    result['min'] = pd.to_numeric(min_val, errors='coerce')
            
            # Get max
            if max_idx < len(df):
                max_val = df.iloc[max_idx][metric_column]
                if not pd.isna(max_val) and str(max_val) != '--':
                    result['max'] = pd.to_numeric(max_val, errors='coerce')
            
            # Apply unit conversion if needed
            for key in result:
                if result[key] is not None:
                    result[key] = convert_value_based_on_unit(result[key], metric_column, target_unit)
            
            # If mode not available, use mean
            if result['mode'] is None and result['mean'] is not None:
                result['mode'] = result['mean']
            
            # If min/max not available, estimate from std
            if result['min'] is None and result['mean'] is not None and result['std'] is not None:
                result['min'] = max(0, result['mean'] - 2 * result['std'])
            
            if result['max'] is None and result['mean'] is not None and result['std'] is not None:
                result['max'] = result['mean'] + 2 * result['std']
            
            return result
        else:
            print(f"Warning: Column {metric_column} not found in {file_path}")
            return None
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

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

def get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp=False, filtering=False):
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
    filtered = "_filtered" if filtering else ""
    
    patterns = {
        'pki': f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki{scenario_suffix}{filtered}.csv",
        'pki_client_auth': f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki_client-auth{scenario_suffix}{filtered}.csv",
        'psk': f"udp{rasp_prefix}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_psk{scenario_suffix}{filtered}.csv",
        'nosec': f"udp{rasp_prefix}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}{filtered}.csv",
        'nosec_flexible': f"udp{rasp_prefix}_conv_stats*n{n}*_nosec*{scenario_suffix}{filtered}.csv"
    }
    
    return patterns

def setup_output_dirs(data_dir, custom_suffix=None):
    """
    Setup input and output directories based on custom suffix.
    
    Args:
        custom_suffix (str or None): Optional suffix for data and plot directories.
        
    Returns:
        tuple: (data_dir, plots_dir) directory names.
    """
    if custom_suffix:
        data_dir_ = f"{data_dir}/bench-data-{custom_suffix}"
        plots_dir = f"{data_dir}/plots-{custom_suffix}"
    else:
        data_dir_ = f"{data_dir}/bench-data"
        plots_dir = f"{data_dir}/plots"
    
    return data_dir_, plots_dir

def create_scatter_plot(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None, target_unit=None, filtering=False):
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
        target_unit (str or None): Target unit for conversion if needed.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(data_dir,custom_suffix)
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
    filtered = "_filtered" if filtering else ""

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
        patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp, filtering)
        
        # PSK file pattern
        psk_pattern = patterns['psk']
        psk_files = find_files(data_dir_path, psk_pattern)
        
        if psk_files:
            psk_file_path = psk_files[0]
            # Pass target_unit to read_csv for potential conversion
            metric_value_psk, std_dev_psk = read_csv(psk_file_path, metric, n, target_unit)
            if metric_value_psk is not None:
                data['psk']['values'].append(metric_value_psk)
                data['psk']['std_devs'].append(std_dev_psk)
                data['psk']['algorithms'].append(algorithm)
        else:
            print(f"Warning: Could not find PSK file for algorithm {algorithm}")

    # Get PKI data for each algorithm and certificate type
    for cert_type in cert_types_list:
        for algorithm in algorithms_list:
            # Get file patterns
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp, filtering)
            
            # PKI file pattern
            pki_pattern = patterns['pki']
            pki_files = find_files(data_dir_path, pki_pattern)
            
            # Try with client-auth suffix if needed
            if not pki_files:
                pki_pattern = patterns['pki_client_auth']
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                pki_file_path = pki_files[0]
                # Pass target_unit to read_csv for potential conversion
                metric_value_pki, std_dev_pki = read_csv(pki_file_path, metric, n, target_unit)
                if metric_value_pki is not None:
                    data[cert_type]['values'].append(metric_value_pki)
                    data[cert_type]['std_devs'].append(std_dev_pki)
                    data[cert_type]['algorithms'].append(algorithm)
            else:
                print(f"Warning: Could not find PKI file for algorithm {algorithm} with {cert_type}")
    
    # Get nosec data
    patterns = get_file_patterns("", "", n, s, p, scenario, rasp, filtering)
    nosec_pattern = patterns['nosec']
    nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if not nosec_files:
        # Try a more flexible pattern
        nosec_pattern = patterns['nosec_flexible']
        nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if nosec_files:
        nosec_file_path = nosec_files[0]
        # Pass target_unit to read_csv for potential conversion
        metric_value_horizontal, std_dev_horizontal = read_csv(nosec_file_path, metric, n, target_unit)
        if metric_value_horizontal is not None:
            data['nosec']['value'] = metric_value_horizontal
            data['nosec']['std_dev'] = std_dev_horizontal
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
    # Pass target_unit to format_labels to display correct units
    ax.set_ylabel(format_labels(metric, target_unit))
    title = f'{format_labels(metric, target_unit)} by algorithm and security mode - n={n}, scenario={scenario}'
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
    # Use the display metric name for the output filename
    display_metric = get_display_metric_label(metric, target_unit)
    clean_metric = display_metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/scatter_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}{filtered}.pdf'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

def create_bar_plot(metric, algorithms_list, cert_types_list, n, scenarios, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None, target_unit=None, filtering=False):
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
        target_unit (str or None): Target unit for conversion, (ms for duration or mWh for Energy).
    """
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(data_dir,custom_suffix)
    data_dir_path = os.path.join(script_directory, data_dir_)
    
    print(f"Data directory: {data_dir_path}")
    print(f"Plot directory: {plots_dir}")
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # Generate colors for different certificate types
    cert_colors = get_certificate_colors(cert_types_list)
    
    # Define signature families
    sig_families = {
        'Classical' : ['RSA_2048', 'EC_P256', 'EC_ED25519'],
        'DILITHIUM' : ['DILITHIUM_LEVEL2', 'DILITHIUM_LEVEL3', 'DILITHIUM_LEVEL5'],
        'FALCON' : ['FALCON_LEVEL1', 'FALCON_LEVEL5']
    }
    
    family_patterns = {
        'Classical' : '', 
        'DILITHIUM' : 'x',
        'FALCON' : '.'
    }
    
    # Set pattern for each cert type
    def get_pattern(cert_type):
        for family, certs in sig_families.items():
            if cert_type in certs:
                return family_patterns[family]
        return ''

    # Common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    rasp_prefix = "rasp" if rasp else ""
    filtered = "_filtered" if filtering else ""
    
    # Data collection with proper tracking
    bar_data = []  # List to store all bar data
    x_labels = []  # Labels for x-axis
    
    # Process each algorithm and scenario combination
    for alg_idx, algorithm in enumerate(algorithms_list):
        for scen_idx, scenario in enumerate(scenarios):
            scenario_suffix = f"_scenario{scenario}"
            
            # Group bars for this algorithm-scenario combo
            group_bars = []
            
            # Process each certificate type (PKI)
            for cert_idx, cert_type in enumerate(cert_types_list):
                patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp, filtering)
                
                # Try PKI patterns
                pki_pattern = patterns['pki']
                pki_files = find_files(data_dir_path, pki_pattern)
                
                if not pki_files:
                    pki_pattern = patterns['pki_client_auth']
                    pki_files = find_files(data_dir_path, pki_pattern)
                
                if pki_files:
                    file_path = pki_files[0]
                    # Pass target_unit to read_csv for potential conversion
                    metric_value, std_dev_value = read_csv(file_path, metric, n, target_unit)
                    
                    if metric_value is not None and std_dev_value is not None:
                        group_bars.append({
                            'value': metric_value,
                            'std': std_dev_value,
                            'color': cert_colors[cert_type],
                            'pattern': get_pattern(cert_type),
                            'label': f'PKI ({cert_type})',
                            'type': 'pki',
                            'cert_type': cert_type
                        })
                        print(f"Found data for {algorithm} {cert_type} {scenario}: {metric_value}")
            
            # Process PSK
            patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp, filtering)
            psk_pattern = patterns['psk']
            psk_files = find_files(data_dir_path, psk_pattern)
            
            if psk_files:
                file_path = psk_files[0]
                # Pass target_unit to read_csv for potential conversion
                metric_value, std_dev_value = read_csv(file_path, metric, n, target_unit)
                
                if metric_value is not None and std_dev_value is not None:
                    group_bars.append({
                        'value': metric_value,
                        'std': std_dev_value,
                        'color': security_mode_colors['psk'],
                        'pattern': '',
                        'label': 'PSK',
                        'type': 'psk',
                        'cert_type': None
                    })
                    print(f"Found data for {algorithm} PSK {scenario}: {metric_value}")
            
            # Only add the group if it has data
            if group_bars:
                bar_data.append({
                    'algorithm': algorithm,
                    'scenario': scenario,
                    'bars': group_bars,
                    'label': f"{algorithm}-{scenario}"
                })
    
    # Process NoSec for each scenario
    for scen_idx, scenario in enumerate(scenarios):
        scenario_suffix = f"_scenario{scenario}"
        
        patterns = get_file_patterns("", "", n, s, p, scenario, rasp, filtering)
        nosec_pattern = patterns['nosec']
        nosec_files = find_files(data_dir_path, nosec_pattern)
        
        if not nosec_files:
            nosec_pattern = patterns['nosec_flexible']
            nosec_files = find_files(data_dir_path, nosec_pattern)
        
        if nosec_files:
            file_path = nosec_files[0]
            # Pass target_unit to read_csv for potential conversion
            metric_value, std_dev_value = read_csv(file_path, metric, n, target_unit)
            
            if metric_value is not None and std_dev_value is not None:
                bar_data.append({
                    'algorithm': 'NoSec',
                    'scenario': scenario,
                    'bars': [{
                        'value': metric_value,
                        'std': std_dev_value,
                        'color': security_mode_colors['nosec'],
                        'pattern': '',
                        'label': 'NoSec',
                        'type': 'nosec',
                        'cert_type': None
                    }],
                    'label': f"NoSec-{scenario}"
                })
                print(f"Found data for NoSec {scenario}: {metric_value}")
    
    # Now plot all bars with proper positioning
    bar_width = 0.8
    group_spacing = 0.5
    current_pos = 0
    xtick_positions = []
    xtick_labels = []
    
    # Track all unique labels for legend
    legend_entries = {}
    pattern_entries = {}
    
    for group_idx, group in enumerate(bar_data):
        num_bars = len(group['bars'])
        
        # Calculate positions for this group
        bar_positions = []
        for i in range(num_bars):
            pos = current_pos + i * bar_width
            bar_positions.append(pos)
        
        # Plot bars in this group
        for bar_idx, bar in enumerate(group['bars']):
            pos = bar_positions[bar_idx]
            
            # Plot the bar
            rect = ax.bar(pos, bar['value'], width=bar_width, 
              color=bar['color'], edgecolor='black', linewidth=1.5,
              hatch=bar['pattern'])
            
            if bar['pattern']:
                rect[0].set_path_effects([path_effects.withStroke(linewidth=1, foreground='white')])
            
            # Add error bar
            ax.errorbar(pos, bar['value'], yerr=bar['std'], 
                        capsize=5, fmt='none', color='black', linewidth=1.5)
            
            # Track legend entry
            if bar['label'] not in legend_entries:
                legend_entries[bar['label']] = bar['color']
                
            # Track pattern entries for certificate families
            if bar['cert_type']:
                for family, certs in sig_families.items():
                    if bar['cert_type'] in certs:
                        pattern_entries[family] = family_patterns[family]
        
        # Set x-tick at center of group
        group_center = current_pos + (num_bars - 1) * bar_width / 2
        xtick_positions.append(group_center)
        xtick_labels.append(group['label'])
        
        # Move to next group position
        current_pos += num_bars * bar_width + group_spacing
    
    # Set x-ticks and labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=10)
    
    # Set axis labels and title
    # Pass target_unit to format_labels to display correct units
    ax.set_ylabel(format_labels(metric, target_unit))
    ax.set_xlabel(r'Algorithm - Scenario - Mode')
    
    # Format title
    title = f'{format_labels(metric, target_unit)} by algorithm and security mode - n={n}, scenario={scenario}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', p={p}'
    ax.set_title(title)
    
    # Create custom legend
    # First part: security mode legend
    color_handles = [Patch(facecolor=color, label=label, edgecolor='black') 
                    for label, color in legend_entries.items()]
    
    # Second part: pattern legend for certificate families
    pattern_handles = []
    if pattern_entries:
        pattern_handles.append(Line2D([0], [0], color='none', label='Certificate Families:'))
        for family, pattern in pattern_entries.items():
            pattern_handles.append(Patch(facecolor='lightgray', edgecolor='black', 
                                       hatch=pattern, label=f'{family}'))
    
    # Combine handles
    all_handles = color_handles
    if pattern_handles:
        all_handles.append(Line2D([0], [0], color='none', label=''))  # Separator
        all_handles.extend(pattern_handles)
    
    ax.legend(handles=all_handles, loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Generate the output filename
    # Use the display metric name for the output filename
    display_metric = get_display_metric_label(metric, target_unit)
    clean_metric = display_metric.replace(' ', '_').replace('(', '').replace(')', '')
    scenarios_str = "".join(scenarios)
    
    # Create the plots directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/barplot_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}_{scenarios_str}{filtered}.pdf'

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()
    
def create_heat_map(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None, target_unit=None, filtering=False):
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
        target_unit (str or None): Target unit for conversion, if any.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(data_dir,custom_suffix)
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
    filtered = "_filtered" if filtering else ""
    
    # Variables to track min and max values for color scaling
    valid_data_found = False
    min_value = float('inf')
    max_value = float('-inf')
    
    # Collect data for the heat map
    for alg_idx, algorithm in enumerate(algorithms_list):
        for cert_idx, cert_type in enumerate(cert_types_list):
            # Get file patterns
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp, filtering)
            
            # PKI file pattern
            pki_pattern = patterns['pki']
            pki_files = find_files(data_dir_path, pki_pattern)
            
            # Try with client-auth if needed
            if not pki_files:
                pki_pattern = patterns['pki_client_auth']
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                file_path = pki_files[0]
                # Pass target_unit to read_csv for potential conversion
                metric_value, _ = read_csv(file_path, metric, n, target_unit)
                
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
    # Use display metric name for colorbar label
    display_metric = get_display_metric_label(metric, target_unit)
    cbar.set_label(display_metric)
    
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
    
    # LaTeX-formatted labels
    ax.set_xlabel(r'Certificate Types')
    ax.set_ylabel(r'Algorithms')
    
    # Format title - use format_labels to get proper unit display
    title = f'Heat Map of {format_labels(metric, target_unit)} - n={n}, scenario={scenario}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', p={p}'
    ax.set_title(title)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    # Use the display metric name for the output filename
    display_metric = get_display_metric_label(metric, target_unit)
    clean_metric = display_metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/heatmap_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}.pdf'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()
    
def create_box_plot(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None, target_unit=None, log_scale=True, filtering=False):
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
        target_unit (str or None): Target unit for conversion, if any.
        log_scale (bool): Whether to use logarithmic scale for y-axis.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(data_dir,custom_suffix)
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
    filtered = "_filtered" if filtering else ""
    
    # Collect raw data for each configuration
    for algorithm in algorithms_list:
        for cert_type in cert_types_list:
            # Get file patterns
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp, filtering)
            
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
                        #print(f"DEBUG: data_rows shape: {data_rows.shape}")
                        #print(f"DEBUG: First 10 rows of data_rows:")
                        #print(data_rows.head(10))
                        # Extract the metric column
                        if metric in data_rows.columns:
                            # Filter out non-numeric values (if any)
                            #print(f"DEBUG: Raw {metric} column values:")
                            metric_column = data_rows[metric]
                            #print(f"DEBUG: Total values in column: {len(metric_column)}")
                            #print(f"DEBUG: Non-null values: {metric_column.notna().sum()}")
                            #print(f"DEBUG: Sample values: {metric_column.dropna().head(10).tolist()}")
                            
                            raw_values = pd.to_numeric(data_rows[metric], errors='coerce').dropna().values
                            #print(f"DEBUG: Final raw_values length: {len(raw_values)}")
                            #print(f"DEBUG: Final raw_values sample: {raw_values[:10]}")
                            
                            # Apply unit conversion if needed
                            if target_unit:
                                raw_values = np.array([convert_value_based_on_unit(val, metric, target_unit) 
                                                       for val in raw_values])
                            
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
        patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp, filtering)
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
                        
                        # Apply unit conversion if needed
                        if target_unit:
                            raw_values = np.array([convert_value_based_on_unit(val, metric, target_unit) 
                                                  for val in raw_values])
                        
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
    patterns = get_file_patterns("", "", n, s, p, scenario, rasp, filtering)
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
                    
                    # Apply unit conversion if needed
                    if target_unit:
                        raw_values = np.array([convert_value_based_on_unit(val, metric, target_unit) 
                                              for val in raw_values])
                    
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
    
    # Group data by algorithm for better organization
    algorithm_groups = {}
    nosec_data = None

    # First, organize data by algorithm
    for key, data in raw_data_dict.items():
        if len(data['values']) > 0:
            if data['algorithm'] == 'NoSec':
                nosec_data = data
            else:
                if data['algorithm'] not in algorithm_groups:
                    algorithm_groups[data['algorithm']] = []
                algorithm_groups[data['algorithm']].append(data)

    # Prepare box plot data with consistent spacing
    box_data = []
    box_colors = []
    positions = []
    tick_positions = []
    tick_labels = []
    group_boundaries = []
    
    # Define spacing parameters - use consistent values
    box_width = 1.0
    box_gap = 0.5  # Gap between boxes in the same group
    group_spacing = 1.0  # Space between algorithm groups
    
    current_pos = 0  # Start position
    
    # Process each algorithm in the specified order
    for alg_idx, algorithm in enumerate(algorithms_list):
        if algorithm in algorithm_groups:
            # Get configurations for this algorithm
            alg_configs = algorithm_groups[algorithm]
            
            # Separate PKI and PSK configs
            pki_configs = [cfg for cfg in alg_configs if cfg['cert_type'] not in ('PSK', 'NoSec')]
            psk_configs = [cfg for cfg in alg_configs if cfg['cert_type'] == 'PSK']
            
            # Sort PKI configs by certificate type according to cert_types_list order
            if pki_configs:
                cert_order = {cert: idx for idx, cert in enumerate(cert_types_list)}
                pki_configs.sort(key=lambda x: cert_order.get(x['cert_type'], 999))
            
            # Place PSK at the end of each algorithm group
            alg_configs = pki_configs + psk_configs
            
            # Record group start position for label placement
            group_start = current_pos + box_gap
            
            # Add data for each configuration in this algorithm group
            for config in alg_configs:
                box_data.append(config['values'])
                box_colors.append(config['color'])
                positions.append(current_pos)
                current_pos += box_width + box_gap
            
            # Remove the extra gap after the last box in the group
            #current_pos -= box_gap
            
            # Set the label position at the center of this group
            tick_positions.append((group_start + current_pos) / 2)
            tick_labels.append(algorithm.replace("_", " "))
            
            # Add group boundary if not the last algorithm
            if alg_idx < len(algorithms_list) - 1:
                # Add spacing after the group
                current_pos += group_spacing/2
                
                # Place boundary line in the middle of the gap
                boundary_pos = current_pos - (group_spacing)
                group_boundaries.append(boundary_pos)
    
    # Add NoSec at the end if present
    if nosec_data is not None:
        # Add a wider spacing before NoSec
        nosec_spacing = group_spacing * 1.2  # 20% wider for emphasis
        
        # Add the boundary before NoSec
        boundary_pos = current_pos #+ (nosec_spacing/2)
        group_boundaries.append(boundary_pos)
        
        # Position for NoSec bar
        current_pos += nosec_spacing * 1.5
        
        # Add NoSec data
        box_data.append(nosec_data['values'])
        box_colors.append(nosec_data['color'])
        positions.append(current_pos)
        
        # Add tick for NoSec
        tick_positions.append(current_pos)
        tick_labels.append("NoSec")
        
        # Move past NoSec for proper plot limits
        current_pos += box_width
    
    # Create box plot with precisely calculated positions
    bp = ax.boxplot(box_data, positions=positions, 
                    patch_artist=True, 
                    showfliers=False,
                    medianprops={'color': 'black'}, 
                    widths=box_width*0.9, # Make boxes slightly narrower
    )
    # Set colors for each box
    #for patch, color in zip(bp['boxes'], box_colors):
    #    patch.set_facecolor(color)
    #    patch.set_alpha(0.7)
    
    # Make boxes transparent
    for patch in bp['boxes']:
        patch.set_facecolor('none')
        patch.set_edgecolor('black')
        
    # Plot raw data points with jitter
    for pos, data, color in zip(positions, box_data, box_colors):
        jitter = np.random.normal(0, box_width * 0.1, size=len(data))
        ax.scatter(pos + jitter, data, color=color, alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
        
    # Add vertical dashed lines between algorithm groups
    for boundary in group_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1.0, zorder=0)  # Put lines behind boxes

    # Set custom tick positions and labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, ha='center')

    # Set appropriate x-axis limits with some padding
    ax.set_xlim(-box_width, current_pos + box_width)
    
    if log_scale:
        ax.set_yscale('log')
    
    # Pass target_unit to format_labels to display correct units
    ax.set_ylabel(format_labels(metric, target_unit))
    title = f'Variability of {format_labels(metric, target_unit)} across configurations - n={n}, scenario={scenario}'
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
    # Use the display metric name for the output filename
    display_metric = get_display_metric_label(metric, target_unit)
    clean_metric = display_metric.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/boxplot_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}{filtered}.pdf'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Analyze and save outliers  
    outlier_results = analyze_and_save_outliers(
        box_data, 
        metric, 
        scenario, 
        plots_dir,
        n=n
    )
    
    plt.show()
    
def analyze_and_save_outliers(box_data, metric_name, scenario, plots_dir, n=None, multiplier=5.0):
    """
    Simple outlier analysis using magnitude-based detection.
    
    Outliers are defined as values that are more than `multiplier` times 
    larger than the minimum value in the dataset.
    
    Parameters:
    -----------
    box_data : list of arrays
        Data for each configuration 
    metric_name : str  
        Name of the metric (for filename)
    scenario : str
        Scenario identifier 
    plots_dir : str
        Directory to save the CSV file
    n : int or None
        Number of iterations (for documentation)
    multiplier : float
        Multiplier for outlier detection (default: 5.0)
    """
    
    # Extract only mean values from box_data (every other value starting from 0)
    mean_data = []
    for data in box_data:
        data_array = np.array(data)
        # Take every other value starting from index 0 (the means)
        means_only = data_array[::2]  # This gives you indices 0, 2, 4, 6, ... (the means)
        mean_data.append(means_only)
    
    # Analyze outliers for each configuration
    outlier_summary = []
    total_measurements = 0
    total_outliers = 0
    
    for i, data in enumerate(mean_data):
        data = np.array(data)
        print(f"\nConfig {i}:")
        print(f"  Data length: {len(data)}")
        print(f"  Dataset (means): {data}")
        print(f"  Data range: {np.min(data):.6f} to {np.max(data):.6f}")
        
        # Magnitude-based outlier detection
        min_value = np.min(data)
        threshold = min_value * multiplier
        outliers = data[data > threshold]
        
        n_total = len(data)
        n_outliers = len(outliers)
        outlier_percentage = (n_outliers / n_total) * 100 if n_total > 0 else 0
        
        total_measurements += n_total
        total_outliers += n_outliers
        
        outlier_info = {
            'config_index': i,
            'config_name': f'Config_{i+1}',
            'n_total': n_total,
            'n_outliers': n_outliers,
            'outlier_percentage': round(outlier_percentage, 1),
            'outlier_values': outliers.tolist() if n_outliers > 0 else [],
            'min_value': round(min_value, 6),
            'threshold': round(threshold, 6),
            'median': round(np.median(data), 6),
            'mean': round(np.mean(data), 6),
            'max_value': round(np.max(data), 6),
            'detection_method': f'magnitude (>{multiplier}x min)',
        }
        
        outlier_summary.append(outlier_info)
    
    # Print summary to screen
    print(f"\n{'='*70}")
    print(f"OUTLIER ANALYSIS - {metric_name} (Scenario {scenario})")
    print(f"{'='*70}")
    print(f"Total configurations: {len(mean_data)}")
    print(f"Total measurements: {total_measurements}")
    print(f"Total outliers: {total_outliers} ({total_outliers/total_measurements*100:.1f}%)")
    if n:
        print(f"Iterations per configuration: {n}")
    print(f"Detection method: Magnitude-based (values > {multiplier}x minimum)")
    print(f"\nConfiguration Details:")
    print("-" * 70)
    
    for info in outlier_summary:
        print(f"{info['config_name']} (index {info['config_index']}):")
        print(f"  Measurements: {info['n_total']}")
        print(f"  Range: {info['min_value']:.6f} to {info['max_value']:.6f}")
        print(f"  Threshold: {info['threshold']:.6f} (>{multiplier}x min)")
        print(f"  Outliers: {info['n_outliers']} ({info['outlier_percentage']}%)")
        print(f"  Central tendency: median={info['median']:.6f}, mean={info['mean']:.6f}")
        
        if info['n_outliers'] > 0:
            outlier_values = info['outlier_values']
            if len(outlier_values) <= 5:
                print(f"  Outlier values: {[round(x, 6) for x in outlier_values]}")
            else:
                print(f"  Outlier values: {[round(x, 6) for x in outlier_values[:3]]} ... {[round(x, 6) for x in outlier_values[-2:]]}")
        print()
    
    # Create a flattened version for CSV
    csv_data = []
    for info in outlier_summary:
        row = {
            'config_index': info['config_index'],
            'config_name': info['config_name'],
            'n_total': info['n_total'],
            'n_outliers': info['n_outliers'], 
            'outlier_percentage': info['outlier_percentage'],
            'min_value': info['min_value'],
            'threshold': info['threshold'],
            'median': info['median'],
            'mean': info['mean'],
            'max_value': info['max_value'],
            'detection_method': info['detection_method']
        }
        
        # Add outlier values as separate columns
        outlier_values = info['outlier_values']
        for j, outlier_val in enumerate(outlier_values):
            row[f'outlier_{j+1}'] = round(outlier_val, 6)
        
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    
    # Generate filename
    clean_metric = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"outliers_{clean_metric}_scenario{scenario}_magnitude{multiplier}x.csv"
    filepath = f"./{plots_dir}/{filename}"
    
    # Ensure directory exists
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    # Save CSV
    df.to_csv(filepath, index=False)
    print(f"Outlier data saved to: {filepath}")
    
    # Summary file
    summary_filename = f"outliers_summary_{clean_metric}_scenario{scenario}_magnitude{multiplier}x.txt"
    summary_filepath = f"./{plots_dir}/{summary_filename}"
    
    with open(summary_filepath, 'w') as f:
        f.write(f"OUTLIER ANALYSIS SUMMARY\n")
        f.write(f"Metric: {metric_name}\n")
        f.write(f"Scenario: {scenario}\n")
        f.write(f"Total configurations: {len(mean_data)}\n")
        f.write(f"Total measurements: {total_measurements}\n")
        f.write(f"Total outliers: {total_outliers} ({total_outliers/total_measurements*100:.1f}%)\n")
        if n:
            f.write(f"Iterations per configuration: {n}\n")
        f.write(f"Detection method: Magnitude-based (values > {multiplier}x minimum)\n")
        f.write(f"Multiplier used: {multiplier}\n\n")
        
        f.write("Config_Index,Config_Name,Total,Outliers,Percentage,Min_Value,Threshold,Median,Mean\n")
        for info in outlier_summary:
            f.write(f"{info['config_index']},{info['config_name']},{info['n_total']},{info['n_outliers']},{info['outlier_percentage']}%,{info['min_value']:.6f},{info['threshold']:.6f},{info['median']:.6f},{info['mean']:.6f}\n")
    
    print(f"Summary saved to: {summary_filepath}")
    
    return outlier_summary
    
def create_discrete_candlestick_plot(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None, target_unit=None, filtering=False):
    """
    Create candlestick-style plots for discrete metrics showing min-max range with mode.
    Improved version with:
    - Fixed legend (no empty slot)
    - Different line styles for different certificate families
    
    Args:
        metric (str): The discrete metric to be plotted.
        algorithms_list (list): List of algorithm names.
        cert_types_list (list): List of certificate types to include.
        n (int): Number of repetitions.
        scenario (str): Scenario identifier (A, B, or C).
        rasp (bool): Whether the data is from a Raspberry Pi.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter (parallelization mode).
        data_dir (str): Directory containing the data files.
        custom_suffix (str): Optional suffix for data and plot directories.
        target_unit (str or None): Target unit for conversion, if any.
    """
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Get data and plots directories
    data_dir_, plots_dir = setup_output_dirs(data_dir,custom_suffix)
    data_dir_path = os.path.join(script_directory, data_dir_)
    
    print(f"Data directory: {data_dir_path}")
    print(f"Plot directory: {plots_dir}")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Generate colors for different certificate types
    cert_colors = get_certificate_colors(cert_types_list)
    
    # Define signature families and line styles
    sig_families = {
        'Classical': ['RSA_2048', 'EC_P256', 'EC_ED25519'],
        'DILITHIUM': ['DILITHIUM_LEVEL2', 'DILITHIUM_LEVEL3', 'DILITHIUM_LEVEL5'],
        'FALCON': ['FALCON_LEVEL1', 'FALCON_LEVEL5']
    }
    
    family_line_styles = {
        'Classical': '-',
        'DILITHIUM': '--',
        'FALCON': '-.'
    }
    
    # Helper function to get the line style for a certificate
    def get_line_style(cert_type):
        for family, certs in sig_families.items():
            if cert_type in certs:
                return family_line_styles[family]
        return '-'  # Default solid line
    
    # Construct the common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    rasp_prefix = "rasp" if rasp else ""
    filtered = "_filtering" if filtering else ""
    
    # Calculate positions for candlesticks with consistent ordering
    num_algorithms = len(algorithms_list)
    num_cert_types = len(cert_types_list)
    
    # Total items per algorithm group: cert_types + PSK
    items_per_algo = num_cert_types + 1
    
    # Calculate spacing
    item_spacing = 2.0  # Space between individual candlesticks
    group_spacing = 0.5  # Space between algorithm groups
    
    # Store data for plotting with consistent ordering
    plot_data = []
    all_values = []
    
    # Process data in the exact order specified by cert_types_list
    for alg_idx, algorithm in enumerate(algorithms_list):
        base_position = alg_idx * (items_per_algo * item_spacing + group_spacing)
        
        # First: PKI certificates in the order they were specified
        for cert_idx, cert_type in enumerate(cert_types_list):
            patterns = get_file_patterns(algorithm, cert_type, n, s, p, scenario, rasp, filtering)
            pki_pattern = patterns['pki']
            pki_files = find_files(data_dir_path, pki_pattern)
            
            if not pki_files:
                pki_pattern = patterns['pki_client_auth']
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                file_path = pki_files[0]
                # Pass target_unit for unit conversion
                data = read_csv_discrete(file_path, metric, n, target_unit)
                
                if data and data.get('mode') is not None:
                    position = base_position + cert_idx * item_spacing
                    plot_data.append({
                        'pos': position,
                        'data': data,
                        'label': f'{algorithm} {cert_type}',
                        'algorithm': algorithm,
                        'cert_type': cert_type,
                        'color': cert_colors[cert_type],
                        'line_style': get_line_style(cert_type)
                    })
                    all_values.extend([data.get('min', 0), data.get('max', 0), data.get('mode', 0)])
        
        # Then: PSK for this algorithm
        patterns = get_file_patterns(algorithm, "", n, s, p, scenario, rasp, filtering)
        psk_pattern = patterns['psk']
        psk_files = find_files(data_dir_path, psk_pattern)
        
        if psk_files:
            file_path = psk_files[0]
            # Pass target_unit for unit conversion
            data = read_csv_discrete(file_path, metric, n, target_unit)
            
            if data and data.get('mode') is not None:
                position = base_position + num_cert_types * item_spacing
                plot_data.append({
                    'pos': position,
                    'data': data,
                    'label': f'{algorithm} PSK',
                    'algorithm': algorithm,
                    'cert_type': 'PSK',
                    'color': security_mode_colors['psk'],
                    'line_style': '-'  # Default solid line for PSK
                })
                all_values.extend([data.get('min', 0), data.get('max', 0), data.get('mode', 0)])
    
    # Finally: NoSec (positioned after all algorithm groups)
    patterns = get_file_patterns("", "", n, s, p, scenario, rasp, filtering)
    nosec_pattern = patterns['nosec']
    nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if not nosec_files:
        nosec_pattern = patterns['nosec_flexible']
        nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if nosec_files:
        file_path = nosec_files[0]
        # Pass target_unit for unit conversion
        data = read_csv_discrete(file_path, metric, n, target_unit)
        
        if data and data.get('mode') is not None:
            # Position NoSec with extra spacing from the last group
            last_algo_end = (num_algorithms - 1) * (items_per_algo * item_spacing + group_spacing) + items_per_algo * item_spacing
            position = last_algo_end + group_spacing
            plot_data.append({
                'pos': position,
                'data': data,
                'label': 'NoSec',
                'algorithm': 'NoSec',
                'cert_type': 'NoSec',
                'color': security_mode_colors['nosec'],
                'line_style': '-'  # Default solid line for NoSec
            })
            all_values.extend([data.get('min', 0), data.get('max', 0), data.get('mode', 0)])
    
    # Create candlestick plot with fat dots for modes
    for item in plot_data:
        data = item['data']
        pos = item['pos']
        color = item['color']
        line_style = item['line_style']
        
        mode = data.get('mode', data.get('mean'))
        if mode is None:
            continue
            
        min_val = data.get('min', mode)
        max_val = data.get('max', mode)
        
        # Increase line width significantly
        line_width = item_spacing * 0.8  # Much wider horizontal lines
        
        # Always draw vertical line if there's a range
        if min_val != max_val:
            # Draw vertical line with increased width and appropriate line style
            ax.plot([pos, pos], [min_val, max_val], 
                    color=color, linewidth=2, linestyle=line_style)
            
            # Draw horizontal lines with increased width and length
            ax.plot([pos - line_width/2, pos + line_width/2], [min_val, min_val], 
                    color=color, linewidth=2, linestyle='-', solid_capstyle='round')
            ax.plot([pos - line_width/2, pos + line_width/2], [max_val, max_val], 
                    color=color, linewidth=2, linestyle='-', solid_capstyle='round')
        else:
            # If min=max=mode, draw a single horizontal line
            ax.plot([pos - line_width/2, pos + line_width/2], [mode, mode], 
                    color=color, linewidth=2, linestyle='-', solid_capstyle='round')
        
        # Draw the mode point LAST so it's on top
        ax.plot(pos, mode, 'o', color='gray', markersize=8, markeredgecolor='gray', 
                markeredgewidth=1.0, zorder=10)  # Higher zorder puts it on top
    
    # Set x-axis labels at algorithm centers
    algorithm_positions = {}
    algorithm_groups = {}
    
    # Group positions by algorithm
    for item in plot_data:
        alg = item['algorithm']
        if alg != 'NoSec':
            if alg not in algorithm_groups:
                algorithm_groups[alg] = []
            algorithm_groups[alg].append(item['pos'])
    
    # Calculate centers for each algorithm
    for alg, positions in algorithm_groups.items():
        if positions:
            algorithm_positions[alg] = sum(positions) / len(positions)
    
    # Add NoSec separately
    nosec_items = [item for item in plot_data if item['algorithm'] == 'NoSec']
    if nosec_items:
        algorithm_positions['NoSec'] = nosec_items[0]['pos']
    
    # Set ticks at algorithm centers
    ax.set_xticks(list(algorithm_positions.values()))
    ax.set_xticklabels(list(algorithm_positions.keys()), rotation=0, ha='center', fontsize=11)
    
    # Add correct vertical dashed lines between algorithm groups
    if len(algorithm_groups) > 1:
        sorted_algs = sorted(algorithm_groups.keys(), key=lambda x: min(algorithm_groups[x]))
        for i in range(len(sorted_algs) - 1):
            curr_positions = algorithm_groups[sorted_algs[i]]
            next_positions = algorithm_groups[sorted_algs[i+1]]
            
            # Find the gap between groups
            max_curr = max(curr_positions)
            min_next = min(next_positions)
            separator_pos = (max_curr + min_next) / 2
            
            ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
    
    # Add separator before NoSec
    if nosec_items and algorithm_groups:
        last_positions = []
        for positions in algorithm_groups.values():
            last_positions.extend(positions)
        max_algo_pos = max(last_positions)
        nosec_pos = nosec_items[0]['pos']
        separator_pos = (max_algo_pos + nosec_pos) / 2
        ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
        
    # Set y-axis limits with proper padding
    if all_values:
        non_none_values = [v for v in all_values if v is not None]
        if non_none_values:
            y_min = min(non_none_values)
            y_max = max(non_none_values)
            padding = (y_max - y_min) * 0.1 if y_max > y_min else y_max * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
    
    # Set labels and title
    # Pass target_unit to format_labels for proper unit display
    ax.set_ylabel(format_labels(metric, target_unit))
    ax.set_xlabel(r'Algorithm / Configuration')
    ax.margins(x=0.02)
    title = f'{format_labels(metric, target_unit)} - n={n}, scenario={scenario}'
    if s is not None:
        title += f', s={s}'
    if p is not None:
        title += f', {p}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Create legend with proper positioning
    # Create dict for certificate colors
    cert_legend_items = {}
    for cert_type in cert_types_list:
        cert_legend_items[f"PKI ({cert_type})"] = cert_colors[cert_type]
    
    # Add PSK and NoSec
    cert_legend_items['PSK'] = security_mode_colors['psk'] 
    cert_legend_items['NoSec'] = security_mode_colors['nosec']
    
    # Create line style legend items
    line_style_legend_items = {}
    for family, style in family_line_styles.items():
        line_style_legend_items[family] = style
    
    # Create handles for certificate types (colored patches)
    cert_handles = [Patch(facecolor=color, label=label) for label, color in cert_legend_items.items()]
    
    # Create handles for family line styles
    line_style_handles = []
    for family, style in family_line_styles.items():
        line_style_handles.append(Line2D([0], [0], color='black', linestyle=style, label=family))
    
    # Add the mode point to legend
    mode_handle = Line2D([0], [0], marker='o', color='gray', label='Mode', 
                         markersize=10, markeredgecolor='gray', markeredgewidth=1, linestyle='')
    
    # Combine all handles
    all_handles = cert_handles + [Line2D([0], [0], color='none', label='')] + line_style_handles + [mode_handle]
    
    # Create legend
    ax.legend(handles=all_handles, loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_axisbelow(True)
    
    # Save the plot
    # Use the display metric name for the output filename
    display_metric = get_display_metric_label(metric, target_unit)
    clean_metric = display_metric.replace(' ', '_').replace('(', '').replace(')', '')
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    output_file = f'./{plots_dir}/candlestick_{rasp_prefix}_{clean_metric}_n{n}{s_suffix}{p_suffix}{scenario_suffix}{filtered}.pdf'
    
    plt.tight_layout(rect=[0, 0.02, 0.95, 0.98])
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()
    
def parse_args():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description='CoAP Benchmark Plotting Tool',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Default values
    default_algorithms = "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5"
    default_cert_types = "RSA_2048,EC_P256,EC_ED25519,DILITHIUM_LEVEL2,DILITHIUM_LEVEL3,DILITHIUM_LEVEL5,FALCON_LEVEL1,FALCON_LEVEL5"
    
    # Required arguments - now with defaults
    parser.add_argument('metric', help='Metric to be plotted (e.g., "duration", "duration ms", "Energy (Wh)", "Energy (mWh)")')
    parser.add_argument('--algorithms', default=default_algorithms, 
                        help=f'Comma-separated list of algorithms (default: "{default_algorithms}")')
    parser.add_argument('--cert-types', dest='cert_types', default=default_cert_types,
                        help=f'Comma-separated list of certificate types (default: "{default_cert_types}")')
    parser.add_argument('n', type=int, help='Number of repetitions')
    
    # Plot type selection
    plot_type = parser.add_mutually_exclusive_group(required=True)
    plot_type.add_argument('--scatter', action='store_true', help='Generate scatter plot')
    plot_type.add_argument('--barplot', action='store_true', help='Generate bar plot')
    plot_type.add_argument('--heatmap', action='store_true', help='Generate heat map')
    plot_type.add_argument('--boxplot', action='store_true', help='Generate box plot for variability analysis')
    plot_type.add_argument('--candlestick', action='store_true', help='Generate candlestick plot for discrete metrics')
    
    # Scenarios
    parser.add_argument('--scenarios', default='A', 
                        help='Comma-separated list of scenarios (e.g., "A,B,C"). For scatter, heatmap, boxplot, and candlestick plots, only the first scenario is used.')
    
    # Optional arguments
    parser.add_argument('--rasp', action='store_true', help='Use Raspberry Pi dataset')
    parser.add_argument('--s', type=int, help='Optional s parameter')
    parser.add_argument('--p', help='Optional p parameter (parallelization mode)')
    parser.add_argument('--filtered', action='store_true', help='Use filtered dataset (reduced outliers)')
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
    
    # Parse the metric and target unit
    base_metric, target_unit = parse_metric_with_unit(args.metric)
    
    # Setup matplotlib style with LaTeX unless disabled
    setup_matplotlib_style(use_latex=True)
    
    # Use only the first scenario for plots that don't support multiple scenarios
    scenario = scenarios[0]
    
    if args.scatter:
        print(f"Generating scatter plot for scenario {scenario}...")
        create_scatter_plot(
            base_metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix,
            target_unit=target_unit, filtering=args.filtered
        )
    elif args.heatmap:
        print(f"Generating heat map for scenario {scenario}...")
        create_heat_map(
            base_metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix,
            target_unit=target_unit, filtering=args.filtered
        )
    elif args.boxplot:
        print(f"Generating box plot for scenario {scenario}...")
        create_box_plot(
            base_metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix,
            target_unit=target_unit, log_scale=True, filtering=args.filtered
        )
    elif args.candlestick:
        print(f"Generating candlestick plot for scenario {scenario}...")
        create_discrete_candlestick_plot(
            base_metric, algorithms, cert_types, args.n, scenario,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix,
            target_unit=target_unit, filtering=args.filtered
        )
    else:
        # Bar plot supports multiple scenarios
        print(f"Generating bar plot for scenarios {', '.join(scenarios)}...")
        create_bar_plot(
            base_metric, algorithms, cert_types, args.n, scenarios,
            args.rasp, args.s, args.p, args.data_dir, args.custom_suffix,
            target_unit=target_unit, filtering=args.filtered
        )

if __name__ == "__main__":
    main()
    