import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
import glob

# Generate the color palette
color_palette = list(mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(np.linspace(0, 1, 9)))
security_mode_colors = {
    "pki": color_palette[0],
    "psk": color_palette[3],
    "nosec": color_palette[6]
}

def read_csv(file_path, metric_column, rasp=False):    
    """
    Read CSV file and extract the metric value and standard deviation.
    
    Args:
        file_path (str): Path to the CSV file.
        metric_column (str): Name of the column to extract.
        rasp (bool): Whether the data is from a Raspberry Pi.
        
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
    """Find files matching a pattern in the base directory"""
    try:
        matching_files = glob.glob(os.path.join(base_dir, pattern))
        return matching_files
    except Exception as e:
        print(f"Error searching for files: {e}")
        return []

def create_scatter_plots(metric, algorithms_list, cert_types_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data', custom_suffix=None):
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
    
    # Handle custom directory naming
    if custom_suffix:
        data_dir = f"bench-data-{custom_suffix}"
        plots_dir = f"bench-plots-{custom_suffix}"
    else:
        plots_dir = "bench-plots"
    
    data_dir_path = os.path.join(script_directory, data_dir)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set colors for different modes
    pki_base_color = color_palette[0]
    psk_color = color_palette[1]
    nosec_color = color_palette[2]
    
    # Generate additional colors for different certificate types
    cert_colors = {}
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

    # Construct the common file pattern parts
    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    rasp_prefix = "_rasp" if rasp else ""

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
        # PSK file pattern
        psk_pattern = f"udp{rasp_prefix}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_psk{scenario_suffix}.csv"
        psk_files = find_files(data_dir_path, psk_pattern)
        print(f"Searching for PKI files with pattern: {psk_pattern}")
        print(f"Files found: {psk_files}")
        
        if psk_files:
            psk_file_path = psk_files[0]
            metric_value_psk, std_dev_psk = read_csv(psk_file_path, metric, rasp)
            if metric_value_psk is not None:
                data['psk']['values'].append(metric_value_psk)
                data['psk']['std_devs'].append(std_dev_psk)
                data['psk']['algorithms'].append(algorithm)
                print(f"Found PSK file for {algorithm}: {psk_file_path}")
            else:
                print(f"Warning: Could not read PSK data from {psk_file_path}")
        else:
            print(f"Warning: Could not find PSK file for algorithm {algorithm}")

    # Get PKI data for each algorithm and certificate type
    for cert_type in cert_types_list:
        for algorithm in algorithms_list:
            # PKI file pattern - look for files that contain both KEM algorithm and certificate type
            pki_pattern = f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki{scenario_suffix}.csv"
            pki_files = find_files(data_dir_path, pki_pattern)
            print(f"Searching for PKI files with pattern: {pki_pattern}")
            print(f"Files found: {pki_files}")
            
            # Also try with client-auth suffix if it exists
            if not pki_files:
                pki_pattern = f"udp{rasp_prefix}_conv_stats_{algorithm}_{cert_type}_n{n}{s_suffix}{p_suffix}_pki_client-auth{scenario_suffix}.csv"
                pki_files = find_files(data_dir_path, pki_pattern)
            
            if pki_files:
                pki_file_path = pki_files[0]
                metric_value_pki, std_dev_pki = read_csv(pki_file_path, metric, rasp)
                if metric_value_pki is not None:
                    data[cert_type]['values'].append(metric_value_pki)
                    data[cert_type]['std_devs'].append(std_dev_pki)
                    data[cert_type]['algorithms'].append(algorithm)
                    print(f"Found PKI file for {algorithm} with {cert_type}: {pki_file_path}")
                else:
                    print(f"Warning: Could not read PKI data from {pki_file_path}")
            else:
                print(f"Warning: Could not find PKI file for algorithm {algorithm} with {cert_type}")
    
    # Get nosec data
    nosec_pattern = f"udp{rasp_prefix}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}.csv"
    nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if not nosec_files:
        # Try a more flexible pattern if the exact one doesn't work
        nosec_pattern = f"udp{rasp_prefix}_conv_stats*n{n}*_nosec*{scenario_suffix}.csv"
        nosec_files = find_files(data_dir_path, nosec_pattern)
    
    if nosec_files:
        nosec_file_path = nosec_files[0]
        metric_value_horizontal, std_dev_horizontal = read_csv(nosec_file_path, metric, rasp)
        if metric_value_horizontal is not None:
            data['nosec']['value'] = metric_value_horizontal
            data['nosec']['std_dev'] = std_dev_horizontal
            print(f"Using nosec file: {nosec_file_path}")
        else:
            print(f"Warning: Could not read nosec data from {nosec_file_path}")
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
    algorithms_str = "_".join(algorithms_list)
    cert_types_str = "_".join([cert_type.replace('_', '') for cert_type in cert_types_list])
    
    s_suffix_file = f"_s{s}" if s else ""
    p_suffix_file = f"_{p}" if p else ""
    
    output_file = f'./bench-plots/{"rasp_" if rasp else ""}{clean_metric}_n{n}{s_suffix_file}{p_suffix_file}_{algorithms_str}_{cert_types_str}_scenario{scenario}.png'
    
    # Create the directory if it doesn't exist
    os.makedirs(f'./{plots_dir}', exist_ok=True)
    
    output_file = f'./{plots_dir}/{"rasp_" if rasp else ""}{clean_metric}_n{n}{s_suffix_file}{p_suffix_file}_{algorithms_str}_{cert_types_str}_scenario{scenario}.png'
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python3 coap_benchmark_plots.py <metric> <algorithms_list> <cert_types_list> <n> <rasp> <scenario> [s] [p] [custom_suffix]")
        print("\nExample: python3 coap_benchmark_plots.py 'duration' 'KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5' 'DILITHIUM_LEVEL2,RSA_2048' 50 true A")
        print("\nFor energy metrics: python3 coap_benchmark_plots.py 'Energy (Wh)' 'KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5' 'DILITHIUM_LEVEL2' 10 true A")
        print("\nWith custom directory: python3 coap_benchmark_plots.py 'duration' 'KYBER_LEVEL1' 'DILITHIUM_LEVEL2' 10 true A 10 background test1")
        print("  This reads from bench-data-test1 and outputs to bench-plots-test1")
        sys.exit(1)

    # Parse standard arguments
    metric = sys.argv[1]
    algorithms_list = [alg.strip() for alg in sys.argv[2].split(',')]
    cert_types_list = [cert.strip() for cert in sys.argv[3].split(',')]
    n = int(sys.argv[4])
    rasp = sys.argv[5].lower() == "true"
    scenario = sys.argv[6].upper()
    
    if scenario not in ['A', 'B', 'C']:
        print("Error: scenario must be one of 'A', 'B', or 'C'.")
        sys.exit(1)

    # Process optional arguments
    s, p, custom_suffix = None, None, None
    remaining_args = sys.argv[7:]
    
    # Parse s, p and custom_suffix parameters (if provided)
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
    
    # Call function with all parameters
    create_scatter_plots(metric, algorithms_list, cert_types_list, n, scenario, 
                        rasp, s, p, custom_suffix=custom_suffix)