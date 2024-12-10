
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np

color_palette = list(mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(np.linspace(0, 1, 3)))

def read_csv(file_path, metric_column, rasp=False):    
    # Read CSV file and split the complex column
    df = pd.read_csv(file_path, sep=';')

    # Get the last but one row and the specified metric column
    metric_value = float(df.iloc[-2][metric_column])
    std_dev_value = float(df.iloc[-1][metric_column])

    return metric_value, std_dev_value

def create_scatter_plots(metric, algorithms_list, n, scenario, rasp=False, s=None, p=None, data_dir='bench-data'):
    # Get the absolute path of the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set colors for PSK and PKI points
    pki_color = color_palette[0]
    psk_color = color_palette[1]

    # Lists to store PSK and PKI values
    psk_values = []
    pki_values = []
    psk_std_dev = []
    pki_std_dev = []

    # Plot scatter plots and lines for each algorithm
    for i, algorithm in enumerate(algorithms_list):
        # Read data from CSV file for psk
        s_suffix = f"_s{s}" if s else ""
        p_suffix = f"_{p}" if p else ""
        scenario_suffix = f"_scenario{scenario}"
        csv_file_path = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_psk{scenario_suffix}.csv')
        metric_value_psk, std_dev_psk = read_csv(csv_file_path, metric, rasp)  # Get metric and std dev values
        psk_values.append(metric_value_psk)
        psk_std_dev.append(std_dev_psk)

        # Read data from CSV file for pki
        csv_file_path = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_pki{scenario_suffix}.csv')
        metric_value_pki, std_dev_pki = read_csv(csv_file_path, metric, rasp)  # Get metric and std dev values
        pki_values.append(metric_value_pki)
        pki_std_dev.append(std_dev_pki)

        print(f"Reading data for algorithm: {algorithm}, n: {n}, s: {s}, p: {p}, scenario: {scenario}")
        print(f"CSV file path for psk: {csv_file_path}")
        print(f"CSV file path for pki: {csv_file_path}")
    
    # Read data for the first plot (horizontal line)
    csv_file_path_horizontal = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}.csv')
    metric_value_horizontal, std_dev_horizontal = read_csv(csv_file_path_horizontal, metric, rasp)  # Get metric and std dev values

    # Plot scatter plots for psk and pki with error bars
    ax.errorbar(algorithms_list, pki_values, yerr=pki_std_dev, label='PKI', color=pki_color, fmt='o', capsize=5)
    ax.errorbar(algorithms_list, psk_values, yerr=psk_std_dev, label='PSK', color=psk_color, fmt='o', capsize=5)

    # Plot lines connecting points for psk and pki
    ax.plot(algorithms_list, pki_values, '--', color=pki_color, label='_nolegend_')
    ax.plot(algorithms_list, psk_values, '--', color=psk_color, label='_nolegend_')

    # Plot horizontal line for nosec mode with error bars
    ax.errorbar(algorithms_list, [metric_value_horizontal] * len(algorithms_list), yerr=std_dev_horizontal, label='NoSec', color=color_palette[2], fmt='o', capsize=5)

    # Plot line connecting points for nosec mode
    ax.plot(algorithms_list, [metric_value_horizontal] * len(algorithms_list), '--', color=color_palette[2], label='_nolegend_')

    # Set labels and title
    ax.set_xlabel('Algorithms')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by algorithm and security mode - n={n}, s={s}, p={p}, scenario={scenario}' if s is not None or p is not None else f'{metric} by algorithm and security mode - n={n}, scenario={scenario}')
    
    # Calculate the minimum and maximum values considering error bars
    min_value = min(min(psk_values) - min(psk_std_dev), min(pki_values) - min(pki_std_dev), metric_value_horizontal - std_dev_horizontal)
    max_value = max(max(psk_values) + max(psk_std_dev), max(pki_values) + max(pki_std_dev), metric_value_horizontal + std_dev_horizontal)

    # Calculate the range based on the maximum absolute value
    value_range = 0.1 * max(abs(min_value), abs(max_value))

    # Set y-axis limits based on the calculated range
    ax.set_ylim(min_value - value_range, max_value + value_range)

    # Show legend
    ax.legend()

    # Save the plot
    clean_metric = metric.replace(' ', '_')
    string_algorithms_list = [str(element) for element in algorithms_list]
    delimiter = "_"
    result_algorithms_list = delimiter.join(string_algorithms_list)

    s_suffix = f"_s{s}" if s else ""
    p_suffix = f"_{p}" if p else ""
    scenario_suffix = f"_scenario{scenario}"
    
    plt.savefig(f'./bench-plots/{"rasp_" if rasp else ""}{clean_metric}_n{n}{s_suffix}{p_suffix}_{result_algorithms_list}{scenario_suffix}.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 8:
        print("Usage: python3 coap_benchmark_plots.py <metric> <algorithms_list> <n> <scenario> <rasp> [s] [p]")
        sys.exit(1)

    metric = sys.argv[1]
    algorithms_list = sys.argv[2].split(',')
    n = int(sys.argv[3])
    scenario = sys.argv[4].upper()
    if scenario not in ['A', 'B', 'C']:
        print("Error: scenario must be one of 'A', 'B', or 'C'.")
        sys.exit(1)
    rasp = sys.argv[5].lower() == "true"

    s = None
    p = None

    if len(sys.argv) >= 7:
        if sys.argv[6].isdigit():
            s = int(sys.argv[6])
            if len(sys.argv) == 8:
                p = sys.argv[7]
        else:
            p = sys.argv[6]
            if len(sys.argv) == 8:
                s = int(sys.argv[7]) if sys.argv[7].isdigit() else None

    create_scatter_plots(metric, algorithms_list, n, scenario, rasp, s, p)