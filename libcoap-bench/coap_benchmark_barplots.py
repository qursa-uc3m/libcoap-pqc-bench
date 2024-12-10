# import sys
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import matplotlib.colors as mcolors
# from matplotlib.patches import Patch

# # Generate the color palette
# color_palette = list(mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(np.linspace(0, 1, 9)))
# security_mode_colors = {
#     "pki": color_palette[0],
#     "psk": color_palette[3],
#     "nosec": color_palette[6]
# }

# def read_csv(file_path, metric_column):
#     """
#     Read CSV file and extract the metric and standard deviation values from specified columns.
    
#     Args:
#         file_path (str): Path to the CSV file.
#         metric_column (str): Column name for the metric value.

#     Returns:
#         tuple: Metric value and standard deviation value, or None, None if an error occurs.
#     """
#     try:
#         df = pd.read_csv(file_path, sep=';')
#         metric_value = float(df.iloc[-2][metric_column])
#         std_dev_value = float(df.iloc[-1][metric_column])
#         return metric_value, std_dev_value
#     except FileNotFoundError:
#         print(f"Error: File not found - {file_path}")
#         return None, None
#     except KeyError:
#         print(f"Error: Column '{metric_column}' not found in {file_path}")
#         return None, None
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return None, None

# def create_bar_plot(metric, algorithms_list, n, scenarios, rasp=False, s=None, p=None, data_dir='bench-data'):
#     """
#     Create bar plot for the specified metric and algorithms under different security modes and scenarios.

#     Args:
#         metric (str): The metric to be plotted.
#         algorithms_list (list): List of algorithm names.
#         n (int): Number of repetitions.
#         scenarios (list): List of scenarios to consider.
#         rasp (bool): If True, use Rasp dataset.
#         s (int or None): Optional 's' parameter.
#         p (str or None): Optional 'p' parameter.
#         data_dir (str): Directory containing the data files.
#     """
#     script_directory = os.path.dirname(os.path.realpath(__file__))
    
#     fig, ax = plt.subplots(figsize=(14, 8))

#     security_modes = ['pki', 'psk', 'nosec']
#     regular_modes = security_modes[:-1]  # ['pki', 'psk']
    
#     bars_per_algorithm = len(regular_modes) * len(scenarios)

#     # Calculate the number of algorithms
#     num_algorithms = len(algorithms_list)

#     # Calculate the total number of positions for regular modes
#     total_positions = num_algorithms * bars_per_algorithm

#     # Define the spacing between blocks
#     block_spacing = 2  # Increase block spacing to separate nosec mode clearly

#     # Initialize the x_positions array for regular modes
#     x_positions = np.zeros(total_positions)

#     # Calculate x_positions for regular modes
#     for i in range(num_algorithms):
#         start_pos = i * (bars_per_algorithm + block_spacing)
#         for j in range(bars_per_algorithm):
#             x_positions[i * bars_per_algorithm + j] = start_pos + j * (1 + 0.05)

#     # Calculate positions for 'nosec' mode
#     nosec_positions = np.zeros(len(scenarios))
#     nosec_start = x_positions[-1] + block_spacing  # Start after the last regular mode position + spacing

#     for j in range(len(scenarios)):
#         nosec_positions[j] = nosec_start + j * (1 + 0.05)

#     x_labels = [''] * (total_positions + len(nosec_positions))  # Initialize x_labels with empty strings
#     legend_entries = {}

#     # Plot regular modes
#     for i, algorithm in enumerate(algorithms_list):
#         for j, scenario in enumerate(scenarios):
#             for k, mode in enumerate(regular_modes):
#                 s_suffix = f"_s{s}" if s else ""
#                 p_suffix = f"_{p}" if p else ""
#                 scenario_suffix = f"_scenario{scenario}"

#                 csv_file_path = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_{mode}{scenario_suffix}.csv')
                
#                 metric_value, std_dev_value = read_csv(csv_file_path, metric)

#                 if metric_value is not None and std_dev_value is not None:
#                     position = i * bars_per_algorithm + j * len(regular_modes) + k
                    
#                     bar_color = security_mode_colors[mode]
                    
#                     # Add error bars with cap style
#                     (_, caps, _) = ax.errorbar(x_positions[position], metric_value, yerr=std_dev_value, capsize=5, fmt='o', color='black', markersize=5)
#                     for cap in caps:
#                         cap.set_markeredgewidth(1)

#                     # Create the bar
#                     ax.bar(x_positions[position], metric_value, width=1, color=bar_color, edgecolor='black', linewidth=1)
                    
#                     legend_entries[f'{mode}'] = bar_color
#                     x_labels[position] = f'{algorithm}-{scenario}-{mode}'

#     # Plot 'nosec' mode separately
#     for j, scenario in enumerate(scenarios):
#         s_suffix = f"_s{s}" if s else ""
#         p_suffix = f"_{p}" if p else ""
#         scenario_suffix = f"_scenario{scenario}"

#         csv_file_path = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}.csv')
        
#         metric_value, std_dev_value = read_csv(csv_file_path, metric)

#         if metric_value is not None and std_dev_value is not None:
#             position = j
#             bar_color = security_mode_colors['nosec']
            
#             # Add error bars with cap style
#             (_, caps, _) = ax.errorbar(nosec_positions[position], metric_value, yerr=std_dev_value, capsize=5, fmt='o', color='black', markersize=5)
#             for cap in caps:
#                 cap.set_markeredgewidth(1)

#             # Create the bar
#             ax.bar(nosec_positions[position], metric_value, width=1, color=bar_color, edgecolor='black', linewidth=1)
            
#             legend_entries['nosec'] = bar_color
#             x_labels[total_positions + position] = f'{scenario}-nosec'

#     # Set the x-ticks and labels
#     filtered_x_positions = [x for x, label in zip(np.concatenate((x_positions, nosec_positions)), x_labels) if label]
#     filtered_x_labels = [label for label in x_labels if label]

#     ax.set_xticks(filtered_x_positions)
#     ax.set_xticklabels(filtered_x_labels, rotation=45, ha='right', fontsize=10)

#     ax.set_xlabel('(Algorithm -) Scenario - Mode')
#     ax.set_ylabel(metric)
#     ax.set_title(f'{metric} by algorithm, security mode, and scenario - n={n}, s={s}, p={p}')
    
#     # Create a custom legend
#     handles = [Patch(facecolor=color, label=label) for label, color in legend_entries.items()]
#     ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)

#     plt.tight_layout()
#     algorithms_str = "_".join(algorithms_list)
#     plt.savefig(f'./bench-plots/barplot_{"rasp_" if rasp else ""}{metric}_n{n}_{s if s else ""}_{p if p else ""}_{algorithms_str}.png')
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) < 6 or len(sys.argv) > 9:
#         print("Usage: python3 coap_benchmark_barplots.py <metric> <algorithms_list> <n> <rasp> <scenarios_list> [s] [p]")
#         sys.exit(1)

#     metric = sys.argv[1]
#     algorithms_list = sys.argv[2].split(',')
#     n = int(sys.argv[3])
#     rasp = sys.argv[4].lower() == "true"
#     scenarios_list = sys.argv[5].split(',')

#     s, p = None, None
#     if len(sys.argv) >= 7:
#         if sys.argv[6].isdigit():
#             s = int(sys.argv[6])
#             if len(sys.argv) == 8:
#                 p = sys.argv[7]
#         else:
#             p = sys.argv[6]
#             if len(sys.argv) == 8:
#                 s = int(sys.argv[7]) if sys.argv[7].isdigit() else None

#     create_bar_plot(metric, algorithms_list, n, scenarios_list, rasp, s, p)

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

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

def create_bar_plot(metric, algorithms_list, n, scenarios, rasp=False, s=None, p=None, data_dir='bench-data'):
    """
    Create bar plot for the specified metric and algorithms under different security modes and scenarios.

    Args:
        metric (str): The metric to be plotted.
        algorithms_list (list): List of algorithm names.
        n (int): Number of repetitions.
        scenarios (list): List of scenarios to consider.
        rasp (bool): If True, use Rasp dataset.
        s (int or None): Optional 's' parameter.
        p (str or None): Optional 'p' parameter.
        data_dir (str): Directory containing the data files.
    """
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    fig, ax = plt.subplots(figsize=(14, 8))

    security_modes = ['pki', 'psk', 'nosec']
    regular_modes = security_modes[:-1]  # ['pki', 'psk']
    
    bars_per_algorithm = len(scenarios) * len(regular_modes)

    # Calculate the number of algorithms
    num_algorithms = len(algorithms_list)

    # Calculate the total number of positions for regular modes
    total_positions = num_algorithms * bars_per_algorithm

    # Define the spacing between blocks
    block_spacing = 2  # Increase block spacing to separate nosec mode clearly

    # Initialize the x_positions array for regular modes
    x_positions = np.zeros(total_positions)

    # Calculate x_positions for regular modes
    for i in range(num_algorithms):
        start_pos = i * (bars_per_algorithm + block_spacing)
        for j in range(bars_per_algorithm):
            x_positions[i * bars_per_algorithm + j] = start_pos + j * (1 + 0.05)

    # Calculate positions for 'nosec' mode
    nosec_positions = np.zeros(len(scenarios))
    nosec_start = x_positions[-1] + block_spacing  # Start after the last regular mode position + spacing

    for j in range(len(scenarios)):
        nosec_positions[j] = nosec_start + j * (1 + 0.05)

    x_labels = [''] * (total_positions + len(nosec_positions))  # Initialize x_labels with empty strings
    legend_entries = {}

    # Plot regular modes
    for i, algorithm in enumerate(algorithms_list):
        for k, mode in enumerate(regular_modes):
            for j, scenario in enumerate(scenarios):
                s_suffix = f"_s{s}" if s else ""
                p_suffix = f"_{p}" if p else ""
                scenario_suffix = f"_scenario{scenario}"

                csv_file_path = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_{algorithm}_n{n}{s_suffix}{p_suffix}_{mode}{scenario_suffix}.csv')
                
                metric_value, std_dev_value = read_csv(csv_file_path, metric)

                if metric_value is not None and std_dev_value is not None:
                    position = i * bars_per_algorithm + k * len(scenarios) + j
                    
                    bar_color = security_mode_colors[mode]
                    
                    # Add error bars with cap style
                    (_, caps, _) = ax.errorbar(x_positions[position], metric_value, yerr=std_dev_value, capsize=5, fmt='o', color='black', markersize=5)
                    for cap in caps:
                        cap.set_markeredgewidth(1)

                    # Create the bar
                    ax.bar(x_positions[position], metric_value, width=1, color=bar_color, edgecolor='black', linewidth=1)
                    
                    legend_entries[f'{mode}'] = bar_color
                    x_labels[position] = f'{algorithm}-{mode}-{scenario}'

    # Plot 'nosec' mode separately
    for j, scenario in enumerate(scenarios):
        s_suffix = f"_s{s}" if s else ""
        p_suffix = f"_{p}" if p else ""
        scenario_suffix = f"_scenario{scenario}"

        csv_file_path = os.path.join(script_directory, data_dir, f'udp{"_rasp" if rasp else ""}_conv_stats_n{n}{s_suffix}{p_suffix}_nosec{scenario_suffix}.csv')
        
        metric_value, std_dev_value = read_csv(csv_file_path, metric)

        if metric_value is not None and std_dev_value is not None:
            position = j
            bar_color = security_mode_colors['nosec']
            
            # Add error bars with cap style
            (_, caps, _) = ax.errorbar(nosec_positions[position], metric_value, yerr=std_dev_value, capsize=5, fmt='o', color='black', markersize=5)
            for cap in caps:
                cap.set_markeredgewidth(1)

            # Create the bar
            ax.bar(nosec_positions[position], metric_value, width=1, color=bar_color, edgecolor='black', linewidth=1)
            
            legend_entries['nosec'] = bar_color
            x_labels[total_positions + position] = f'nosec-{scenario}'

    # Set the x-ticks and labels
    filtered_x_positions = [x for x, label in zip(np.concatenate((x_positions, nosec_positions)), x_labels) if label]
    filtered_x_labels = [label for label in x_labels if label]

    ax.set_xticks(filtered_x_positions)
    ax.set_xticklabels(filtered_x_labels, rotation=45, ha='right', fontsize=10)

    ax.set_xlabel('(Algorithm -) Mode - Scenario')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by algorithm, security mode, and scenario - n={n}, s={s}, p={p}')
    
    # Create a custom legend
    handles = [Patch(facecolor=color, label=label) for label, color in legend_entries.items()]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)

    plt.tight_layout()
    algorithms_str = "_".join(algorithms_list)
    plt.savefig(f'./bench-plots/barplot_{"rasp_" if rasp else ""}{metric}_n{n}_{s if s else ""}_{p if p else ""}_{algorithms_str}.png')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 9:
        print("Usage: python3 coap_benchmark_barplots.py <metric> <algorithms_list> <n> <rasp> <scenarios_list> [s] [p]")
        sys.exit(1)

    metric = sys.argv[1]
    algorithms_list = sys.argv[2].split(',')
    n = int(sys.argv[3])
    rasp = sys.argv[4].lower() == "true"
    scenarios_list = sys.argv[5].split(',')

    s, p = None, None
    if len(sys.argv) >= 7:
        if sys.argv[6].isdigit():
            s = int(sys.argv[6])
            if len(sys.argv) == 8:
                p = sys.argv[7]
        else:
            p = sys.argv[6]
            if len(sys.argv) == 8:
                s = int(sys.argv[7]) if sys.argv[7].isdigit() else None

    create_bar_plot(metric, algorithms_list, n, scenarios_list, rasp, s, p)

