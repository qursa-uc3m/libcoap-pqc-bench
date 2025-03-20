import csv
import numpy as np
import sys
import os

# Check if the output file path is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python3 metrics_extractor.py <output_file_path>")
    sys.exit(1)

# Get the output file path from the command-line arguments
output_file_path = sys.argv[1]
output_dir = os.path.dirname(output_file_path)

# Read times from "time_output.txt" in the same directory as the output file
try:
    time_file_path = os.path.join(output_dir, 'time_output.txt')
    with open(time_file_path, 'r') as file:
        times = [float(line.strip()) for line in file]
    # print("Times:", times)
except Exception as e:
    print(f"Error reading {time_file_path}:", e)
    sys.exit(1)

# Read CPU cycles from "cycles_output.txt" in the same directory as the output file
try:
    cycles_file_path = os.path.join(output_dir, 'cycles_output.txt')
    with open(cycles_file_path, 'r') as file:
        cycles = int(file.read().strip())
    # print("Cycles:", cycles)
except Exception as e:
    print(f"Error reading {cycles_file_path}:", e)
    sys.exit(1)

# Calculate mean and standard deviation for times
try:
    mean_time = np.mean(times)
    std_dev_time = np.std(times)
    # print("Mean time:", mean_time)
    # print("Standard deviation time:", std_dev_time)
except Exception as e:
    print("Error calculating statistics:", e)
    sys.exit(1)

# Prepare data for CSV
try:
    rows = [[t, cycles] for t in times]
    rows.append(['------------', '------------'])
    rows.append([mean_time, cycles])
    rows.append([std_dev_time, 0])
    # print("Rows prepared for CSV:", rows)
except Exception as e:
    print("Error preparing rows:", e)
    sys.exit(1)

# Write to CSV with ';' as the separator
try:
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['duration', 'CPU cycles'])
        writer.writerows(rows)
    print(f"CSV file '{output_file_path}' created successfully.")
except Exception as e:
    print("Error writing to output file:", e)
    sys.exit(1)
