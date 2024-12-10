import csv
import os
import numpy as np
import argparse

def convert_to_csv(input_file, output_file, cpu_cycles_value):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Define the CSV writer
        csv_writer = csv.writer(outfile, delimiter=';')

        # Write a new header to the CSV file
        csv_writer.writerow(["frames -> B", "bytes -> B", "frames A <-", "bytes A <-", "total frames", "total bytes", "duration", "CPU cycles"])

        # Process each line in the input file
        data = []
        for line in infile:
            # Remove dots and replace commas with dots
            line = line.replace('.', '').replace(',', '.')

            # Split the line into columns
            columns = line.strip().split()
            columns[0:3] = [''.join(columns[0:3])]
            del columns[3]
            del columns[5]
            del columns[7]
            del columns[0]
            del columns[-2]

            # Convert relevant information to float
            frames_out, bytes_out, frames_in, bytes_in, total_frames, total_bytes, duration = map(float, columns[:7])

            # Create a new formatted string with the desired order of columns
            # Set CPU cycles column to the specified value
            csv_row = [frames_out, bytes_out, frames_in, bytes_in, total_frames, total_bytes, duration, cpu_cycles_value]
            data.append(csv_row)

            # Write the information to the CSV file without speechmarks
            csv_writer.writerow(csv_row)

        # Add a line of hyphens
        csv_writer.writerow(['-' * 12 for _ in range(8)])

        # Calculate mean and standard deviation
        mean_values = np.mean(data, axis=0)
        std_values = np.std(data, axis=0)

        # Add a line with mean values
        csv_writer.writerow(['{:.4f}'.format(mean) for mean in mean_values])

        # Add a line with standard deviation values
        csv_writer.writerow(['{:.4f}'.format(std) for std in std_values])

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Convert input file to CSV format')

    # Add arguments for input and output file paths
    parser.add_argument('input_file', help='Path to the input file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    parser.add_argument('cpu_cycles_value', type=float, help='Value for the CPU cycles column')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the convert_to_csv function with the provided file paths and CPU cycles value
    convert_to_csv(args.input_file, args.output_file, args.cpu_cycles_value)
