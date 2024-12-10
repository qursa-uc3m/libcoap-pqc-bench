import csv
import os
import numpy as np
import argparse

def convert_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Define the CSV writer
        csv_writer = csv.writer(outfile, delimiter=';')

        # Write a new header to the CSV file
        csv_writer.writerow(["frames -> B", "bytes -> B", "frames A <-", "bytes A <-", "total frames", "total bytes"])

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

            # Check the unit of column 6 and multiply by 1000 if it's in kB
            if columns[7] == 'kB':
                columns[6] = str(float(columns[6]) * 1000)
            del columns[7]

            del columns[0]
            del columns[-2]
            del columns[-1]  # Remove the duration column

            # Convert relevant information to float
            frames_out, bytes_out, frames_in, bytes_in, total_frames, total_bytes = map(float, columns[:6])

            # Create a new formatted string with the desired order of columns
            csv_row = [frames_out, bytes_out, frames_in, bytes_in, total_frames, total_bytes]
            data.append(csv_row)

            # Write the information to the CSV file without speechmarks
            csv_writer.writerow(csv_row)

        # Add a line of hyphens
        csv_writer.writerow(['-' * 12 for _ in range(6)])

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

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the convert_to_csv function with the provided file paths
    convert_to_csv(args.input_file, args.output_file)
