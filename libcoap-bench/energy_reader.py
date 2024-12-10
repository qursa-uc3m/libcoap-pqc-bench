import csv
from datetime import datetime, time
import argparse

def sum_power_in_time_range(csv_file_path, start_time_str, end_time_str):
    # Define the date and time format
    datetime_format = "%Y-%m-%d %H:%M:%S"
    time_format = "%H:%M:%S"

    # Parse the input time strings
    start_time = datetime.strptime(start_time_str, time_format).time()
    end_time = datetime.strptime(end_time_str, time_format).time()

    total_power = 0

    # Read the CSV file and calculate total power
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  # Use semicolon as delimiter
        header = next(reader, None)  # Read the header row

        if header is None:
            print(f"Error: Empty or invalid CSV file: {csv_file_path}")
            return None

        for row in reader:
            if len(row) < 4:
                print(f"Warning: Skipping row with unexpected format: {row}")
                continue

            try:
                timestamp_str = row[0].strip()  # Assuming timestamp is in the first column
                power = float(row[3].strip())   # Assuming power is in the fourth column
            except (IndexError, ValueError) as e:
                print(f"Error: Failed to parse row: {row}, {e}")
                continue

            # Parse the timestamp and extract time
            try:
                timestamp = datetime.strptime(timestamp_str, datetime_format)
                time_of_day = timestamp.time()
            except ValueError as e:
                print(f"Error: Failed to parse timestamp: {timestamp_str}, {e}")
                continue

            # Check if the time is within the specified range
            if start_time <= time_of_day <= end_time:
                total_power += power

    # Divide the total power by 3600 to get energy consumption in Wh
    energy_consumption = total_power / 3600

    return energy_consumption

def add_wh_column_to_csv(csv_file_path, energy_consumption, output_file=None):
    # Prepare the updated rows with "Wh" column
    updated_rows = []

    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')  # Use semicolon as delimiter
        header = next(reader, None)  # Read the header row

        if header is None:
            print(f"Error: Empty or invalid CSV file: {csv_file_path}")
            return

        # Prepare the updated header with "Wh" column
        new_header = header + ['Wh']

        # Count the total number of rows in the CSV file
        total_rows = sum(1 for row in reader)

        # Reset the reader to read from the beginning again
        csvfile.seek(0)
        next(reader)  # Skip the header row again

        # Iterate through each row and append the calculated energy consumption
        for i, row in enumerate(reader):
            if i == total_rows - 3:  # Third-to-last row
                updated_row = row + ['------------']
            elif i == total_rows - 2:  # Second-to-last row
                updated_row = row + [energy_consumption]
            elif i == total_rows - 1:  # Last row
                updated_row = row + [0.0000]
            else:
                updated_row = row + [energy_consumption]
            updated_rows.append(updated_row)

    # Write updated data to the output CSV file
    if output_file:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')  # Use semicolon as delimiter
            writer.writerow(new_header)  # Write header with 'Wh' column
            writer.writerows(updated_rows)
        print(f"Updated CSV written to {output_file}")
    else:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')  # Use semicolon as delimiter
            writer.writerow(new_header)  # Write header with 'Wh' column
            writer.writerows(updated_rows)
        print(f"Updated CSV file {csv_file_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Add "Wh" column to CSV file with specified values.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file to modify')
    parser.add_argument('energy_csv', type=str, help='Path to the energy CSV file')
    parser.add_argument('start_time', type=str, help='Start time in format hh:mm:ss')
    parser.add_argument('end_time', type=str, help='End time in format hh:mm:ss')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file (optional)')
    args = parser.parse_args()

    # Calculate energy consumption
    energy_consumption = sum_power_in_time_range(args.energy_csv, args.start_time, args.end_time)

    if energy_consumption is None:
        print("Error: Calculation of energy consumption failed.")
    else:
        # Add "Wh" column to CSV file
        add_wh_column_to_csv(args.csv_file, energy_consumption, args.output_file)