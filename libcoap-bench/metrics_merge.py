import pandas as pd
import sys

# Function to read a CSV and calculate the mode for each column
def calculate_modes(input_csv):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv, sep=';')
    
    # Calculate the mode for each column
    modes = df.mode().iloc[0]
    
    return modes

# Function to append the modes to another CSV
def append_modes_to_csv(modes, output_csv):
    # Create a DataFrame with the modes
    mode_df = pd.DataFrame([modes], columns=modes.index)
    
    # Read the output CSV file into a DataFrame
    output_df = pd.read_csv(output_csv, sep=';')
    
    # Append the mode_df to the output_df for each row
    num_rows = len(output_df)
    mode_df = pd.concat([mode_df] * num_rows, ignore_index=True)
    mode_df.iloc[-1] = 0
    mode_df.iloc[-3] = "------------"   
    # Add column names with '_mode' suffix to avoid collision
    # mode_df.columns = [f'{col}' for col in mode_df.columns]
    
    # Concatenate the original DataFrame with the modes DataFrame
    result_df = pd.concat([output_df, mode_df], axis=1)
    
    # Write the result to the same output CSV file
    result_df.to_csv(output_csv, index=False, sep=";")
    result_df.columns = result_df.columns.map(lambda x: ';'.join(x))

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python mode_appender.py <input_csv> <output_csv>")
        sys.exit(1)
    
    # Get the input and output file paths from the command-line arguments
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    # Calculate modes from the input CSV
    modes = calculate_modes(input_csv)
    
    # Append the modes to the output CSV
    append_modes_to_csv(modes, output_csv)
