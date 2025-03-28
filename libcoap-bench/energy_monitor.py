#!/usr/bin/env python3
import serial
import time
import codecs
import argparse
import os
import csv
from datetime import datetime
import logging
import threading
from collections import deque
import sys
import signal

# Setup signal handler
def signal_handler(sig, frame):
    print("\nReceived termination signal. Shutting down...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set up logging
logging.basicConfig(
    filename="./libcoap-bench/energy_monitor.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------
# Capture Mode: Energy Monitoring
# -------------------------------
class UM34CReader:
    def __init__(self, port="/dev/rfcomm0", timeout=2):
        self.port = port
        self.timeout = timeout
        self.serial = None
        self.measurements = []
        self.start_time = None
        self.end_time = None
        self.lock = threading.Lock()
        self.read_times = deque(maxlen=20)  # Track recent read times
        self.max_power = 0.0  # Track maximum power observed
        
        # Device-specific parameters
        self.min_read_interval = 0.3  # Minimum time between reads (seconds)
    
    def connect(self):
        try:
            self.serial = serial.Serial(port=self.port, baudrate=9600, timeout=self.timeout)
            # Short delay after connecting to ensure device is ready
            time.sleep(0.2)
            logging.info(f"Connected to device on {self.port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to device: {e}")
            return False
    
    def read_measurement(self):
        if not self.serial:
            return None
            
        try:
            start_read = time.time()
            
            # Send command to get data
            self.serial.write(bytes.fromhex("f0"))
            data = self.serial.read(130)
            
            end_read = time.time()
            read_duration = end_read - start_read
            
            with self.lock:
                self.read_times.append(read_duration)
            
            if len(data) < 130:
                logging.warning(f"Incomplete data received: got {len(data)} bytes")
                return None
                
            measurement = self.parse_data(data)
            
            with self.lock:
                self.measurements.append(measurement)
                
                # Update max power if current reading is higher
                if measurement['power'] > self.max_power:
                    self.max_power = measurement['power']
                
                if not self.start_time:
                    self.start_time = datetime.now()
                    
                self.end_time = datetime.now()
            
            return measurement
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None
    
    def parse_data(self, data):
        data = codecs.encode(data, "hex").decode("utf-8")
        
        result = {}
        result["timestamp"] = time.time()
        result["voltage"] = int("0x" + data[4] + data[5] + data[6] + data[7], 0) / 100
        result["current"] = int("0x" + data[8] + data[9] + data[10] + data[11], 0) / 1000
        result["power"] = int("0x" + data[12] + data[13] + data[14] + data[15] + data[16] +
                           data[17] + data[18] + data[19], 0) / 1000
        result["temperature"] = int("0x" + data[20] + data[21] + data[22] + data[23], 0)
        
        return result
    
    def get_optimal_read_interval(self):
        """Calculate the optimal interval between read operations based on device limitations"""
        with self.lock:
            if not self.read_times:
                return self.min_read_interval
            
            # Calculate average read time
            avg_read_time = sum(self.read_times) / len(self.read_times)
            
            # Return the actual optimal interval (with a small safety margin)
            # but never less than the minimum read interval
            return max(self.min_read_interval, avg_read_time * 1.05)
    
    def disconnect(self):
        if self.serial:
            self.serial.close()
            logging.info("Disconnected from device")
    
    def calculate_energy(self):
        with self.lock:
            if not self.measurements:
                return 0, 0, 0
                
            total_power = sum(m["power"] for m in self.measurements)
            avg_power = total_power / len(self.measurements)
            max_power = self.max_power
            
            if self.start_time and self.end_time:
                duration_seconds = (self.end_time - self.start_time).total_seconds()
                energy_wh = avg_power * (duration_seconds / 3600)

                # Create directory if it doesn't exist
                benchmark_data_dir = "./libcoap-bench/bench-data"
                os.makedirs(benchmark_data_dir, exist_ok=True)
                
                # Save the time range for other scripts
                with open("./libcoap-bench/bench-data/initial_and_final_time.txt", "w") as f:
                    f.write(f"{self.start_time.strftime('%H:%M:%S')}\n")
                    f.write(f"{self.end_time.strftime('%H:%M:%S')}\n")
                    
                return avg_power, max_power, energy_wh
            
            return 0, 0, 0
    
    def save_measurements(self, filename):
        with self.lock:
            if not self.measurements:
                return False
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, "w", newline="") as csvfile:
                fieldnames = ["timestamp", "voltage", "current", "power", "temperature"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for m in self.measurements:
                    writer.writerow(m)
                    
            return True
    
    def add_energy_to_csv(self, csv_path):
        if not os.path.exists(csv_path):
            logging.warning(f"CSV file {csv_path} does not exist")
            return False
        
        avg_power, max_power, energy_wh = self.calculate_energy()
        
        # Read the CSV
        rows = []
        try:
            with open(csv_path, newline='') as csvfile:
                # First, detect the delimiter by looking at the first line
                first_line = csvfile.readline().strip()
                delimiter = ';'  # Default to semicolon
                if ',' in first_line and ';' not in first_line:
                    delimiter = ','
                
                # Reset to beginning of file
                csvfile.seek(0)
                
                reader = csv.reader(csvfile, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i == 0:  # Header row
                        rows.append(row + ['Power (W)', 'Max Power (W)', 'Energy (Wh)'])
                    else:
                        # Determine what values to add
                        power_value = f"{avg_power:.6f}"
                        max_power_value = f"{max_power:.6f}"
                        energy_value = f"{energy_wh:.6f}"
                        
                        # Special handling for certain rows
                        if len(rows) > 3 and i >= len(rows) - 3:
                            if i == len(rows) - 3:  # Third to last row
                                power_value = '------------'
                                max_power_value = '------------'
                                energy_value = '------------'
                            elif i == len(rows) - 1:  # Last row
                                power_value = '0.0000'
                                max_power_value = '0.0000'
                                energy_value = '0.0000'
                        
                        rows.append(row + [power_value, max_power_value, energy_value])
            
            # Write back to CSV
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerows(rows)
                
            logging.info(f"Added energy data to {csv_path}: Power={avg_power:.6f}W, Max Power={max_power:.6f}W, Energy={energy_wh:.6f}Wh")
            return True
            
        except Exception as e:
            logging.error(f"Error updating CSV: {e}")
            return False

def generate_filename(custom_name=None, prefix="energy_"):
    """Generate a filename based on custom name or timestamp"""
    # Create directory if it doesn't exist
    os.makedirs("./libcoap-bench/bench-data", exist_ok=True)
    
    if custom_name:
        # Use custom name if provided
        return f"./libcoap-bench/bench-data/{prefix}{custom_name}.csv"
    else:
        # Use timestamp for default naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"./libcoap-bench/bench-data/{prefix}measurements_{timestamp}.csv"

def monitor_energy(duration=0, sample_rate=0.3, port="/dev/rfcomm0", custom_name=None):
    """
    Monitor energy consumption from UM34C device
    
    Args:
        duration: Monitoring duration in seconds (0 for unlimited)
        sample_rate: Target sample rate in seconds
        port: Serial port to connect to
        custom_name: Custom name for output files (optional)
    """
    reader = UM34CReader(port=port)
    
    if not reader.connect():
        print("Failed to connect to device. Check connections and permissions.")
        return False
    
    print(f"Connected to device on {port}")
    print(f"Monitoring energy for {'unlimited time' if duration == 0 else str(duration) + ' seconds'}")
    
    # Track performance metrics
    samples_collected = 0
    start_monitoring = time.time()
    
    try:
        # Take a few initial readings to calibrate timing
        for _ in range(3):
            reader.read_measurement()
            time.sleep(0.3)  # Initial cautious delay
        
        # Get optimal read interval based on device performance
        optimal_read_interval = reader.get_optimal_read_interval()
        use_optimal = sample_rate < optimal_read_interval
        
        if use_optimal:
            print(f"Requested sample rate ({sample_rate}s) is faster than device capabilities.")
            print(f"Using optimal read interval: {optimal_read_interval:.3f}s (maximum ~{1/optimal_read_interval:.1f} samples/second)")
            effective_rate = optimal_read_interval
        else:
            print(f"Using requested sample rate: {sample_rate}s")
            effective_rate = sample_rate
        
        print("Press Ctrl+C to stop monitoring...")
        
        # Main monitoring loop
        start_time = time.time()
        next_read_time = start_time
        
        while True:
            current_time = time.time()
            
            # Check if we should exit based on duration
            if duration > 0 and current_time - start_time >= duration:
                break
            
            # Check if it's time to take a new reading
            if current_time >= next_read_time:
                measurement = reader.read_measurement()
                
                if measurement:
                    samples_collected += 1
                    if args.verbose:
                        print(f"V: {measurement['voltage']}V, I: {measurement['current']}A, P: {measurement['power']}W, T: {measurement['temperature']}°C")
                
                # Update optimal interval if we're using adaptive timing
                if use_optimal:
                    optimal_read_interval = reader.get_optimal_read_interval()
                    effective_rate = optimal_read_interval
                
                # Set next read time
                next_read_time = current_time + effective_rate
            else:
                # Small sleep to prevent CPU spinning
                remaining = next_read_time - current_time
                if remaining > 0.01:
                    time.sleep(min(0.01, remaining / 2))
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        end_monitoring = time.time()
        total_duration = end_monitoring - start_monitoring
        
        reader.disconnect()
        
        # Calculate energy metrics
        avg_power, max_power, energy_wh = reader.calculate_energy()
        actual_monitoring_duration = (reader.end_time - reader.start_time).total_seconds() if reader.start_time else 0
        
        print(f"\nMonitoring completed:")
        print(f"- Samples collected: {samples_collected}")
        print(f"- Monitoring duration: {actual_monitoring_duration:.2f} seconds")
        if samples_collected > 0:
            print(f"- Effective sample rate: {samples_collected / actual_monitoring_duration:.2f} Hz")
        print(f"- Average power: {avg_power:.6f}W")
        print(f"- Maximum power: {max_power:.6f}W")
        print(f"- Energy consumption: {energy_wh:.6f}Wh")
        
        # Generate filename based on custom name or timestamp
        filename = generate_filename(custom_name)
        
        # Save measurements
        reader.save_measurements(filename)
        print(f"- Measurements saved to {filename}")

        # Find and update benchmark CSV
        benchmark_csv = find_latest_csv_file()
        if benchmark_csv:
            if reader.add_energy_to_csv(benchmark_csv):
                print(f"- Energy data added to benchmark CSV: {benchmark_csv}")
            else:
                print(f"- Failed to add energy data to benchmark CSV")
        else:
            print("- No benchmark CSV file found to update")
        
        return True

# ----------------------------
# Merge Mode: Energy CSV Post-Processing
# ----------------------------
def extract_energy_data(energy_file):
    """Extract average power, max power and energy consumption from energy file."""
    power_w = 0
    max_power_w = 0
    energy_wh = 0
    try:
        with open(energy_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Calculate max power from individual measurements
            max_power_w = max([float(row.get('power', 0)) for row in rows], default=0)
            
            # Calculate average power
            total_power = sum([float(row.get('power', 0)) for row in rows])
            power_w = total_power / len(rows) if rows else 0
            
            # Calculate energy consumption
            if len(rows) > 1:
                first_timestamp = float(rows[0].get('timestamp', 0))
                last_timestamp = float(rows[-1].get('timestamp', 0))
                duration_hours = (last_timestamp - first_timestamp) / 3600
                energy_wh = power_w * duration_hours
    except Exception as e:
        print(f"Error reading energy file: {e}")
    return power_w, max_power_w, energy_wh

def add_energy_to_udp_csv(benchmark_file, power_w, max_power_w, energy_wh):
    """Add energy data to benchmark CSV file."""
    rows = []
    delimiter = ';'  # Default delimiter
    try:
        with open(benchmark_file, 'r') as f:
            first_line = f.readline().strip()
            if ',' in first_line and ';' not in first_line:
                delimiter = ','
            f.seek(0)
            reader_csv = csv.reader(f, delimiter=delimiter)
            
            # First, count the total lines to identify the last row
            all_rows = list(reader_csv)
            total_rows = len(all_rows)
            
            # Now process each row
            for i, row in enumerate(all_rows):
                if i == 0:  # Header row
                    rows.append(row + ['Power (W)', 'Max Power (W)', 'Energy (Wh)'])
                elif i == total_rows - 1:  # Last row (standard deviation)
                    # Set energy values to 0 for the standard deviation row
                    rows.append(row + ['0', '0', '0'])
                elif '-----------' in row[0]:  # Separator row
                    rows.append(row + ['------------', '------------', '------------'])
                else:  # Normal data rows
                    power_str = f"{power_w:.6f}"
                    max_power_str = f"{max_power_w:.6f}"
                    energy_str = f"{energy_wh:.6f}"
                    rows.append(row + [power_str, max_power_str, energy_str])
                    
        with open(benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerows(rows)
        
        logging.info(f"Added energy data to {benchmark_file}: Power={power_w:.6f}W, Max Power={max_power_w:.6f}W, Energy={energy_wh:.6f}Wh")
        return True
        
    except Exception as e:
        print(f"Error updating benchmark file: {e}")
        logging.error(f"Error updating benchmark file: {e}")
        return False

def find_latest_csv_file(directory="./libcoap-bench/bench-data"):
    """Find the most recently modified CSV file in the directory."""
    try:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return None
        csv_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if f.endswith('.csv') and not (f.endswith('_energy.csv') or f.endswith('_ws.csv'))]
        if not csv_files:
            print(f"Warning: No CSV files found in {directory}")
            return None
        latest_file = max(csv_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest CSV file: {e}")
        return None
    
# ----------------------------
# Main: Argument Parsing & Mode Selection
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Monitoring and Post-Processing Script")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--capture', action='store_true', help='Capture measurements and save to CSV')
    mode_group.add_argument('--merge', metavar='ENERGY_FILE', help='Merge energy data from the specified energy CSV file into a benchmark CSV')
    
    # Capture mode options
    parser.add_argument('--duration', type=int, default=0, help='Monitoring duration in seconds (0 for unlimited)')
    parser.add_argument('--rate', type=float, default=0.3, help='Sample rate in seconds (default: 0.3)')
    parser.add_argument('--port', type=str, default="/dev/rfcomm0", help='Serial port')
    parser.add_argument('--name', type=str, help='Custom name for output file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    # Merge mode options
    parser.add_argument('--benchmark', type=str, help='Path to the benchmark CSV file')
    parser.add_argument('--auto', action='store_true', help='Automatically find the latest benchmark CSV file')
    
    args = parser.parse_args()

    if args.capture:
        monitor_energy(
            duration=args.duration,
            sample_rate=args.rate,
            port=args.port,
            custom_name=args.name,
        )
    elif args.merge:
        energy_file = args.merge
        if not os.path.exists(energy_file):
            print(f"Energy file {energy_file} does not exist")
            exit(1)
        
        benchmark_file = args.benchmark
        if args.auto or not benchmark_file:
            benchmark_file = find_latest_csv_file()
            if not benchmark_file:
                print("Error: No benchmark CSV file found")
                exit(1)
        
        power_w, max_power_w, energy_wh = extract_energy_data(energy_file)
        if power_w > 0 or energy_wh > 0:
            if add_energy_to_udp_csv(benchmark_file, power_w, max_power_w, energy_wh):
                print(f"Successfully added energy data to {benchmark_file}")
            else:
                print(f"Failed to add energy data to {benchmark_file}")
        else:
            print("No valid energy data found")