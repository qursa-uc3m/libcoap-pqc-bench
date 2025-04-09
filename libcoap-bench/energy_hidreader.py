#!/usr/bin/env python3
# filepath: /home/dsobral/repos/libcoap-pqc-bench/libcoap-bench/energy_hidreader.py

"""
FNIRSI USB Power Meter Data Reader (HID Implementation)

This script reads data from FNIRSI USB power meters (FNB48, FNB58, C1 models)
and saves the measurements to a CSV file for energy monitoring and benchmarking.

Dependencies:
- Python 3.6+
- hidapi library (pip install hidapi)
- crc package (optional, pip install crc)

Usage:
  python3 energy_hidreader.py [options]

Options:
  --output FILE      Output file name (default: energy_data.csv)
  --duration SECONDS Duration to collect data in seconds (default: 0 = infinite)
  --crc              Enable CRC checks for data integrity
  --identify         Just identify the connected device and exit
  --list-devices     List all USB devices and exit
  --verbose          Enable verbose output
  --force-reset      Force a USB reset before starting
"""

import sys
import os
import time
import argparse
import csv
import signal
import numpy as np
from datetime import datetime
import subprocess
from typing import Union, Optional, List, Dict, Any, Tuple

try:
    import hid
except ImportError:
    print("Error: hidapi package not installed. Please install it with: pip install hidapi", file=sys.stderr)
    sys.exit(1)

# FNIRSI USB Power Meter Device IDs
DEVICE_IDS = {
    "FNB48": {"VID": 0x0483, "PID": 0x003A},
    "C1": {"VID": 0x0483, "PID": 0x003B},
    "FNB58": {"VID": 0x2E3C, "PID": 0x5558},
    "FNB48S": {"VID": 0x2E3C, "PID": 0x0049}
}

# Global state for accumulating measurements
class MeasurementState:
    def __init__(self):
        self.energy = 0.0       # Energy in Watt-seconds (Ws)
        self.capacity = 0.0     # Capacity in Ampere-seconds (As)
        self.start_time = None  # Start time of measurement
        self.temp_ema = None    # Exponential moving average for temperature
        self.max_power = 0.0    # Maximum power observed
        self.max_current = 0.0  # Maximum current observed
        self.max_voltage = 0.0  # Maximum voltage observed
        self.samples_count = 0  # Number of samples collected
        self.power_values = []  # Store power values
        self.last_elapsed_time= 0.0  # Track the elapsed time of the last sample


def reset_usb_device(vid, pid, verbose=False):
    """
    Attempt to reset the USB device using system commands
    
    Args:
        vid: Vendor ID in hex (e.g., 0x0483)
        pid: Product ID in hex (e.g., 0x003A)
        verbose: Whether to print verbose info
    
    Returns:
        Success status (boolean)
    """
    # Convert to string format needed for system commands
    vid_str = f"{vid:04x}"
    pid_str = f"{pid:04x}"
    
    try:
        # Find the device bus and device number using lsusb
        if verbose:
            print(f"Attempting to reset USB device {vid_str}:{pid_str}...", file=sys.stderr)
        
        # Try using usbreset utility if available
        try:
            # First check if usbreset is available (common on some Linux distros)
            result = subprocess.run(["which", "usbreset"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            
            if result.returncode == 0:
                reset_cmd = ["sudo", "usbreset", f"{vid_str}:{pid_str}"]
                result = subprocess.run(reset_cmd, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       text=True)
                
                if result.returncode == 0:
                    if verbose:
                        print("Device reset successful using usbreset utility", file=sys.stderr)
                    return True
        except Exception as e:
            if verbose:
                print(f"usbreset method failed: {e}", file=sys.stderr)
        
        # Method 2: Manual reset through sysfs
        try:
            # Find device location in sysfs
            cmd = ["lsusb", "-d", f"{vid_str}:{pid_str}"]
            result = subprocess.run(cmd, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            
            if result.returncode != 0:
                if verbose:
                    print("Could not find USB device", file=sys.stderr)
                return False
            
            # Parse the output to get bus and device number
            # Format: "Bus 00X Device 00Y: ID XXXX:YYYY Name"
            output = result.stdout.strip()
            parts = output.split()
            
            if len(parts) < 6:
                return False
            
            bus = parts[1]
            device = parts[3][:-1]  # Remove colon
            
            # Find the device path in sysfs
            path = f"/sys/bus/usb/devices/{bus}-{device}/authorized"
            
            # Check if path exists
            if not os.path.exists(path):
                # Try alternative pattern for paths
                path = f"/sys/bus/usb/devices/{bus}-{device[:-1]}.{device[-1]}/authorized"
                if not os.path.exists(path):
                    if verbose:
                        print(f"Could not find sysfs path for device {bus}-{device}", file=sys.stderr)
                    return False
            
            # Toggle authorized attribute to reset device
            try:
                with open(path, 'w') as f:
                    f.write("0")  # Deauthorize/disable
                time.sleep(0.5)   # Wait for device to be disabled
                with open(path, 'w') as f:
                    f.write("1")  # Authorize/enable again
                time.sleep(1.0)   # Wait for device to be re-enabled
                
                if verbose:
                    print(f"Successfully reset USB device through sysfs", file=sys.stderr)
                return True
            except (IOError, PermissionError) as e:
                if verbose:
                    print(f"Failed to reset device through sysfs: {e}", file=sys.stderr)
                    print("Try running the script with sudo for USB reset functionality", file=sys.stderr)
                return False
            
        except Exception as e:
            if verbose:
                print(f"Manual reset method failed: {e}", file=sys.stderr)
            return False
            
    except Exception as e:
        if verbose:
            print(f"Failed to reset USB device: {e}", file=sys.stderr)
        return False
    
    return False


# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\nReceived termination signal. Shutting down...", file=sys.stderr)
    if os.path.exists("fnirsi_stop"):
        os.remove("fnirsi_stop")
    sys.exit(0)


def setup_crc():
    """Set up CRC calculator if crc module is available"""
    try:
        import crc
        # CRC parameters reverse-engineered from device data
        width = 8
        poly = 0x39
        init_value = 0x42
        final_xor_value = 0x00
        reverse_input = False
        reverse_output = False
        configuration = crc.Configuration(width, poly, init_value, final_xor_value, reverse_input, reverse_output)
        
        if hasattr(crc, "CrcCalculator"):  # crc 1.x
            crc_calculator = crc.CrcCalculator(configuration, use_table=True)
            return crc_calculator.calculate_checksum
        else:  # crc 2.x+
            calculator = crc.Calculator(configuration, optimized=True)
            return calculator.checksum
    except ImportError:
        print("Warning: crc package not installed. CRC checks disabled.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Failed to set up CRC calculator: {e}", file=sys.stderr)
        return None


def list_all_usb_devices():
    """List all HID devices connected to the system"""
    print("HID Devices:")
    devices = hid.enumerate()
    for device in devices:
        manufacturer = device.get('manufacturer_string', 'Unknown')
        product = device.get('product_string', 'Unknown')
        
        # Check if this is a known FNIRSI device
        for name, ids in DEVICE_IDS.items():
            if device['vendor_id'] == ids["VID"] and device['product_id'] == ids["PID"]:
                if manufacturer == 'Unknown':
                    manufacturer = "FNIRSI"
                if product == 'Unknown':
                    product = name
        
        print(f"  Device: {device['vendor_id']:04x}:{device['product_id']:04x} {manufacturer} {product}")
    return True


def find_device():
    """
    Find a compatible FNIRSI power meter device
    
    Returns:
        Tuple containing:
        - Device path or None if not found
        - Boolean indicating if it's an FNB58 or FNB48S model
        - String with the model name
        - Dictionary with device VID/PID info
    """
    model_name = "Unknown"
    device_path = None
    is_fnb58_or_fnb48s = False
    device_info = None
    
    devices = hid.enumerate()
    
    for device in devices:
        for name, ids in DEVICE_IDS.items():
            if device['vendor_id'] == ids["VID"] and device['product_id'] == ids["PID"]:
                device_path = device['path']
                model_name = name
                is_fnb58_or_fnb48s = name in ["FNB58", "FNB48S"]
                device_info = ids
                return device_path, is_fnb58_or_fnb48s, model_name, device_info
    
    return None, False, model_name, None


def setup_device(device_path, max_attempts=3, verbose=False):
    """
    Set up the HID device for communication with retry logic
    
    Args:
        device_path: HID device path
        max_attempts: Maximum number of attempts to open the device
        verbose: Whether to print verbose information
        
    Returns:
        HID device object
    """
    if verbose:
        print("Opening HID device...", file=sys.stderr)
    
    device = hid.device()
    
    # Try multiple times to open the device with increasing delays
    attempt = 0
    last_error = None
    
    while attempt < max_attempts:
        try:
            device.open_path(device_path)
            
            # Set the device to non-blocking mode
            device.set_nonblocking(1)
            
            if verbose:
                try:
                    print(f"Manufacturer: {device.get_manufacturer_string()}", file=sys.stderr)
                    print(f"Product: {device.get_product_string()}", file=sys.stderr)
                    print(f"Serial Number: {device.get_serial_number_string()}", file=sys.stderr)
                except Exception as e:
                    print(f"Could not get device strings: {e}", file=sys.stderr)
            
            # Send initialization commands to "wake up" the device
            try:
                # Send a null report to initialize communication
                device.write([0, 0x00] + [0x00] * 62)
                time.sleep(0.01)
            except Exception as e:
                print(f"Warning: Could not send init packet: {e}", file=sys.stderr)
            
            return device
            
        except IOError as e:
            last_error = e
            attempt += 1
            wait_time = attempt * 1.0  # Linear backoff
            
            if verbose:
                print(f"Attempt {attempt}/{max_attempts} failed: {e}", file=sys.stderr)
                print(f"Waiting {wait_time:.1f} seconds before retry...", file=sys.stderr)
            
            time.sleep(wait_time)
    
    print(f"Error opening device after {max_attempts} attempts: {last_error}", file=sys.stderr)
    sys.exit(1)


def request_data(is_fnb58_or_fnb48s, device):
    """Send data request commands to the device"""
    # Note: When using hidapi, we need to add a report ID byte (0) at the beginning
    # The HID report is 64 bytes, total 65 with report ID
    try:
        device.write([0, 0xaa, 0x81] + [0x00] * 61 + [0x8e])
        time.sleep(0.02)  # Add small delay between commands
        device.write([0, 0xaa, 0x82] + [0x00] * 61 + [0x96])
        time.sleep(0.02)
        
        if is_fnb58_or_fnb48s:
            device.write([0, 0xaa, 0x82] + [0x00] * 61 + [0x96])
        else:
            device.write([0, 0xaa, 0x83] + [0x00] * 61 + [0x9e])
            
    except IOError as e:
        print(f"Error sending commands to device: {e}", file=sys.stderr)
        return False
        
    return True


def read_data(device, timeout=1000):
    """
    Read data from device with timeout
    
    Returns:
        Data bytes or None if timeout
    """
    start_time = time.time()
    while (time.time() - start_time) * 1000 < timeout:
        try:
            data = device.read(64)
            if data and len(data) > 0:
                return data
        except IOError as e:
            print(f"Error reading from device: {e}", file=sys.stderr)
            return None
            
        time.sleep(0.001)  # Short sleep to prevent CPU hogging
    
    return None


def drain_endpoint(device, timeout=100, verbose=False):
    """
    Drain any pending data from the HID device
    
    Args:
        device: HID device object
        timeout: Maximum time to spend draining in milliseconds
        verbose: Whether to print verbose info
    """
    start_time = time.time()
    drained_count = 0
    
    while (time.time() - start_time) * 1000 < timeout:
        try:
            data = device.read(64)
            if data and len(data) > 0:
                drained_count += 1
                if verbose:
                    print(f"Drained packet {drained_count} of {len(data)} bytes", file=sys.stderr)
            else:
                # No more data to read
                break
        except Exception as e:
            if verbose:
                print(f"Error during drain: {e}", file=sys.stderr)
            break
            
        time.sleep(0.001)  # Short sleep
    
    if verbose and drained_count > 0:
        print(f"Drained {drained_count} packets in total", file=sys.stderr)


def close_device_safely(device, verbose=False):
    """
    Close the HID device safely with proper cleanup
    
    Args:
        device: HID device object
        verbose: Whether to print verbose info
    """
    if device:
        try:
            # First try to send a final reset command
            try:
                # Send a null packet to reset device state
                device.write([0, 0xaa, 0x00] + [0x00] * 62)
                time.sleep(0.1)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not send reset packet: {e}", file=sys.stderr)
            
            # Drain any pending data
            drain_endpoint(device, verbose=verbose)
            
            # Close the device
            device.close()
            
            # Wait to ensure OS has time to properly release the device
            time.sleep(0.5)
            
            if verbose:
                print("Device closed successfully", file=sys.stderr)
                
        except Exception as e:
            print(f"Error while closing device: {e}", file=sys.stderr)


def decode_packet(data, state, calculate_crc, time_interval, alpha, csv_writer, end_time=None, verbose=False):
    """
    Decode a data packet and update measurement state
    
    Args:
        data: Raw data packet from the device
        state: MeasurementState object to update
        calculate_crc: CRC calculation function or None
        time_interval: Time interval between samples (seconds)
        alpha: Smoothing factor for temperature EMA
        csv_writer: CSV writer object for saving data
        verbose: Whether to print verbose info
        
    Returns:
        Boolean indicating if packet was valid and decoded
    """
    # For HID reports, we start at index 0 (no 0xaa prefix like with pyusb)
    # Second byte is payload type: 0x04 is data packet
    if len(data) < 2:
        return False
    
    packet_type = data[1]
    if packet_type != 0x04:
        # Ignore non-data packets
        return False
    
    # Verify CRC checksum if enabled
    if calculate_crc:
        actual_checksum = data[-1]
        expected_checksum = calculate_crc(bytearray(data[1:-1]))
        if actual_checksum != expected_checksum:
            if verbose:
                print(
                    f"Ignoring packet with invalid checksum. "
                    f"Expected: {expected_checksum:02x} Actual: {actual_checksum:02x}",
                    file=sys.stderr,
                )
            return False
    
    # Initialize start time if not already set
    if state.start_time is None:
        state.start_time = time.time() - 4 * time_interval
    
    # Current time reference for this packet
    t0 = time.time() - 4 * time_interval
    
    # Each packet contains 4 samples
    for i in range(4):
        offset = 2 + 15 * i
        
        # Extract measurements from packet
        voltage = (
            data[offset + 3] * 256 * 256 * 256
            + data[offset + 2] * 256 * 256
            + data[offset + 1] * 256
            + data[offset + 0]
        ) / 100000
        
        current = (
            data[offset + 7] * 256 * 256 * 256
            + data[offset + 6] * 256 * 256
            + data[offset + 5] * 256
            + data[offset + 4]
        ) / 100000
        
        temp_C = (data[offset + 13] + data[offset + 14] * 256) / 10.0
        
        # Calculate exponential moving average for temperature
        if state.temp_ema is not None:
            state.temp_ema = temp_C * (1.0 - alpha) + state.temp_ema * alpha
        else:
            state.temp_ema = temp_C
        
        # Calculate power and update accumulated values
        power = voltage * current
        state.power_values.append(power)
        state.energy += power * time_interval
        
        # Update max values
        state.max_power = max(state.max_power, power)
        state.max_current = max(state.max_current, current)
        state.max_voltage = max(state.max_voltage, voltage)
        
        # Calculate elapsed time for this sample
        t = t0 + i * time_interval
        elapsed = t - state.start_time
        state.last_elapsed_time = elapsed

        if end_time and t > end_time:
            return True
        
        # Increment sample counter
        state.samples_count += 1
        
        # Write data to CSV
        csv_writer.writerow([
            f"{state.samples_count}",     # Sample number
            f"{t:.6f}",                  # Timestamp
            f"{elapsed:.6f}",            # Elapsed time
            f"{voltage:.6f}",            # Voltage (V)
            f"{current:.6f}",            # Current (A)
            f"{power:.6f}",              # Power (W)
            f"{state.temp_ema:.3f}",     # Temperature (°C)
            f"{state.energy:.6f}",       # Accumulated energy (Ws)
        ])
    
    return True


def calculate_stddev(state):
    """Calculate standard deviation of a list of values"""
    if len(state.power_values) < 2:
        return 0.0
    return np.std(state.power_values, ddof=1)


def print_summary(state, duration):
    """Print a summary of the collected measurements"""
    if state.start_time is None:
        print("\nNo data was collected.", file=sys.stderr)
        return
    
    if state.samples_count > 0 and len(state.power_values) > 0:
        actual_elapsed_time = state.last_elapsed_time
    
    actual_duration = duration if duration > 0 else time.time() - state.start_time
    power_mean = np.mean(state.power_values)
    power_stddev = calculate_stddev(state)
    
    print("\n---------- Measurement Summary ----------", file=sys.stderr)
    print(f"Duration: {actual_duration:.2f} seconds", file=sys.stderr)
    print(f"Measurement elapsed time: {actual_elapsed_time:.2f} seconds", file=sys.stderr)
    print(f"Samples collected: {state.samples_count}", file=sys.stderr)
    print(f"Effective Sample Rate: {state.samples_count/actual_elapsed_time:.2f} sps", file=sys.stderr)
    print(f"Average power: {power_mean:.6f} W", file=sys.stderr)
    print(f"Maximum power: {state.max_power:.6f} W", file=sys.stderr)
    print(f"Power std deviation: {power_stddev:.6f} W", file=sys.stderr)
    print(f"Maximum current: {state.max_current:.6f} A", file=sys.stderr)
    print(f"Maximum voltage: {state.max_voltage:.6f} V", file=sys.stderr)
    print(f"Total energy (Accumulated): {state.energy:.6f} Ws ({state.energy / 3600:.6f} Wh)", file=sys.stderr)
    print(f"Total energy (elapsed time): {power_mean * actual_elapsed_time:.6f} Ws ({power_mean * actual_elapsed_time / 3600:.6f} Wh)", file=sys.stderr)
    print(f"Total energy (duration): {power_mean * actual_duration:.6f} Ws ({power_mean * actual_duration / 3600:.6f} Wh)", file=sys.stderr)
    if state.temp_ema is not None:
        print(f"Last temperature: {state.temp_ema:.3f} °C", file=sys.stderr)
    print("-------------------------------------------", file=sys.stderr)


def save_energy_summary(state, output_file):
    """
    Save energy data summary to a file compatible with the benchmark system
    
    This creates a simplified CSV with just the primary energy metrics that
    can be used with the existing metrics processing scripts.
    """
    if state.start_time is None:
        return
    
    actual_duration = time.time() - state.start_time
    power_mean = np.mean(state.power_values)
    power_std_dev = calculate_stddev(state)
    
    # Calculate energy uncertainty through error propagation
    # For energy = power * time, if time is exact, then:
    # σ_energy = σ_power * time
    energy_std_dev = power_std_dev * actual_duration / 3600  # Convert to Wh

    # Create simplified energy data for benchmarking system
    summary_file = os.path.splitext(output_file)[0] + ".csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["timestamp", "power", "max_power", "energy"])
        
        # Write the actual data row
        writer.writerow([
            time.time(),
            f"{power_mean:.6f}",  # Average power
            f"{state.max_power:.6f}",                # Max power
            f"{state.energy / 3600:.6f}"             # Energy in Wh
        ])
        
        # Write separator row
        writer.writerow(["-----------", "-----------", "-----------", "-----------"])
        
        # Write mean values row (same as the data row for a single measurement)
        writer.writerow([
            time.time(),
            f"{power_mean:.6f}",  # Average power
            f"{state.max_power:.6f}",                # Max power
            f"{state.energy / 3600:.6f}"             # Energy in Wh
        ])
        
         # Write standard deviation row
        writer.writerow([
            "0",
            f"{power_std_dev:.6f}",                  # Power std dev
            "0",                                     # No std dev for max power
            f"{energy_std_dev:.6f}"                  # Energy std dev in Wh
        ])
    
    print(f"Energy summary saved to {summary_file}", file=sys.stderr)
    return summary_file


def main():
    """Main program function"""
    # Register signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="FNIRSI USB Power Meter Data Reader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=str, default="energy_data",
                        help="Output file name")
    parser.add_argument("--duration", type=float, default=0,
                        help="Duration to collect data in seconds (0 = infinite)")
    parser.add_argument("--crc", action="store_true",
                        help="Enable CRC checks for data integrity")
    parser.add_argument("--identify", action="store_true",
                        help="Just identify the connected device and exit")
    parser.add_argument("--list-devices", action="store_true",
                        help="List all USB devices and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Set temperature smoothing factor (0-1)")
    parser.add_argument("--force-reset", action="store_true",
                        help="Force USB device reset before starting")
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retry attempts for device operations")
    args = parser.parse_args()
    
    # List all USB devices if requested
    if args.list_devices:
        list_all_usb_devices()
        return 0
    
    # Find the device
    device_path, is_fnb58_or_fnb48s, model_name, device_info = find_device()
    
    if not device_path:
        print("Error: FNIRSI USB power meter not found. Check connection and permissions.", file=sys.stderr)
        print("Available USB devices:", file=sys.stderr)
        list_all_usb_devices()
        return 1
    
    print(f"Found {model_name} USB power meter", file=sys.stderr)
    
    # Force reset the USB device if requested
    if args.force_reset and device_info:
        print("Attempting USB device reset...", file=sys.stderr)
        reset_success = reset_usb_device(device_info["VID"], device_info["PID"], args.verbose)
        if reset_success:
            print("USB device reset successful", file=sys.stderr)
        else:
            print("USB device reset failed. Continuing anyway...", file=sys.stderr)
        
        # Wait after reset to allow device to stabilize
        time.sleep(2)
        
        # Re-enumerate to get updated path after reset
        device_path, is_fnb58_or_fnb48s, model_name, device_info = find_device()
        if not device_path:
            print("Error: Device not found after reset. Try reconnecting it manually.", file=sys.stderr)
            return 1
    
    # If only identifying the device, exit now
    if args.identify:
        return 0
    
    # Set up CRC calculator if requested
    crc_calculator = None
    if args.crc:
        try:
            crc_calculator = setup_crc()
            if crc_calculator:
                print("CRC checks enabled", file=sys.stderr)
        except Exception as e:
            print(f"Warning: CRC setup failed: {e}", file=sys.stderr)
    
    # Set up the device for communication
    device = None
    try:
        device = setup_device(device_path, args.retry, args.verbose)
    except Exception as e:
        print(f"Error setting up device: {e}", file=sys.stderr)
        return 1
    
    # Prepare the output file
    output_file = args.output+"_raw.csv"
    print(f"Data will be saved to: {output_file}", file=sys.stderr)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Initialize measurement state
    state = MeasurementState()
    
    # At the moment only 100 sps is supported by the device
    sps = 100
    time_interval = 1.0 / sps
    
    # Open the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write CSV header
        csv_writer.writerow([
            "sample",
            "timestamp",
            "elapsed",
            "voltage_V",
            "current_A",
            "power_W",
            "temp_C",
            "energy_Ws",
        ])
        
        # Request initial data from the device
        if not request_data(is_fnb58_or_fnb48s, device):
            print("Failed to request data from device", file=sys.stderr)
            close_device_safely(device, args.verbose)
            return 1
        
        # Allow more time for the device to respond
        time.sleep(0.01)
        
        # Set up data refresh timing
        refresh = 1.0 if is_fnb58_or_fnb48s else 0.003  # 1 s for FNB58 / FNB48S, 3 ms for others
        continue_time = time.time() + refresh
        
        # Calculate end time if duration is specified
        end_time = time.time() + args.duration - 0.03 if args.duration > 0 else None
        
        # Add counters for error handling
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Main measurement loop
        try:
            while True:
                # Check if duration has been reached
                current_time = time.time()
                if end_time and current_time >= end_time:
                    print("Specified duration reached", file=sys.stderr)
                    break
                
                # Read data from the device
                try:
                    data = read_data(device, timeout=5000)
                    if data:
                        # Process data if valid
                        if decode_packet(data, state, crc_calculator, time_interval, args.alpha, csv_writer, end_time, args.verbose):
                            consecutive_errors = 0  # Reset error counter on success
                        
                    else:
                        # No data received
                        consecutive_errors += 1
                        print(f"No data received ({consecutive_errors}/{max_consecutive_errors}), retrying...", file=sys.stderr)
                        
                        if consecutive_errors >= max_consecutive_errors:
                            print("Too many consecutive errors. Device may be unresponsive.", file=sys.stderr)
                            break
                    
                    # Request more data if it's time
                    if time.time() >= continue_time:
                        continue_time = time.time() + refresh
                        device.write([0, 0xaa, 0x83] + [0x00] * 61 + [0x9e])
                    
                except Exception as e:
                    print(f"Error reading data: {e}", file=sys.stderr)
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors. Exiting.", file=sys.stderr)
                        break
                
                # Check for stop request (file or keyboard interrupt)
                if os.path.exists("fnirsi_stop"):
                    print("Stop file detected, terminating", file=sys.stderr)
                    break
        
        finally:
            # Use our safer close function
            close_device_safely(device, args.verbose)
            
            # Print summary of collected data
            print_summary(state, args.duration)
            
            # Save energy summary for benchmarking system
            output_file = args.output+".csv"
            summary_file = save_energy_summary(state, output_file)
            
            # Create a touch file to indicate successful completion
            with open(f"{output_file}.done", 'w') as f:
                f.write(f"Completed at {datetime.now().isoformat()}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())