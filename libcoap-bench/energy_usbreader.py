#!/usr/bin/env python3
"""
FNIRSI USB Power Meter Data Reader

This script reads data from FNIRSI USB power meters (FNB48, FNB58, C1 models)
and saves the measurements to a CSV file for energy monitoring and benchmarking.

Dependencies:
- Python 3.6+
- pyusb library (pip install pyusb)
- crc package (optional, pip install crc)

Usage:
  python3 energy_reader.py [options]

Options:
  --output FILE      Output file name (default: energy_data.csv)
  --duration SECONDS Duration to collect data in seconds (default: 0 = infinite)
  --crc              Enable CRC checks for data integrity
  --identify         Just identify the connected device and exit
  --list-devices     List all USB devices and exit
  --verbose          Enable verbose output
"""

import sys
import os
import time
import argparse
import csv
import signal
import numpy as np
from datetime import datetime
from typing import Union, Optional, List, Dict, Any, Tuple

try:
    import usb.core
    import usb.util
except ImportError:
    print("Error: pyusb package not installed. Please install it with: pip install pyusb", file=sys.stderr)
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
    """List all USB devices connected to the system"""
    print("USB Devices:")
    devices = usb.core.find(find_all=True)
    for dev in devices:
        try:
            manufacturer = usb.util.get_string(dev, dev.iManufacturer)
        except:
            # If reading manufacturer fails, check if this is a known FNIRSI device
            manufacturer = "Unknown"
            for name, ids in DEVICE_IDS.items():
                if dev.idVendor == ids["VID"] and dev.idProduct == ids["PID"]:
                    manufacturer = "FNIRSI"
                    break
        
        try:
            product = usb.util.get_string(dev, dev.iProduct)
        except:
            # If reading product fails, check if this is a known FNIRSI device
            product = "Unknown"
            for name, ids in DEVICE_IDS.items():
                if dev.idVendor == ids["VID"] and dev.idProduct == ids["PID"]:
                    product = name
                    break
            
        print(f"  Bus {dev.bus:03d} Device {dev.address:03d}: ID {dev.idVendor:04x}:{dev.idProduct:04x} {manufacturer} {product}")
    return True


def find_device() -> Tuple[Optional[usb.core.Device], bool, str]:
    """
    Find a compatible FNIRSI power meter device
    
    Returns:
        Tuple containing:
        - Device object or None if not found
        - Boolean indicating if it's an FNB58 or FNB48S model
        - String with the model name
    """
    model_name = "Unknown"
    
    # Try FNB48
    dev = usb.core.find(idVendor=DEVICE_IDS["FNB48"]["VID"], idProduct=DEVICE_IDS["FNB48"]["PID"])
    if dev:
        model_name = "FNB48"
        return dev, False, model_name
    
    # Try C1
    dev = usb.core.find(idVendor=DEVICE_IDS["C1"]["VID"], idProduct=DEVICE_IDS["C1"]["PID"])
    if dev:
        model_name = "C1"
        return dev, False, model_name
    
    # Try FNB58
    dev = usb.core.find(idVendor=DEVICE_IDS["FNB58"]["VID"], idProduct=DEVICE_IDS["FNB58"]["PID"])
    if dev:
        model_name = "FNB58"
        return dev, True, model_name
    
    # Try FNB48S
    dev = usb.core.find(idVendor=DEVICE_IDS["FNB48S"]["VID"], idProduct=DEVICE_IDS["FNB48S"]["PID"])
    if dev:
        model_name = "FNB48S"
        return dev, True, model_name
    
    return None, False, model_name


def setup_device(dev, verbose=False):
    """
    Set up the USB device for communication
    
    Args:
        dev: USB device object
        verbose: Whether to print verbose information
        
    Returns:
        Tuple containing input and output endpoints
    """
    if verbose:
        print("Setting up device...", file=sys.stderr)
    
    # Reset the device to ensure a clean state
    try:
        dev.reset()
    except usb.core.USBError as e:
        print(f"Warning: Could not reset device: {e}", file=sys.stderr)
    
    # Find the HID interface
    interface_hid_num = find_hid_interface(dev)
    if verbose:
        print(f"Using HID interface #{interface_hid_num}", file=sys.stderr)
    
    # Ensure kernel drivers are detached
    if verbose:
        print("Detaching kernel drivers if needed...", file=sys.stderr)
    ensure_all_interfaces_not_busy(dev)
    
    # Set the active configuration
    if verbose:
        print("Setting active configuration...", file=sys.stderr)
    try:
        dev.set_configuration()
    except usb.core.USBError as e:
        print(f"Error setting configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get the active configuration
    cfg = dev.get_active_configuration()
    intf = cfg[(interface_hid_num, 0)]
    
    # Find endpoints
    ep_out = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT,
    )
    
    ep_in = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN,
    )
    
    if not ep_in or not ep_out:
        print("Error: Could not find required USB endpoints", file=sys.stderr)
        sys.exit(1)
    
    return ep_in, ep_out


def find_hid_interface(dev):
    """Find the HID interface number"""
    for cfg in dev:
        for interface in cfg:
            if interface.bInterfaceClass == 0x03:  # HID class
                return interface.bInterfaceNumber
    
    # If no HID interface found, default to interface 0
    print("Warning: No HID interface found, using interface 0", file=sys.stderr)
    return 0


def ensure_all_interfaces_not_busy(dev):
    """Detach kernel drivers from all interfaces if necessary"""
    for cfg in dev:
        for interface in cfg:
            ensure_interface_not_busy(dev, interface)


def ensure_interface_not_busy(dev, interface):
    """Detach kernel driver from a specific interface if it's active"""
    if dev.is_kernel_driver_active(interface.bInterfaceNumber):
        try:
            dev.detach_kernel_driver(interface.bInterfaceNumber)
        except usb.core.USBError as e:
            print(f"Could not detach kernel driver from interface({interface.bInterfaceNumber}): {e}", file=sys.stderr)
            sys.exit(1)


def request_data(is_fnb58_or_fnb48s, ep_out):
    """Send data request commands to the device"""
    # Setup communication with power meter
    ep_out.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
    ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    
    if is_fnb58_or_fnb48s:
        ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    else:
        ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


def decode_packet(data, state, calculate_crc, time_interval, alpha, csv_writer, verbose=False):
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
    # Data is 64 bytes (64 bytes of HID data minus vendor constant 0xaa)
    # First byte is HID vendor constant 0xaa
    # Second byte is payload type: 0x04 is data packet
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
        state.start_time = time.time()
    
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
        
        dp = (data[offset + 8] + data[offset + 9] * 256) / 1000
        dn = (data[offset + 10] + data[offset + 11] * 256) / 1000
        
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
        state.capacity += current * time_interval
        
        # Update max values
        state.max_power = max(state.max_power, power)
        state.max_current = max(state.max_current, current)
        state.max_voltage = max(state.max_voltage, voltage)
        
        # Calculate elapsed time for this sample
        t = t0 + i * time_interval
        elapsed = t - state.start_time
        
        # Increment sample counter
        state.samples_count += 1
        
        # Write data to CSV
        csv_writer.writerow([
            f"{t:.6f}",                  # Timestamp
            f"{elapsed:.6f}",            # Elapsed time
            f"{state.samples_count}",     # Sample number
            f"{voltage:.6f}",            # Voltage (V)
            f"{current:.6f}",            # Current (A)
            f"{power:.6f}",              # Power (W)
            f"{state.temp_ema:.3f}",     # Temperature (°C)
            f"{dp:.3f}",                 # D+ voltage
            f"{dn:.3f}",                 # D- voltage
            f"{state.energy:.6f}",       # Accumulated energy (Ws)
            f"{state.capacity:.6f}"      # Accumulated capacity (As)
        ])
    
    return True

def calculate_stddev(state):
    """Calculate standard deviation of a list of values"""
    if len(state.power_values) < 2:
        return 0.0
    return np.std(state.power_values, ddof=1)

def drain_endpoint(ep_in, timeout=1000, verbose=False):
    """
    Drain any pending data from the input endpoint
    
    Args:
        ep_in: Input endpoint object
        timeout: Read timeout in milliseconds
        verbose: Whether to print verbose info
    """
    try:
        while True:
            data = ep_in.read(size_or_buffer=64, timeout=timeout)
            if data and verbose:
                print(f"Drained {len(data)} bytes", file=sys.stderr)
    except usb.core.USBTimeoutError:
        # Timeout indicates no more data to read
        pass
    except Exception as e:
        print(f"Error while draining endpoint: {e}", file=sys.stderr)


def print_summary(state, duration):
    """Print a summary of the collected measurements"""
    if state.start_time is None:
        print("\nNo data was collected.", file=sys.stderr)
        return
    
    actual_duration = time.time() - state.start_time
    power_mean = np.mean(state.power_values)
    power_stddev = calculate_stddev(state)
    
    print("\n---------- Measurement Summary ----------", file=sys.stderr)
    print(f"Duration: {actual_duration:.2f} seconds", file=sys.stderr)
    print(f"Samples collected: {state.samples_count}", file=sys.stderr)
    print(f"Average power: {power_mean:.6f} W", file=sys.stderr)
    print(f"Maximum power: {state.max_power:.6f} W", file=sys.stderr)
    print(f"Power std deviation: {power_stddev:.6f} W", file=sys.stderr)
    print(f"Maximum current: {state.max_current:.6f} A", file=sys.stderr)
    print(f"Maximum voltage: {state.max_voltage:.6f} V", file=sys.stderr)
    print(f"Total energy: {state.energy:.6f} Ws ({state.energy / 3600:.6f} Wh)", file=sys.stderr)
    print(f"Total capacity: {state.capacity:.6f} As ({state.capacity / 3600:.6f} Ah)", file=sys.stderr)
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
    args = parser.parse_args()
    
    # List all USB devices if requested
    if args.list_devices:
        list_all_usb_devices()
        return 0
    
    # Find the device
    dev, is_fnb58_or_fnb48s, model_name = find_device()
    
    if not dev:
        print("Error: FNIRSI USB power meter not found. Check connection and permissions.", file=sys.stderr)
        print("Available USB devices:", file=sys.stderr)
        list_all_usb_devices()
        return 1
    
    print(f"Found {model_name} USB power meter", file=sys.stderr)
    
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
    try:
        ep_in, ep_out = setup_device(dev, args.verbose)
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
            "timestamp",
            "elapsed",
            "sample",
            "voltage_V",
            "current_A",
            "power_W",
            "temp_C",
            "dp_V",
            "dn_V",
            "energy_Ws",
            "capacity_As"
        ])
        
        # Request initial data from the device
        request_data(is_fnb58_or_fnb48s, ep_out)
        
        # Allow time for the device to respond
        time.sleep(0.1)
        
        # Set up data refresh timing
        refresh = 1.0 if is_fnb58_or_fnb48s else 0.003  # 1 s for FNB58 / FNB48S, 3 ms for others
        continue_time = time.time() + refresh
        
        # Calculate end time if duration is specified
        end_time = time.time() + args.duration if args.duration > 0 else None
        
        # Main measurement loop
        try:
            while True:
                # Check if duration has been reached
                if end_time and time.time() >= end_time:
                    print("Specified duration reached", file=sys.stderr)
                    break
                
                # Read data from the device
                try:
                    data = ep_in.read(size_or_buffer=64, timeout=5000)
                    
                    # Decode the data packet
                    decode_packet(data, state, crc_calculator, time_interval, args.alpha, csv_writer, args.verbose)
                    
                    # Request more data if it's time
                    if time.time() >= continue_time:
                        continue_time = time.time() + refresh
                        ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")
                    
                except usb.core.USBTimeoutError:
                    print("USB timeout occurred, retrying...", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Error reading data: {e}", file=sys.stderr)
                    break
                
                # Check for stop request (file or keyboard interrupt)
                if os.path.exists("fnirsi_stop"):
                    print("Stop file detected, terminating", file=sys.stderr)
                    break
        
        finally:
            # Drain any remaining data
            if args.verbose:
                print("Draining remaining data...", file=sys.stderr)
            drain_endpoint(ep_in, verbose=args.verbose)
            
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