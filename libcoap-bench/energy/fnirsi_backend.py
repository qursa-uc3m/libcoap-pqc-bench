#!/usr/bin/env python3
"""FNIRSI USB power meter energy monitoring backend."""

import os
import sys
import time
import csv
import subprocess
import numpy as np
from datetime import datetime

from .base import (MeasurementState, quit_flag, prepare_to_quit,
                   save_energy_summary, print_summary, calculate_stddev)

try:
    import hid
    HID_AVAILABLE = True
except ImportError:
    HID_AVAILABLE = False

DEVICE_IDS = {
    "FNB48": {"VID": 0x0483, "PID": 0x003A},
    "C1": {"VID": 0x0483, "PID": 0x003B},
    "FNB58": {"VID": 0x2E3C, "PID": 0x5558},
    "FNB48S": {"VID": 0x2E3C, "PID": 0x0049}
}


def check_availability() -> bool:
    return HID_AVAILABLE


def reset_usb_device(vid, pid, verbose=False) -> bool:
    """Attempt to reset the USB device using system commands."""
    vid_str, pid_str = f"{vid:04x}", f"{pid:04x}"
    
    try:
        # Try usbreset utility
        result = subprocess.run(["which", "usbreset"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            result = subprocess.run(["sudo", "usbreset", f"{vid_str}:{pid_str}"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return True
        
        # Try sysfs method
        cmd = ["lsusb", "-d", f"{vid_str}:{pid_str}"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return False
        
        output = result.stdout.strip()
        parts = output.split()
        if len(parts) < 6:
            return False
        
        bus, device = parts[1], parts[3][:-1]
        path = f"/sys/bus/usb/devices/{bus}-{device}/authorized"
        
        if not os.path.exists(path):
            path = f"/sys/bus/usb/devices/{bus}-{device[:-1]}.{device[-1]}/authorized"
            if not os.path.exists(path):
                return False
        
        try:
            with open(path, 'w') as f:
                f.write("0")
            time.sleep(0.5)
            with open(path, 'w') as f:
                f.write("1")
            time.sleep(1.0)
            return True
        except (IOError, PermissionError):
            return False
            
    except Exception:
        return False


def list_all_usb_devices():
    """List all HID devices connected to the system."""
    if not HID_AVAILABLE:
        print("Error: hidapi not available", file=sys.stderr)
        return False
    
    print("HID Devices:")
    devices = hid.enumerate()
    for device in devices:
        manufacturer = device.get('manufacturer_string', 'Unknown')
        product = device.get('product_string', 'Unknown')
        
        for name, ids in DEVICE_IDS.items():
            if device['vendor_id'] == ids["VID"] and device['product_id'] == ids["PID"]:
                if manufacturer == 'Unknown':
                    manufacturer = "FNIRSI"
                if product == 'Unknown':
                    product = name
        
        print(f"  Device: {device['vendor_id']:04x}:{device['product_id']:04x} {manufacturer} {product}")
    return True


def find_device():
    """Find a compatible FNIRSI power meter device."""
    if not HID_AVAILABLE:
        return None, False, "Unknown", None
    
    devices = hid.enumerate()
    for device in devices:
        for name, ids in DEVICE_IDS.items():
            if device['vendor_id'] == ids["VID"] and device['product_id'] == ids["PID"]:
                return device['path'], name in ["FNB58", "FNB48S"], name, ids
    return None, False, "Unknown", None


def setup_device(device_path, max_attempts=3, verbose=False):
    """Set up the HID device for communication with retry logic."""
    device = hid.device()
    attempt = 0
    
    while attempt < max_attempts:
        try:
            device.open_path(device_path)
            device.set_nonblocking(1)
            try:
                device.write([0, 0x00] + [0x00] * 62)
                time.sleep(0.01)
            except:
                pass
            return device
        except IOError as e:
            attempt += 1
            time.sleep(attempt * 1.0)
    
    print(f"Error opening device after {max_attempts} attempts", file=sys.stderr)
    sys.exit(1)


def request_data(is_fnb58_or_fnb48s, device):
    """Send data request commands to the device."""
    try:
        device.write([0, 0xaa, 0x81] + [0x00] * 61 + [0x8e])
        time.sleep(0.02)
        device.write([0, 0xaa, 0x82] + [0x00] * 61 + [0x96])
        time.sleep(0.02)
        
        if is_fnb58_or_fnb48s:
            device.write([0, 0xaa, 0x82] + [0x00] * 61 + [0x96])
        else:
            device.write([0, 0xaa, 0x83] + [0x00] * 61 + [0x9e])
        return True
    except IOError:
        return False


def read_data(device, timeout=1000):
    """Read data from device with timeout."""
    start_time = time.time()
    while (time.time() - start_time) * 1000 < timeout:
        try:
            data = device.read(64)
            if data and len(data) > 0:
                return data
        except IOError:
            return None
        time.sleep(0.001)
    return None


def drain_endpoint(device, timeout=100):
    """Drain any pending data from the HID device."""
    start_time = time.time()
    while (time.time() - start_time) * 1000 < timeout:
        try:
            data = device.read(64)
            if not data or len(data) == 0:
                break
        except:
            break
        time.sleep(0.001)


def close_device_safely(device, verbose=False):
    """Close the HID device safely with proper cleanup."""
    if device:
        try:
            try:
                device.write([0, 0xaa, 0x00] + [0x00] * 62)
                time.sleep(0.1)
            except:
                pass
            drain_endpoint(device)
            device.close()
            time.sleep(0.5)
        except:
            pass


def setup_crc():
    """Set up CRC calculator if crc module is available."""
    try:
        import crc
        width, poly, init_value = 8, 0x39, 0x42
        final_xor_value, reverse_input, reverse_output = 0x00, False, False
        configuration = crc.Configuration(width, poly, init_value, final_xor_value, reverse_input, reverse_output)
        
        if hasattr(crc, "CrcCalculator"):
            crc_calculator = crc.CrcCalculator(configuration, use_table=True)
            return crc_calculator.calculate_checksum
        else:
            calculator = crc.Calculator(configuration, optimized=True)
            return calculator.checksum
    except ImportError:
        return None


def decode_packet(data, state, calculate_crc, time_interval, alpha, csv_writer, end_time=None, verbose=False):
    """Decode a data packet and update measurement state."""
    if len(data) < 2 or data[1] != 0x04:
        return False
    
    if calculate_crc:
        actual_checksum = data[-1]
        expected_checksum = calculate_crc(bytearray(data[1:-1]))
        if actual_checksum != expected_checksum:
            return False
    
    if state.start_time is None:
        state.start_time = time.time() - 4 * time_interval
    
    t0 = time.time() - 4 * time_interval
    
    for i in range(4):
        offset = 2 + 15 * i
        
        voltage = (data[offset + 3] * 256 * 256 * 256 + data[offset + 2] * 256 * 256 +
                  data[offset + 1] * 256 + data[offset + 0]) / 100000
        current = (data[offset + 7] * 256 * 256 * 256 + data[offset + 6] * 256 * 256 +
                  data[offset + 5] * 256 + data[offset + 4]) / 100000
        temp_C = (data[offset + 13] + data[offset + 14] * 256) / 10.0
        
        if state.temp_ema is not None:
            state.temp_ema = temp_C * (1.0 - alpha) + state.temp_ema * alpha
        else:
            state.temp_ema = temp_C
        
        power = voltage * current
        state.power_values.append(power)
        state.energy += power * time_interval
        
        state.max_power = max(state.max_power, power)
        state.max_current = max(state.max_current, current)
        state.max_voltage = max(state.max_voltage, voltage)
        
        t = t0 + i * time_interval
        elapsed = t - state.start_time
        state.last_elapsed_time = elapsed

        if end_time and t > end_time:
            return True
        
        state.samples_count += 1
        csv_writer.writerow([f"{state.samples_count}", f"{t:.6f}", f"{elapsed:.6f}",
                            f"{voltage:.6f}", f"{current:.6f}", f"{power:.6f}",
                            f"{state.temp_ema:.3f}", f"{state.energy:.6f}"])
    
    return True


def run(args) -> int:
    """Run FNIRSI energy monitoring."""
    import energy.base as base
    
    if not HID_AVAILABLE:
        print("Error: hidapi not installed. Install with: pip install hidapi", file=sys.stderr)
        return 1
    
    if args.list_devices:
        list_all_usb_devices()
        return 0
    
    device_path, is_fnb58_or_fnb48s, model_name, device_info = find_device()
    
    if not device_path:
        print("Error: FNIRSI USB power meter not found.", file=sys.stderr)
        list_all_usb_devices()
        return 1
    
    print(f"Found {model_name} USB power meter", file=sys.stderr)
    
    if args.force_reset and device_info:
        print("Attempting USB device reset...", file=sys.stderr)
        reset_usb_device(device_info["VID"], device_info["PID"], args.verbose)
        time.sleep(5)
        device_path, is_fnb58_or_fnb48s, model_name, device_info = find_device()
        if not device_path:
            print("Error: Device not found after reset.", file=sys.stderr)
            return 1
    
    if args.identify:
        return 0
    
    crc_calculator = setup_crc() if args.crc else None
    
    device = setup_device(device_path, args.retry, args.verbose)
    
    output_file = args.output + "_raw.csv"
    print(f"Data will be saved to: {output_file}", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    state = MeasurementState()
    sps = 100
    time_interval = 1.0 / sps
    
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["sample", "timestamp", "elapsed", "voltage_V", "current_A", "power_W", "temp_C", "energy_Ws"])
        
        if not request_data(is_fnb58_or_fnb48s, device):
            close_device_safely(device, args.verbose)
            return 1
        
        time.sleep(0.05)
        
        refresh = 1.0 if is_fnb58_or_fnb48s else 0.003
        continue_time = time.time() + refresh
        end_time = time.time() + args.duration - 0.03 if args.duration > 0 else None
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        start_pipe_signaled = False
        
        try:
            while not base.quit_flag:
                if base.prepare_to_quit:
                    print("Preparing to quit...", file=sys.stderr)
                    if args.stop_pipe and os.path.exists(args.stop_pipe):
                        try:
                            with open(args.stop_pipe, 'w') as f:
                                f.write("DONE\n")
                        except:
                            pass
                    break
                
                if end_time and time.time() >= end_time:
                    print("Duration reached", file=sys.stderr)
                    break
                
                try:
                    data = read_data(device, timeout=5000)
                    if data:
                        valid = decode_packet(data, state, crc_calculator, time_interval, args.alpha, csv_writer, end_time, args.verbose)
                        if valid:
                            consecutive_errors = 0
                            if not start_pipe_signaled and args.start_pipe and os.path.exists(args.start_pipe):
                                try:
                                    with open(args.start_pipe, 'w') as f:
                                        f.write("READY\n")
                                    start_pipe_signaled = True
                                except:
                                    pass
                        else:
                            consecutive_errors += 1
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print("Too many consecutive errors.", file=sys.stderr)
                            break
                    
                    if time.time() >= continue_time:
                        continue_time = time.time() + refresh
                        device.write([0, 0xaa, 0x83] + [0x00] * 61 + [0x9e])
                        
                except Exception:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        break

                if os.path.exists("fnirsi_stop"):
                    break
                    
        finally:
            close_device_safely(device, args.verbose)
            print_summary(state, args.duration, is_codecarbon=False)
            summary_file = save_energy_summary(state, args.output + ".csv", is_codecarbon=False)
            
            with open(f"{args.output}.csv.done", 'w') as f:
                f.write(f"Completed at {datetime.now().isoformat()}\n")
            
            if args.stop_pipe and os.path.exists(args.stop_pipe) and not base.prepare_to_quit:
                try:
                    with open(args.stop_pipe, 'w') as f:
                        f.write("ERROR\n")
                except:
                    pass
    
    return 0
