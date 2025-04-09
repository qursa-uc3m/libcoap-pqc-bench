# FNIRSI FNB58 Power Meter Setup Instructions

This guide provides step-by-step instructions for setting up and using the FNIRSI FNB58 USB power meter with the `energy_monitor.py` script for energy consumption measurements.

## Initial Setup

### 1. Install Required Dependencies

The script requires Python 3.6+ with the following packages:

```bash
# Create and activate a Python virtual environment (recommended)
python -m venv .bench-env
source .bench-env/bin/activate

# Install required packages
pip install hidapi crc pandas numpy
```

> Note: Our script uses `hidapi` instead of `pyusb`, which better handles the device as an HID device.

> Note: The necessary dependencies are included in the requirements.txt file, but it does not harm to check if they had been correctly installed

### 2. Set Up USB Permissions (One-time Setup)

To avoid permission issues when accessing the USB device, create a udev rule:

```bash
# Create a new udev rule file
sudo nano /etc/udev/rules.d/99-fnirsi.rules

# Add this line to the file
SUBSYSTEM=="usb", ATTRS{idVendor}=="2e3c", ATTRS{idProduct}=="5558", MODE="0666"

# Save and exit (Ctrl+O, Enter, Ctrl+X)

# Apply the new rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

After configuring the udev rule, disconnect and reconnect your FNB58 meter for the changes to take effect.

## Verifying the Setup

### 1. Check Device Detection

First, verify that your power meter is properly detected:

```bash
python energy_monitor.py --list-devices
```

You should see your FNB58 meter listed with VID/PID `2e3c:5558` and possibly its manufacturer and product name.

For a simpler check:

```bash
python energy_monitor.py --identify
```

This should display "Found FNB58 USB power meter" if the device is connected properly.

## Using the Script

### Basic Usage

Here are some common usage scenarios:

#### 1. Short Measurement Test (5 seconds)

```bash
# Create directory for results
mkdir -p ./bench-tests

# Run a 5-second test
python energy_monitor.py --duration 5 --output ./bench-tests/TEST1
```

This will create:
- `./bench-tests/TEST1_raw.csv` (detailed measurements)
- `./bench-tests/TEST1.csv` (summary data)

#### 2. Run with Verbose Output

```bash
python energy_monitor.py --duration 10 --verbose --output ./bench-tests/TEST2
```

#### 3. Continuous Monitoring (until Ctrl+C)

```bash
python energy_monitor.py --verbose --output ./bench-tests/continuous_test
```

### Command Line Options

```
usage: energy_monitor.py [-h] [--output OUTPUT] [--duration DURATION] [--crc] [--identify] [--list-devices] [--verbose] [--alpha ALPHA] [--force-reset] [--retry RETRY] [--merge ENERGY_FILE] [--benchmark BENCH_FILE]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Output file name (default: energy_data)
  --duration DURATION   Duration to collect data in seconds (0 = infinite) (default: 0)
  --crc                 Enable CRC checks for data integrity (default: False)
  --identify            Just identify the connected device and exit (default: False)
  --list-devices        List all USB devices and exit (default: False)
  --verbose             Enable verbose output (default: False)
  --alpha ALPHA         Set temperature smoothing factor (0-1) (default: 0.9)
  --force-reset         Force USB device reset before starting (default: False)
  --retry RETRY         Number of retry attempts for device operations (default: 3)
  --merge ENERGY_FILE   Merge energy data from ENERGY_FILE into benchmark CSV
  --benchmark BENCH_FILE Benchmark CSV file to merge energy data into
```

### Integrating with Benchmark Tests

To merge energy data with benchmark results:

```bash
python energy_monitor.py --merge ./bench-tests/TEST1.csv --benchmark ./bench-data/benchmark_results.csv
```

This will add energy-related columns (Power, Max Power, Energy) to the benchmark results file.

## Understanding the Output

The script generates two main files:

1. **Raw Data File** (`TEST1_raw.csv`):
   - Sample count
   - Timestamp
   - Elapsed time
   - Voltage (V)
   - Current (A)
   - Power (W)
   - Temperature (°C)
   - Energy (Ws)

2. **Summary File** (`TEST1.csv`):
   - Timestamp
   - Voltage
   - Current
   - Power
   - Temperature
   - Power (W)
   - Max Power (W)
   - Energy (Wh)
   - Standard deviations

The summary file is formatted with semicolon delimiters for compatibility with the benchmarking system.

## Troubleshooting

1. **Permission errors**: 
   - Ensure the udev rules are set up correctly
   - Reconnect the device after setting up the rules
   - Check that your user has proper permissions

2. **Device not found**:
   - Check if the meter appears in `--list-devices` output
   - Ensure the device is properly powered and connected
   - Try a different USB port

3. **Connection fails after multiple attempts**:
   - Try using the `--force-reset` option to reset the USB device
   - Increase the retry attempts using `--retry 5`
   - Physically disconnect and reconnect the device

4. **Inconsistent readings**:
   - Ensure nothing else is using the device
   - Check your USB cables for damage
   - Try enabling CRC checks with the `--crc` option