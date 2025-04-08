import hid
import time

# Define all possible FNIRSI device IDs
DEVICE_IDS = {
    "FNB58": {"VID": 0x2E3C, "PID": 0x5558}
}

# Try to find any of the FNIRSI devices
dev = None
model_name = "Unknown"

for name, ids in DEVICE_IDS.items():
    try:
        # Open the device using VID and PID
        dev = hid.device()
        dev.open(ids["VID"], ids["PID"])
        model_name = name
        print(f"Found {model_name} device (ID {ids['VID']:04x}:{ids['PID']:04x})")
        break
    except OSError as e:
        print(f"Error opening device: {e}")
        continue

if dev is None:
    print("Device not found")
    exit(1)

print("Device found! Attempting to read information...")

# Try to read information from the device
try:
    # Get the manufacturer and product strings
    manufacturer = dev.get_manufacturer_string()
    product = dev.get_product_string()
    print(f"Manufacturer: {manufacturer}")
    print(f"Product: {product}")

    # Optionally, attempt to read raw data from the device with timeout
    start_time = time.time()
    timeout = 5  # Timeout after 5 seconds
    data = None

    while time.time() - start_time < timeout:
        data = dev.read(64)  # Try to read 64 bytes
        if data:
            print(f"Data received: {data}")
            break
        else:
            print("No data received, retrying...")

    if data is None:
        print("Timeout reached, no data received from the device.")

    print("Success! You have read access to the device.")
except Exception as e:
    print(f"Error reading from device: {e}")
finally:
    # Close the device after interaction
    dev.close()
