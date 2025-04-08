import usb.core
import usb.util

# Define all possible FNIRSI device IDs
DEVICE_IDS = {
    "FNB48": {"VID": 0x0483, "PID": 0x003A},
    "C1": {"VID": 0x0483, "PID": 0x003B},
    "FNB58": {"VID": 0x2E3C, "PID": 0x5558},
    "FNB48S": {"VID": 0x2E3C, "PID": 0x0049}
}

# Try to find any of the FNIRSI devices
dev = None
model_name = "Unknown"

for name, ids in DEVICE_IDS.items():
    dev = usb.core.find(idVendor=ids["VID"], idProduct=ids["PID"])
    if dev:
        model_name = name
        print(f"Found {model_name} device (ID {ids['VID']:04x}:{ids['PID']:04x})")
        break

if dev is None:
    print("Device not found")
    exit(1)
    
print("Device found! Attempting to read configuration...")

# Try to read a configuration
try:
    print(f"Device has {dev.bNumConfigurations} configuration(s)")
    
    # Try to get manufacturer and product strings
    try:
        manufacturer = usb.util.get_string(dev, dev.iManufacturer)
        product = usb.util.get_string(dev, dev.iProduct)
        print(f"Manufacturer: {manufacturer}")
        print(f"Product: {product}")
    except Exception as e:
        print(f"Could not read strings: {e}")
    
    print("Success! You have read access to the device")
except usb.core.USBError as e:
    print(f"Access error: {e}")
    print("You might need proper permissions")