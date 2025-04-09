#!/bin/bash

# Input Variables
while [[ "$1" == --* ]]; do
  case "$1" in
    --install)
      install_mode=true
      shift # Remove --install from the argument list
      ;;
    --name)
      vm_name="$2"
      shift 2 # Remove --name and the VM name from the argument list
      ;;
    *)
      echo "Usage: $0 [--install] [--name <vm_name>]"
      exit 1
      ;;
  esac
done

# If --name is not provided, prompt for the VM name
if [ -z "$vm_name" ]; then
  read -p "Enter the VM name: " vm_name
fi

# ISO path input
if [ "$install_mode" == true ]; then
	read -p "Enter the OS ISO file (full path): " os_iso
fi

# Check if the necessary packages are installed
check_packages() {
  echo "Checking required packages..."
  required_packages=("uml-utilities" "qemu-utils" "bridge-utils")
  
  for pkg in "${required_packages[@]}"; do
    dpkg-query -l "$pkg" &> /dev/null
    if [ $? -ne 0 ]; then
      echo "$pkg is not installed. Installing..."
      sudo apt update && sudo apt install -y "$pkg"
    else
      echo "$pkg is already installed."
    fi
  done
}

# Check if the ISO exists (only called if --install is used)
check_iso() {
  echo "Checking if the ISO file exists..."
  if [ ! -f "$os_iso" ]; then
    echo "ISO file does not exist at $os_iso. Exiting..."
    exit 1
  fi
  echo "ISO file $os_iso found."
}

# Check if the QCOW2 disk exists and is not empty
check_disk() {
  qcow2_disk="/var/lib/libvirt/images/${vm_name}.qcow2"
  if [ -f "$qcow2_disk" ]; then
    disk_size=$(du -sh "$qcow2_disk" | cut -f1)
    if [ "$disk_size" == "0" ]; then
      echo "Disk image $qcow2_disk exists but is empty. Exiting..."
      exit 1
    fi
  else
    echo "Disk image $qcow2_disk does not exist. Creating it ..."
    echo "Creating a 10GB disk image..."
    sudo qemu-img create -f qcow2 "$qcow2_disk" 10G 
    echo "Disk image created."
  fi
}

# Check if virbr0 exists
check_virbr0() {
  echo "Checking if bridge exists..."
  if ! ip link show br0 &> /dev/null; then
    echo "Custom br0 bridge does not exist. Creating it ..."
    sudo ip link add br0 type bridge
    sudo ip link set br0 up
    sudo ip link set enp0s31f6 master br0
    sudo dhclient br0 # Using DHCP to get an IP address for the bridge (gateway)
    echo "Custom bridge (br0) stablished."
  else
    echo "Custom (br0) bridge exists."
  fi
}

# Check if the TAP device exists and is linked to the bridge
check_tap_device() {
  echo "Checking if TAP device tap0 exists..."
  if ! ip link show tap0 &> /dev/null; then
    echo "TAP device does not exist. Creating it..."
    sudo ip tuntap add tap0 mode tap user $(whoami)
    sudo ip link set tap0 up
    sudo ip link set tap0 master br0
    echo "TAP device created and attached to br0."
  else
    echo "TAP device tap0 exists."
  fi
}

# Create the VM with QEMU/KVM for installation
install_vm() {
  echo "Starting the VM for OS installation with ISO $os_iso..."
  
  # Display the GRUB modification message
  echo ""
  echo "---------------------------------------"
  echo "IMPORTANT: During the installation process:"
  echo "Press 'e' to edit the boot parameters when you see the GRUB menu."
  echo "Modify the linux line to add 'console=ttyS0 text' at the end."
  echo "Example: linux /casper/vmlinuz ... console=ttyS0 text"
  echo "---------------------------------------"
  echo ""
  
  # Ask for confirmation before proceeding
  read -p "Do you want to continue with the installation? (y/n): " answer
  case $answer in
    [Yy]* ) 
      # User chose to continue
      echo "Proceeding with installation..."
      ;;
    [Nn]* )
      # User chose not to continue
      echo "Installation aborted."
      exit 1
      ;;
    * )
      # Invalid input
      echo "Invalid input. Please enter y or n."
      exit 1
      ;;
  esac
  
  # Add a simulated progress bar using pv for QEMU execution time (progress is just simulated here)
  echo "Starting OS installation. Please follow the installation process."
  echo "Starting QEMU for VM installation..."
  echo "Installing the VM. Follow the steps in the VM window."
  sudo qemu-system-x86_64 -enable-kvm \
    -m 2048 \
    -smp cpus=2 \
    -hda "/var/lib/libvirt/images/${vm_name}.qcow2" \
    -cdrom "$os_iso" \
    -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
    -device virtio-net-pci,netdev=net0 \
    -nographic \
    -serial mon:stdio \
    -boot d 
}

# Re-launch the VM without the ISO (after installation)
launch_vm() {
  echo "Launching VM for regular usage..."
  echo "VM running in headless mode. Interact with the VM through this terminal."
  sudo qemu-system-x86_64 -enable-kvm \
    -m 2048 \
    -smp cpus=2 \
    -hda "/var/lib/libvirt/images/${vm_name}.qcow2" \
    -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
    -device virtio-net-pci,netdev=net0 \
    -nographic -serial mon:stdio 
  echo "VM session ended."
}

# Main script execution
echo "Starting the VM setup process..."

check_packages
check_disk
check_virbr0
check_tap_device

# Run check_iso and install_vm only if the --install flag is set
if [ "$install_mode" == true ]; then
  check_iso
  install_vm
else
	# Ask for confirmation before proceeding
  read -p "Do you want to launch an existing machine? (y/n): " answer
  case $answer in
    [Yy]* ) 
      # User chose to continue
      echo "Proceeding with installation..."
      ;;
    [Nn]* )
      # User chose not to continue
      echo "Installation aborted."
      exit 1
      ;;
    * )
      # Invalid input
      echo "Invalid input. Please enter y or n."
      exit 1
      ;;
  esac
  launch_vm
fi

echo "VM setup process completed."
