#!/bin/bash

# Default values
INSTALL_DIR="/opt/openssl_dtls13"
SCRIPT_DIR=$(dirname $(readlink -f $0))  # get the full path of the script
DEBUG=0

# Read flags
while getopts p:d: flag
do
    case "${flag}" in
        p) INSTALL_DIR=${OPTARG};;
        d) DEBUG=${OPTARG};;
    esac
done

# Check if the directory exists and is not empty
if [ -d "$INSTALL_DIR" ] && [ "$(ls -A $INSTALL_DIR)" ]; then
  read -p "The directory $INSTALL_DIR already exists and is not empty. Do you want to remove its contents and continue? (y/n): " confirm
  if [ "$confirm" != "y" ]; then
    exit 1
  fi
  sudo rm -rf $INSTALL_DIR
else
  sudo mkdir -p $INSTALL_DIR
fi


sudo mkdir -p $INSTALL_DIR/.local
cd $INSTALL_DIR

# Clone OpenSSL and checkout DTLS 1.3 branch
echo "CLONING OPENSSL WITH DTLS 1.3 SUPPORT..."
sudo git clone -b feature/dtls-1.3 https://github.com/openssl/openssl.git
cd openssl

# If debug mode is set, modify for logging
if [ "$DEBUG" -eq 1 ]; then
  echo "Debug mode is set. Setting up logging..."
  BUILD_TYPE="--debug"
else
  BUILD_TYPE="--release"
  # Add any debug specific modifications here
fi

# Configure and build OpenSSL
echo "BUILDING OPENSSL WITH DTLS 1.3 SUPPORT..."
sudo ./Configure --prefix=$(echo $INSTALL_DIR/.local) $BUILD_TYPE 

read -p "OpenSSL configuration complete. Continue with make and install? (y/n): " confirm
if [ "$confirm" != "y" ]; then
  exit 1
fi
  
sudo make -j$(nproc)
sudo make install_sw
cd ..

# Copy the custom openssl.cnf file - use the one in the same directory
echo "Copying custom OpenSSL configuration..."
sudo mkdir -p $INSTALL_DIR/.local/ssl
if [ -f "$SCRIPT_DIR/openssl.cnf" ]; then
    sudo cp $SCRIPT_DIR/openssl.cnf $INSTALL_DIR/.local/ssl/openssl.cnf
    echo "Custom openssl.cnf copied from $SCRIPT_DIR"
else
    echo "WARNING: Custom openssl.cnf not found in $SCRIPT_DIR"
    # Copy from system OpenSSL if no custom config is provided
    sudo cp /etc/ssl/openssl.cnf $INSTALL_DIR/.local/ssl/openssl.cnf
fi

echo "--------------------------------------------------------------------------"
echo "OpenSSL with DTLS 1.3 support has been installed at $INSTALL_DIR"
echo "Next, run the install_liboqs_for_ossl-dtls13.sh script to add PQC support."
echo "--------------------------------------------------------------------------"