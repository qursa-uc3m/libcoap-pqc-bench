#!/bin/bash

# Exit on error
set -e

INSTALL_TAG="v1.19.0"
#INSTALL_TAG="master"

# Check if wolfSSL is installed
if ! pkg-config --exists wolfssl; then
    echo "Error: wolfSSL is not installed. Please install wolfSSL first."
    exit 1
fi

# Clone wolfMQTT
if [ ! -d "wolfMQTT" ]; then
    git clone https://github.com/wolfSSL/wolfMQTT.git
fi

# Build wolfMQTT
cd wolfMQTT
if [ "$INSTALL_TAG" != "master" ]; then
    git checkout $INSTALL_TAG
fi
./autogen.sh
./configure --enable-sn
make

# Install wolfMQTT
sudo make install

# Update shared library cache
sudo ldconfig

echo "wolfMQTT has been built and installed successfully."