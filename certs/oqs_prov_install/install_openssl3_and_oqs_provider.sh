#!/bin/bash

# Default values
INSTALL_DIR="/opt/oqs_openssl3"
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

# Build OpenSSL 3.*
echo "BUILDING OPENSSL 3.*...."
#sudo git clone -b openssl-3.0.2 git://git.openssl.org/openssl.git
#sudo git clone git://git.openssl.org/openssl.git
sudo git clone https://github.com/openssl/openssl.git
cd openssl

# If debug mode is set, replace the rand_lib.c file
if [ "$DEBUG" -eq 1 ]; then
  echo "Debug mode is set. Replacing rand_lib.c file for logging."
  echo "Taking rand_lib.c from: $SCRIPT_DIR"
  sudo cp "$SCRIPT_DIR/rand_lib.c" ./crypto/rand/rand_lib.c
fi

sudo ./config --prefix=$(echo $INSTALL_DIR/.local)
sudo make 
sudo make install_sw
cd ..

# Build liboqs
echo "BUILDING LIBOQS...."
#sudo git clone -b 0.8.0-rc1 https://github.com/open-quantum-safe/liboqs.git || exit 1
sudo git clone https://github.com/open-quantum-safe/liboqs.git || exit 1
cd liboqs
#sudo git checkout 9f912c957bfe7f4b894aa9661168a310e8dd1a58 || exit 1
sudo cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/.local -S . -B _build || exit 1
sudo cmake --build _build || exit 1
sudo cmake --install _build || exit 1
cd ..


# Build the provider
echo "BUILDING OQS PROVIDER...."
sudo git clone https://github.com/open-quantum-safe/oqs-provider.git || exit 1
cd oqs-provider
#sudo git checkout 4bf202bdbe4a1c9dbb7e88ccd0636c9848d90afc || exit 1 # This is more recent and seems to fit well with the liboqs 0.8.0 release
#sudo git checkout c8cca2f8102805063db05071ec17acd453c4abc6 || exit 1
sudo cmake -DOPENSSL_ROOT_DIR=$INSTALL_DIR/.local -DCMAKE_PREFIX_PATH=$INSTALL_DIR/.local -S . -B _build || exit 1
sudo cmake --build _build || exit 1
cd ..

# Create hard link
echo "CREATING HARD LINK...."
sudo ln -f $INSTALL_DIR/.local/bin/openssl /usr/local/bin/oqs_openssl3

# Add to ~/.bashrc
if ! grep -q "export LD_LIBRARY_PATH=\"$INSTALL_DIR/.local/lib64:\$LD_LIBRARY_PATH\"" ~/.bashrc; then
    echo "EXPORTING LD_LIBRARY_PATH to ~/.bashrc ..."
    echo "# CUSTOM OPENSSL3 installation" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"$INSTALL_DIR/.local/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
fi