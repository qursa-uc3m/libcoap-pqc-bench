#!/bin/bash

# Default values
INSTALL_DIR="/opt/openssl_dtls13"
SCRIPT_DIR=$(dirname $(readlink -f $0))
DEBUG=0

# Use the liboqs version compatible with OQS provider 0.7.0 (apparently linked in feature/dtls-1.3 branch)
LIBOQS_VERSION="0.11.0"  

# Read flags
while getopts p:d:v: flag
do
    case "${flag}" in
        p) INSTALL_DIR=${OPTARG};;
        d) DEBUG=${OPTARG};;
        v) LIBOQS_VERSION=${OPTARG};;
    esac
done

# Ensure version alignment 

if [ "$LIBOQS_VERSION" == "0.11.0" ]; then
    OQSPROV_VERSION="0.7.0"
elif [ "$LIBOQS_VERSION" == "0.12.0" ]; then
    OQSPROV_VERSION="0.8.0"
elif [ "$LIBOQS_VERSION" == "0.10.0" ]; then 
    OQSPROV_VERSION="0.6.0"
else
    echo "WARNING: Too outdated liboqs version."
    exit 1
fi

# Check if OpenSSL DTLS 1.3 is already installed
if [ ! -d "$INSTALL_DIR/.local" ]; then
    echo "ERROR: OpenSSL DTLS 1.3 installation not found at $INSTALL_DIR/.local"
    echo "Please run the install-openssl-dtls-1.3.sh script first."
    exit 1
fi

# Create liboqs and oqs-provider directories
LIBOQS_DIR="$INSTALL_DIR/liboqs"
OQSPROV_DIR="$INSTALL_DIR/oqs-provider"

sudo rm -rf $OQSPROV_DIR
sudo rm -rf $LIBOQS_DIR

sudo mkdir -p $LIBOQS_DIR
sudo mkdir -p $OQSPROV_DIR

# Build liboqs with OpenSSL support
echo "--------------------------------------------------------------------------"
echo "BUILDING LIBOQS -$LIBOQS_VERSION- WITH OPENSSL SUPPORT ..."
echo "--------------------------------------------------------------------------"
echo "Moving to $INSTALL_DIR ..."
cd $INSTALL_DIR
sudo git clone -b $LIBOQS_VERSION https://github.com/open-quantum-safe/liboqs.git $LIBOQS_DIR || exit 1
cd $LIBOQS_DIR

# Configure liboqs to use the OpenSSL DTLS 1.3 installation
sudo cmake \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/.local \
    -DOQS_USE_OPENSSL=ON \
    -DOPENSSL_ROOT_DIR=$INSTALL_DIR/.local \
    -S . -B _build || exit 1
    
sudo cmake --build _build -- -j$(nproc) || exit 1
sudo cmake --install _build || exit 1

#export liboqs_DIR=$INSTALL_DIR/.local
echo "--------------------------------------------------------------------------"
echo "liboqs has been installed with OpenSSL DTLS 1.3 support in: $INSTALL_DIR/.local"

cd ..

# Build OQS Provider
echo "--------------------------------------------------------------------------"
echo "BUILDING OQS PROVIDER -$OQSPROV_VERSION- FOR OPENSSL DTLS 1.3 ..."
echo "--------------------------------------------------------------------------"
sudo git clone -b $OQSPROV_VERSION https://github.com/open-quantum-safe/oqs-provider.git $OQSPROV_DIR || exit 1
cd $OQSPROV_DIR

# Configure OQS Provider to use OpenSSL DTLS 1.3
sudo cmake \
    -DOPENSSL_ROOT_DIR=$INSTALL_DIR/.local \
    -DCMAKE_PREFIX_PATH=$INSTALL_DIR/.local \
    -S . -B _build || exit 1
    
sudo cmake --build _build -- -j$(nproc) || exit 1

# Copy the provider to the OpenSSL modules directory
sudo mkdir -p $INSTALL_DIR/.local/lib/ossl-modules
sudo cp _build/lib/oqsprovider.so $INSTALL_DIR/.local/lib/ossl-modules/

echo "--------------------------------------------------------------------------"
echo "liboqs and OQS provider have been installed for OpenSSL DTLS 1.3"
echo "You can now build libcoap with PQC and DTLS 1.3 support running openssl_env.sh && install_libcoap.sh."
echo "--------------------------------------------------------------------------"
