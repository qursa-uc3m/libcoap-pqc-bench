#!/bin/bash

# Default values
MODE="fork"  # Default mode
WOLFSSL_VERSION_TAG="main"
DEFAULT_RELEASE_TAG="v5.7.6-stable"
DTLS_VERSION="1.3"  # Can be "1.2" or "1.3"
DEBUG_MODE="yes"


# Function to display usage
usage() {
    echo "Usage: $0 [--fork | --release <version>]"
    echo "  --fork           Clone from dasobral/wolfssl-liboqs.git (default)"
    echo "  --release [ver]  Clone from wolfSSL/wolfssl.git with specified version (default: v5.7.6-stable)"
    echo "  -h, --help       Show this help message"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --fork)
      MODE="fork"
      WOLFSSL_VERSION_TAG="main"
      shift
      ;;
    --release)
      MODE="release"
      WOLFSSL_VERSION_TAG="${2:-$DEFAULT_RELEASE_TAG}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Installing missing packages for Raspberry Pi
sudo apt-get update
sudo apt-get install -y autoconf automake libtool coreutils bsdmainutils

# Prompt user for removal of existing installation
read -p "Do you want to remove existing wolfSSL installation? (y/n): " remove_existing
if [ "$remove_existing" == "y" ]; then
    echo "Removing existing wolfSSL libraries..."
    sudo find /usr/local/lib -type f \( -name 'libwolfssl.*' \) -exec rm {} + 2>/dev/null
    sudo find /usr/lib /usr/local/lib -name 'libcoap-3-wolfssl.so*' -exec rm {} + 2>/dev/null
    sudo find /usr/lib /usr/local/lib -name 'libwolfssl.so*' -exec rm {} + 2>/dev/null
    sudo find /usr/local/include -type d -name 'wolfssl' -exec rm -rf {} + 2>/dev/null
    echo "Existing installation removed."
else
    echo "Keeping existing installation..."
fi

echo "Removing existing wolfSSL repository..."
rm -rf ./wolfssl

# Clone appropriate repository based on mode
if [ "$MODE" == "fork" ]; then
  echo "Cloning from wolfSSL fork with OQS support..."
  git clone --branch $WOLFSSL_VERSION_TAG --depth 1 https://github.com/dasobral/wolfssl-liboqs.git wolfssl
else
  echo "Cloning from official wolfSSL release repository..."
  git clone --branch $WOLFSSL_VERSION_TAG --depth 1 https://github.com/wolfSSL/wolfssl.git wolfssl
fi

cd wolfssl

# Apply Dilithium patch if applicable
if [ "$WOLFSSL_VERSION_TAG" != "v5.6.4-stable" ]; then
  echo "Replacing '_ipd' with empty string in Dilithium files..."
  sed -i 's/_ipd//g' wolfcrypt/src/dilithium.c
  sed -i 's/_ipd//g' wolfssl/wolfcrypt/dilithium.h
fi

# Build wolfSSL
./autogen.sh
mkdir build
cd build

#WOLFSSL_FLAGS="--enable-all --enable-dtls --enable-experimental --with-liboqs --enable-kyber=ml-kem --disable-rpk"
#WOLFSSL_FLAGS="--enable-all --enable-dtls --enable-experimental --with-liboqs --disable-rpk --enable-kyber --enable-dilithium"
#WOLFSSL_FLAGS="--enable-all --enable-dtls --with-liboqs --enable-opensslall --enable-opensslextra --enable-kyber"
WOLFSSL_FLAGS="--enable-all --enable-dtls --with-liboqs --enable-experimental --enable-kyber --enable-dilithium --disable-rpk"

if [ "$DEBUG_MODE" == "yes" ]; then
    WOLFSSL_FLAGS="$WOLFSSL_FLAGS --enable-debug"
fi

if [ "$DTLS_VERSION" == "1.3" ]; then
    echo "Installing with DTLS 1.3 support"
    WOLFSSL_FLAGS="$WOLFSSL_FLAGS --enable-dtls13 --enable-dtls-frag-ch"
else
    echo "Installing with DTLS 1.2 support"
fi

../configure $WOLFSSL_FLAGS
make all -j$(nproc)

sudo make install
sudo ldconfig