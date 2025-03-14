#!/bin/bash

# Source the environment setup script with dtls13 parameter
source $(pwd)/openssl_env.sh dtls13

# Default values
BUILD_DIR="$(pwd)/libcoap/build"  # Default libcoap build directory
MODE="both"  # Default mode - can be "server", "client", "verbose" or "both"
SIG_ALG="dilithium5"  # Default signature algorithm to use
verbose_level=0

# Parse command line arguments
while getopts b:m:a:v:V: flag
do
    case "${flag}" in
        b) BUILD_DIR=${OPTARG};;
        m) MODE=${OPTARG};;
        a) SIG_ALG=${OPTARG};;
        v) verbose_level=${OPTARG};;
        V) verbose_level=${OPTARG};;
    esac
done

# Print detailed environment information
echo "============== ENVIRONMENT DETAILS ================"
echo "OpenSSL binary: $(which openssl)"
echo "OpenSSL version: $(openssl version)"
echo "OpenSSL config: $OPENSSL_CONF"
echo "OpenSSL modules: $OPENSSL_MODULES"
echo "Provider path: $PROVIDER_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "=================================================="

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: libcoap build directory not found at $BUILD_DIR"
    echo "Please specify the correct build directory with -b option"
    exit 1
fi

# Set binary paths
COAP_SERVER="$BUILD_DIR/bin/coap-server"
COAP_CLIENT="$BUILD_DIR/bin/coap-client"

# Check if binaries exist based on mode
if [[ "$MODE" == "server" || "$MODE" == "both" ]]; then
    if [ ! -f "$COAP_SERVER" ]; then
        echo "ERROR: coap-server binary not found at $COAP_SERVER"
        exit 1
    fi
fi

if [[ "$MODE" == "client" || "$MODE" == "both" ]]; then
    if [ ! -f "$COAP_CLIENT" ]; then
        echo "ERROR: coap-client binary not found at $COAP_CLIENT"
        exit 1
    fi
fi

# Debug binary linking
echo "============== BINARY LINKING ================"
echo "Checking coap-client linking:"
ldd $COAP_CLIENT | grep -E 'libssl|libcrypto'

echo "Checking coap-server linking:"
ldd $COAP_SERVER | grep -E 'libssl|libcrypto'
echo "=============================================="

# Determine certificate paths
case "$SIG_ALG" in
    dilithium*)
        CERT_DIR="./certs/dilithium"
        if [ "$SIG_ALG" == "dilithium2" ]; then
            ROOT_CERT="${CERT_DIR}/dilithium2_root_cert.pem"
            ENTITY_CERT="${CERT_DIR}/dilithium2_entity_cert.pem"
            ENTITY_KEY="${CERT_DIR}/dilithium2_entity_key.pem"
        elif [ "$SIG_ALG" == "dilithium3" ]; then
            ROOT_CERT="${CERT_DIR}/dilithium3_root_cert.pem"
            ENTITY_CERT="${CERT_DIR}/dilithium3_entity_cert.pem"
            ENTITY_KEY="${CERT_DIR}/dilithium3_entity_key.pem"
        else
            ROOT_CERT="${CERT_DIR}/dilithium5_root_cert.pem"
            ENTITY_CERT="${CERT_DIR}/dilithium5_entity_cert.pem"
            ENTITY_KEY="${CERT_DIR}/dilithium5_entity_key.pem"
        fi
        ;;
    falcon*)
        CERT_DIR="./certs/falcon"
        if [ "$SIG_ALG" == "falcon512" ]; then
            ROOT_CERT="${CERT_DIR}/falcon_level1_root_cert.pem"
            ENTITY_CERT="${CERT_DIR}/falcon_level1_entity_cert.pem"
            ENTITY_KEY="${CERT_DIR}/falcon_level1_entity_key.pem"
        else
            ROOT_CERT="${CERT_DIR}/falcon_level5_root_cert.pem"
            ENTITY_CERT="${CERT_DIR}/falcon_level5_entity_cert.pem"
            ENTITY_KEY="${CERT_DIR}/falcon_level5_entity_key.pem"
        fi
        ;;
    sphincs*)
        CERT_DIR="./certs/sphincs"
        ROOT_CERT="${CERT_DIR}/sphincssha2128fsimple_root_cert.pem"
        ENTITY_CERT="${CERT_DIR}/sphincssha2128fsimple_entity_cert.pem"
        ENTITY_KEY="${CERT_DIR}/sphincssha2128fsimple_entity_key.pem"
        ;;
    *)
        # Default to RSA as fallback
        CERT_DIR="./certs/rsa"
        ROOT_CERT="${CERT_DIR}/rsa_2048_root_cert.pem"
        ENTITY_CERT="${CERT_DIR}/rsa_2048_entity_cert.pem"
        ENTITY_KEY="${CERT_DIR}/rsa_2048_entity_key.pem"
        ;;
esac

# Check if certificates exist
if [ ! -f "$ROOT_CERT" ] || [ ! -f "$ENTITY_CERT" ] || [ ! -f "$ENTITY_KEY" ]; then
    echo "ERROR: Certificates not found for algorithm $SIG_ALG"
    echo "Please run ./certs/generate_scripts.sh first"
    exit 1
fi

# Verify PQC support
if [[ "$MODE" = "verbose" ]]; then 
    echo "============== PQC ALGORITHM SUPPORT ================"
    $OPENSSL_BIN_DIR/openssl list -signature-algorithms -provider oqsprovider 2>/dev/null
    echo "--------------------------------------------------"
    $OPENSSL_BIN_DIR/openssl list -kem-algorithms -provider oqsprovider 2>/dev/null
    echo "====================================================="
fi

# Run the appropriate mode
if [ "$MODE" == "server" ]; then
    # Server mode - run the server in foreground
    echo "Starting CoAP server with DTLS 1.3 and $SIG_ALG (Press Ctrl+C to stop)..."
    # Run with explicit environment settings
    OPENSSL_CONF="$OPENSSL_CONF" \
    LD_LIBRARY_PATH="$OPENSSL_ROOT_DIR/lib64:$OPENSSL_ROOT_DIR/lib:$LD_LIBRARY_PATH" \
    OPENSSL_MODULES="$OPENSSL_MODULES" \
    exec $COAP_SERVER -A ::1 \
        -j $ENTITY_KEY \
        -c $ENTITY_CERT \
        -v $verbose_level -V $verbose_level
    
elif [ "$MODE" == "client" ]; then
    # Client mode - run just the client
    echo "Testing with CoAP client using $SIG_ALG..."
    # Run with explicit environment settings
    OPENSSL_CONF="$OPENSSL_CONF" \
    LD_LIBRARY_PATH="$OPENSSL_ROOT_DIR/lib64:$OPENSSL_ROOT_DIR/lib:$LD_LIBRARY_PATH" \
    OPENSSL_MODULES="$OPENSSL_MODULES" \
    $COAP_CLIENT \
        -R $ROOT_CERT \
        -m get coaps://[::1]/ \
        -v $verbose_level -V $verbose_level
    
else
    # "both" or "verbose" mode - explain the separate shell approach
    echo "------------------------------------------------------------"
    echo "DTLS 1.3 with PQC Test using $SIG_ALG"
    echo "------------------------------------------------------------"
    echo "To test properly, run the server and client in separate shells:"
    echo ""
    echo "1. Start the server in one shell:"
    echo "   $0 -m server -b $BUILD_DIR -a $SIG_ALG"
    echo ""
    echo "2. Then run the client in another shell:"
    echo "   $0 -m client -b $BUILD_DIR -a $SIG_ALG"
    echo ""
    echo "Available signature algorithms:"
    echo " - dilithium2, dilithium3, dilithium5"
    echo " - falcon512, falcon1024"
    echo " - sphincssha2128fsimple"
    echo ""
    echo "Certificate paths for $SIG_ALG:"
    echo "- Root CA: $ROOT_CERT"
    echo "- Entity certificate: $ENTITY_CERT"
    echo "- Entity key: $ENTITY_KEY"
    echo "------------------------------------------------------------"
fi

echo "Done."