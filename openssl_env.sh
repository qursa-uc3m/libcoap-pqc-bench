#!/bin/bash

mode=${1:-dtls13}  # Default to dtls13 if no parameter provided

if [ "$mode" == "dtls13" ]; then
    echo "Using DTLS 1.3 OpenSSL installation"
    echo "... Configuring OpenSSL environment correctly ..."
    
    # OpenSSL DTLS 1.3 paths
    export OPENSSL_DIR="/opt/openssl_dtls13"
    export OPENSSL_ROOT_DIR="$OPENSSL_DIR/.local"
    export OPENSSL_CONF="$OPENSSL_ROOT_DIR/ssl/openssl.cnf"
    
    # Use lib64 directory for libraries (as shown in your installation)
    export LD_LIBRARY_PATH="$OPENSSL_ROOT_DIR/lib64:$OPENSSL_ROOT_DIR/lib:$LD_LIBRARY_PATH"
    export PATH="$OPENSSL_ROOT_DIR/bin:$PATH"
    
    # Set modules directories
    export OPENSSL_MODULES="$OPENSSL_ROOT_DIR/lib64/ossl-modules"
    export OPENSSL_ENGINES="$OPENSSL_ROOT_DIR/lib64/engines-3"
    
    # For OQS provider - checking both possible locations
    if [ -f "$OPENSSL_ROOT_DIR/lib/ossl-modules/oqsprovider.so" ]; then
        export PROVIDER_PATH="$OPENSSL_ROOT_DIR/lib/ossl-modules"
    else
        export PROVIDER_PATH="$OPENSSL_MODULES"
    fi
    
    # Compiler and linker flags for building
    export CPPFLAGS="-I$OPENSSL_ROOT_DIR/include $CPPFLAGS"
    export LDFLAGS="-L$OPENSSL_ROOT_DIR/lib64 -Wl,-rpath,$OPENSSL_ROOT_DIR/lib64 $LDFLAGS"
    export PKG_CONFIG_PATH="$OPENSSL_ROOT_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH"
    
    # For debugging
    export OPENSSL_BIN_DIR="$OPENSSL_ROOT_DIR/bin"
    
    echo "OpenSSL installation path: $OPENSSL_DIR"
    echo "OpenSSL directory: $OPENSSL_ROOT_DIR" 
    echo "OpenSSL configuration: $OPENSSL_CONF"
    echo "OpenSSL modules: $OPENSSL_MODULES"
    echo "OQS provider path: $PROVIDER_PATH"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo ""
    $OPENSSL_ROOT_DIR/bin/openssl version
    $OPENSSL_ROOT_DIR/bin/openssl list -providers
else
    echo "Using system OpenSSL installation"
    unset OPENSSL_CONF
    unset OPENSSL_MODULES
    unset OPENSSL_ENGINES
    # Reset to default path
    export PATH="/usr/bin:$PATH"
fi