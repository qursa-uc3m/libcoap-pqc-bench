#!/bin/bash

# Source environment to get the paths
source $(pwd)/openssl_env.sh dtls13

# Check if OpenSSL configuration exists
if [ ! -f "$OPENSSL_CONF" ]; then
    echo "ERROR: OpenSSL configuration file not found at $OPENSSL_CONF"
    exit 1
fi

echo "Checking OQS provider configuration in $OPENSSL_CONF..."

# Find oqsprovider.so location
if [ -f "$OPENSSL_ROOT_DIR/lib/ossl-modules/oqsprovider.so" ]; then
    OQS_PATH="$OPENSSL_ROOT_DIR/lib/ossl-modules/oqsprovider.so"
elif [ -f "$OPENSSL_ROOT_DIR/lib64/ossl-modules/oqsprovider.so" ]; then
    OQS_PATH="$OPENSSL_ROOT_DIR/lib64/ossl-modules/oqsprovider.so"
else
    OQS_PATH=$(find $OPENSSL_ROOT_DIR -name "oqsprovider.so" -type f | head -1)
    if [ -z "$OQS_PATH" ]; then
        echo "ERROR: Cannot find oqsprovider.so in $OPENSSL_ROOT_DIR"
        exit 1
    fi
fi

echo "Found OQS provider at: $OQS_PATH"

# Check if provider is correctly configured
if grep -q "oqsprovider.so" "$OPENSSL_CONF"; then
    CURRENT_PATH=$(grep -A 5 "\[oqsprovider_section\]" "$OPENSSL_CONF" | grep "module" | head -1 | sed 's/.*= *//')
    echo "Current OQS provider path in config: $CURRENT_PATH"
    
    if [ "$CURRENT_PATH" != "$OQS_PATH" ]; then
        echo "WARNING: OQS provider path in config does not match the actual path"
        echo "Would you like to update it? (y/n)"
        read UPDATE_CONF
        
        if [ "$UPDATE_CONF" == "y" ]; then
            # Make a backup
            sudo cp "$OPENSSL_CONF" "${OPENSSL_CONF}.bak"
            # Update the path
            sudo sed -i "s|module = .*oqsprovider.so|module = $OQS_PATH|g" "$OPENSSL_CONF"
            echo "Updated configuration file. Original saved as ${OPENSSL_CONF}.bak"
        fi
    else
        echo "OQS provider correctly configured in $OPENSSL_CONF"
    fi
else
    echo "WARNING: OQS provider not found in OpenSSL configuration"
    echo "Would you like to add it? (y/n)"
    read ADD_CONF
    
    if [ "$ADD_CONF" == "y" ]; then
        # Make a backup
        sudo cp "$OPENSSL_CONF" "${OPENSSL_CONF}.bak"
        
        # Add provider configuration
        sudo tee -a "$OPENSSL_CONF" > /dev/null << EOF

# OQS Provider Configuration
[provider_sect]
oqsprovider = oqsprovider_section
default = default_sect

[oqsprovider_section]
activate = 1
module = $OQS_PATH

[default_sect]
activate = 1
EOF
        echo "Added OQS provider configuration. Original saved as ${OPENSSL_CONF}.bak"
    fi
fi

# Test if OpenSSL can use the OQS provider
echo "Testing OQS provider loading..."
$OPENSSL_ROOT_DIR/bin/openssl list -providers -verbose

echo "Testing for PQC algorithms..."
$OPENSSL_ROOT_DIR/bin/openssl list -signature-algorithms -provider oqsprovider 2>/dev/null

if [ $? -eq 0 ]; then
    echo "SUCCESS: OQS provider loaded and working correctly"
else
    echo "ERROR: Could not load OQS provider correctly"
    echo "Please check the OpenSSL configuration and installation"
fi