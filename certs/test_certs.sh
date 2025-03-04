#!/bin/bash

# Set OPENSSL environment
source "$(pwd)/openssl_env.sh"
echo ""
# Import certificate configuration
source "$(pwd)/certs/config_certs.sh"

# Function to display usage information
usage() {
    echo "Usage: $0 [--test-all | --test CONFIG_NAME | --setup CONFIG_NAME]"
    echo ""
    echo "Options:"
    echo "  --test-all          Test all certificate configurations"
    echo "  --test CONFIG_NAME  Test a specific certificate configuration"
    echo "  --setup CONFIG_NAME Set up symbolic links for a configuration"
    echo "  --list              List available certificate configurations"
    echo "  -h, --help          Show this help message"
    exit 1
}

# Function to test a certificate configuration
test_cert_config() {
    local config_name="$1"
    local paths
    
    echo "==== Testing certificate configuration: $config_name ===="
    echo ""
    
    # Validate certificate files
    if ! validate_cert_files "$config_name"; then
        echo "FAILED: Certificate validation for $config_name"
        return 1
    fi
    
    # Get the certificate paths
    paths=$(get_cert_paths "$config_name")
    IFS=';' read -r cert_path key_path ca_path <<< "$paths"
    
    # Check certificate details using OpenSSL
    echo "Certificate details ($cert_path):"
    openssl x509 -in "$cert_path" -text -noout | grep -E 'Subject:|Issuer:|Not Before:|Not After:|Public Key Algorithm:' | sed 's/^/    /'
    
    # Check private key details
    echo "Private key details ($key_path):"
    openssl pkey -in "$key_path" -text -noout | grep -E 'Private-Key:' | sed 's/^/    /'
    
    echo "CA certificate details ($ca_path):"
    openssl x509 -in "$ca_path" -text -noout | grep -E 'Subject:|Issuer:|Not Before:|Not After:|Public Key Algorithm:' | sed 's/^/    /'
    
    # Verify certificate chain
    echo "Verifying certificate chain..."
    if openssl verify -CAfile "$ca_path" "$cert_path" > /dev/null 2>&1; then
        echo "    Certificate chain verification: SUCCESS"
    else
        echo "    Certificate chain verification: FAILED"
        echo "    Details: $(openssl verify -CAfile "$ca_path" "$cert_path" 2>&1)"
        return 1
    fi
    
    echo "Configuration $config_name tested successfully"
    echo ""
    return 0
}

# Parse command-line arguments
if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    --test-all)
        echo "Testing all certificate configurations..."
        success=0
        failed=0
        failed_configs=()
        
        for config in "${!CERT_CONFIGS[@]}"; do
            if test_cert_config "$config"; then
                ((success++))
            else
                ((failed++))
                failed_configs+=("$config")
            fi
        done
        
        echo "=== Test Summary ==="
        echo "Successful: $success"
        echo "Failed: $failed"
        
        if [ $failed -gt 0 ]; then
            echo "Failed configurations:"
            for config in "${failed_configs[@]}"; do
                echo "  - $config"
            done
            exit 1
        fi
        ;;
    
    --test)
        if [ -z "$2" ]; then
            echo "Error: Missing configuration name"
            usage
        fi
        test_cert_config "$2"
        exit $?
        ;;
    
    --setup)
        if [ -z "$2" ]; then
            echo "Error: Missing configuration name"
            usage
        fi
        setup_cert_symlinks "$2"
        exit $?
        ;;
    
    --list)
        list_cert_configs
        ;;
    
    -h|--help)
        usage
        ;;
    
    *)
        echo "Error: Unknown option $1"
        usage
        ;;
esac