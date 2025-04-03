#!/bin/bash

# Determine base directory based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CERT_BASE_DIR="${REPO_ROOT}/certs"

# Certificate mapping configuration
declare -A CERT_CONFIGS=(
    # RSA certificates
    ["RSA_2048"]="${CERT_BASE_DIR}/rsa/rsa_2048_entity_cert.pem;${CERT_BASE_DIR}/rsa/rsa_2048_entity_key.pem;${CERT_BASE_DIR}/rsa/rsa_2048_root_cert.pem"
    
    # Dilithium certificates - different security levels
    ["DILITHIUM_LEVEL2"]="${CERT_BASE_DIR}/dilithium/dilithium2_entity_cert.pem;${CERT_BASE_DIR}/dilithium/dilithium2_entity_key.pem;${CERT_BASE_DIR}/dilithium/dilithium2_root_cert.pem"
    ["DILITHIUM_LEVEL3"]="${CERT_BASE_DIR}/dilithium/dilithium3_entity_cert.pem;${CERT_BASE_DIR}/dilithium/dilithium3_entity_key.pem;${CERT_BASE_DIR}/dilithium/dilithium3_root_cert.pem"
    ["DILITHIUM_LEVEL5"]="${CERT_BASE_DIR}/dilithium/dilithium5_entity_cert.pem;${CERT_BASE_DIR}/dilithium/dilithium5_entity_key.pem;${CERT_BASE_DIR}/dilithium/dilithium5_root_cert.pem"
    
    # Falcon certificates - different security levels
    ["FALCON_LEVEL1"]="${CERT_BASE_DIR}/falcon/falcon_level1_entity_cert.pem;${CERT_BASE_DIR}/falcon/falcon_level1_entity_key.pem;${CERT_BASE_DIR}/falcon/falcon_level1_root_cert.pem"
    ["FALCON_LEVEL5"]="${CERT_BASE_DIR}/falcon/falcon_level5_entity_cert.pem;${CERT_BASE_DIR}/falcon/falcon_level5_entity_key.pem;${CERT_BASE_DIR}/falcon/falcon_level5_root_cert.pem"
    
    # Elliptic Curve certificates
    ["EC_P256"]="${CERT_BASE_DIR}/ec/p256_entity_cert.pem;${CERT_BASE_DIR}/ec/p256_entity_key.pem;${CERT_BASE_DIR}/ec/p256_root_cert.pem"
    ["EC_ED25519"]="${CERT_BASE_DIR}/ec/ed25519_entity_cert.pem;${CERT_BASE_DIR}/ec/ed25519_entity_key.pem;${CERT_BASE_DIR}/ec/ed25519_root_cert.pem"

    # Legacy mapping for backward compatibility
    #["DEFAULT"]="${CERT_BASE_DIR}/server_cert.pem;${CERT_BASE_DIR}/server_key.pem;${CERT_BASE_DIR}/root_cert.pem"
)

# Function to list available certificate configurations
list_cert_configs() {
    echo "Available certificate configurations:"
    echo "-----------------------------------"
    
    # Define the order for displaying certificate types
    declare -a order=("RSA_" "EC_" "DILITHIUM_" "FALCON_")
    
    # List certificates in the defined order
    for prefix in "${order[@]}"; do
        for key in "${!CERT_CONFIGS[@]}"; do
            # Skip DEFAULT since it's not a real certificate type
            if [[ "$key" == "DEFAULT" ]]; then
                continue
            fi
            
            # Check if the current key starts with the current prefix
            if [[ "$key" == ${prefix}* ]]; then
                echo "  $key"
            fi
        done
    done
    
    # Show DEFAULT at the end if needed
    if [[ -n "${CERT_CONFIGS[DEFAULT]}" ]]; then
        echo "  DEFAULT (for backward compatibility)"
    fi
}

# Function to get certificate paths for a given configuration
# Returns: cert_path;key_path;ca_path
get_cert_paths() {
    local config_name="$1"
    
    if [[ -z "$config_name" ]]; then
        config_name="DEFAULT"
    fi
    
    # Check if configuration exists
    if [[ -z "${CERT_CONFIGS[$config_name]}" ]]; then
        echo "ERROR: Certificate configuration '$config_name' not found." >&2
        list_cert_configs >&2
        return 1
    fi
    
    echo "${CERT_CONFIGS[$config_name]}"
    return 0
}

# Function to validate certificate files
validate_cert_files() {
    local cert_config="$1"
    local paths
    
    # Get the certificate paths
    paths=$(get_cert_paths "$cert_config")
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Parse the paths
    IFS=';' read -r cert_path key_path ca_path <<< "$paths"
    
    # Check if files exist and are readable
    if [ ! -r "$cert_path" ]; then
        echo "ERROR: Certificate file '$cert_path' does not exist or is not readable." >&2
        return 1
    fi
    
    if [ ! -r "$key_path" ]; then
        echo "ERROR: Key file '$key_path' does not exist or is not readable." >&2
        return 1
    fi
    
    if [ ! -r "$ca_path" ]; then
        echo "ERROR: CA certificate file '$ca_path' does not exist or is not readable." >&2
        return 1
    fi
    
    return 0
}

# Function to create symbolic links for backward compatibility
setup_cert_symlinks() {
    local cert_config="$1"
    local paths
    
    # Get the certificate paths
    paths=$(get_cert_paths "$cert_config")
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Parse the paths
    IFS=';' read -r cert_path key_path ca_path <<< "$paths"
    
    # Create symlinks in the root certificate directory
    ln -sf "$cert_path" "${CERT_BASE_DIR}/server_cert.pem"
    ln -sf "$key_path" "${CERT_BASE_DIR}/server_key.pem"
    ln -sf "$ca_path" "${CERT_BASE_DIR}/root_cert.pem"
    
    echo "Created symbolic links for $cert_config certificate configuration"
    return 0
}

# If this script is run directly, show available configurations
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "$1" == "--list" ]]; then
        list_cert_configs
        exit 0
    elif [[ "$1" == "--validate" && -n "$2" ]]; then
        validate_cert_files "$2"
        exit $?
    elif [[ "$1" == "--setup" && -n "$2" ]]; then
        setup_cert_symlinks "$2"
        exit $?
    else
        echo "Usage: $0 [--list | --validate CONFIG_NAME | --setup CONFIG_NAME]"
        echo ""
        echo "Options:"
        echo "  --list                List available certificate configurations"
        echo "  --validate CONFIG     Validate certificate files for CONFIG"
        echo "  --setup CONFIG        Create symbolic links for CONFIG"
        exit 1
    fi
fi