#!/bin/bash

# pskeys/psk_manager.sh
# A simple script to manage PSK keys for CoAP benchmarking

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PSK_DIR="${SCRIPT_DIR}"
ACTIVE_PSK="${PSK_DIR}/active_psk.txt"

# Color codes for better output readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Raspberry Pi SSH configuration
RASPBERRY_PI_IP="192.168.0.157"
RASPBERRY_PI_USER="root"
RASPBERRY_PI_PATH="~/libcoap-pqc-bench"

# Display usage information
show_usage() {
    echo -e "${BLUE}PSK Key Manager for CoAP Benchmarking${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  generate [size]      Generate a new key with specified bit size (128, 256, 384, 512)"
    echo "                       Default is 256 if not specified"
    echo "  list                 List all available PSK keys"
    echo "  activate <filename>  Set a specific key as the active key (used in benchmarks)"
    echo "  current              Show the currently active key"
    echo "  deploy               Copy the pskeys directory to the Raspberry Pi (requires SSH configuration)"
    echo "  help                 Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 generate 256      # Generate a 256-bit key"
    echo "  $0 activate psk_256_1.key  # Set this key as active"
    echo "  $0 deploy            # Copy pskeys dir to Raspberry Pi"
}

# Generate a new PSK key with specified bit length
generate_key() {
    local bit_size=$1
    local byte_size=$((bit_size / 8))
    local filename="psk_${bit_size}_$(date +%s).key"
    
    echo -e "${BLUE}Generating ${bit_size}-bit PSK key...${NC}"
    
    # Generate the key
    openssl rand -hex $byte_size > "${PSK_DIR}/${filename}"
    
    # Verify the key was generated
    if [[ -f "${PSK_DIR}/${filename}" ]]; then
        echo -e "${GREEN}Key successfully generated:${NC} ${filename}"
        echo -e "${YELLOW}Key value:${NC} $(cat "${PSK_DIR}/${filename}")"
        echo ""
        echo -e "To activate this key, run: ${YELLOW}$0 activate ${filename}${NC}"
    else
        echo -e "${RED}Failed to generate key${NC}"
        return 1
    fi
}

# List all available PSK keys
list_keys() {
    echo -e "${BLUE}Available PSK Keys:${NC}"
    echo "---------------------------------------------"
    
    # Check if any keys exist
    local key_count=0
    for key_file in "${PSK_DIR}"/*.key; do
        if [[ -f "$key_file" ]]; then
            ((key_count++))
            break
        fi
    done
    
    if [[ $key_count -eq 0 ]]; then
        echo -e "${YELLOW}No keys found. Generate a key first with:${NC}"
        echo -e "${YELLOW}$0 generate [bit_size]${NC}"
        return 0
    fi
    
    # Show current active key if exists
    if [[ -f "${ACTIVE_PSK}" ]]; then
        local active_key_value=$(cat "${ACTIVE_PSK}")
        echo -e "${GREEN}Currently active key:${NC}"
        local active_found=false
        
        for key_file in "${PSK_DIR}"/*.key; do
            if [[ -f "$key_file" ]]; then
                local key_value=$(cat "$key_file")
                if [[ "$key_value" == "$active_key_value" ]]; then
                    echo -e "${GREEN}* $(basename "$key_file")${NC} - $key_value ${GREEN}(ACTIVE)${NC}"
                    active_found=true
                    break
                fi
            fi
        done
        
        if [[ "$active_found" != "true" ]]; then
            echo -e "${YELLOW}* Custom key - $active_key_value${NC}"
        fi
        echo "---------------------------------------------"
    fi
    
    # List all keys
    echo -e "${BLUE}All available keys:${NC}"
    for key_file in "${PSK_DIR}"/*.key; do
        if [[ -f "$key_file" ]]; then
            local key_value=$(cat "$key_file")
            local key_name=$(basename "$key_file")
            local key_size=$(echo "$key_name" | grep -o "[0-9]\+" | head -1)
            
            # Check if this is the active key
            if [[ -f "${ACTIVE_PSK}" ]] && [[ "$(cat "${ACTIVE_PSK}")" == "$key_value" ]]; then
                echo -e "${GREEN}* $key_name${NC} - ${key_size}-bit - $key_value ${GREEN}(ACTIVE)${NC}"
            else
                echo "* $key_name - ${key_size}-bit - $key_value"
            fi
        fi
    done
}

# Set a specific key as the active key
activate_key() {
    local key_name=$1
    local key_path="${PSK_DIR}/${key_name}"
    
    # Check if the key exists
    if [[ ! -f "$key_path" ]]; then
        echo -e "${RED}Key not found: ${key_name}${NC}"
        echo -e "Use '${YELLOW}$0 list${NC}' to see available keys"
        return 1
    fi
    
    # Set the key as active
    cp "$key_path" "${ACTIVE_PSK}"
    echo -e "${GREEN}Successfully activated key:${NC} ${key_name}"
    echo -e "Key value: $(cat "${ACTIVE_PSK}")"
    echo -e "This key will be used for all benchmarks until changed"
}

# Show current active key
show_current_key() {
    if [[ -f "${ACTIVE_PSK}" ]]; then
        echo -e "${BLUE}Currently active PSK key:${NC}"
        echo -e "${YELLOW}$(cat "${ACTIVE_PSK}")${NC}"
        
        # Find the key file with this value
        local active_key_value=$(cat "${ACTIVE_PSK}")
        local active_key_name=""
        
        for key_file in "${PSK_DIR}"/*.key; do
            if [[ -f "$key_file" ]] && [[ "$(cat "$key_file")" == "$active_key_value" ]]; then
                active_key_name=$(basename "$key_file")
                break
            fi
        done
        
        if [[ -n "$active_key_name" ]]; then
            echo -e "Key name: ${GREEN}${active_key_name}${NC}"
        else
            echo -e "Key name: ${YELLOW}Custom key (not managed by this script)${NC}"
        fi
    else
        echo -e "${YELLOW}No active key set.${NC}"
        echo -e "Set an active key with: ${YELLOW}$0 activate <key_filename>${NC}"
    fi
}

# Deploy the active PSK key to Raspberry Pi using rsync
deploy_key() {
    if [[ ! -f "${ACTIVE_PSK}" ]]; then
        echo -e "${RED}No active key to deploy.${NC}"
        echo -e "Set an active key first with: ${YELLOW}$0 activate <key_filename>${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Deploying PSK keys directory to Raspberry Pi (${RASPBERRY_PI_IP}) using rsync...${NC}"
    
    # Create the directory on RPi first
    ssh ${RASPBERRY_PI_USER}@${RASPBERRY_PI_IP} "mkdir -p ${RASPBERRY_PI_PATH}/pskeys"
    
    # Use rsync to copy the pskeys directory
    rsync -av --progress "${PSK_DIR}/" "${RASPBERRY_PI_USER}@${RASPBERRY_PI_IP}:${RASPBERRY_PI_PATH}/pskeys/"
    
    if [ $? -eq 0 ]; then
        # Get current key name for the message
        local key_value=$(cat "${ACTIVE_PSK}")
        local active_key_name=""
        
        for key_file in "${PSK_DIR}"/*.key; do
            if [[ -f "$key_file" ]] && [[ "$(cat "$key_file")" == "$key_value" ]]; then
                active_key_name=$(basename "$key_file")
                break
            fi
        done
        
        # Show success message on RPi
        ssh ${RASPBERRY_PI_USER}@${RASPBERRY_PI_IP} "
            echo -e '\033[0;32m================================\033[0m'
            echo -e '\033[0;32mPSK keys successfully deployed\033[0m'
            echo -e '\033[0;32mActive key: $active_key_name\033[0m'
            echo -e '\033[0;32mKey value: $key_value\033[0m'
            echo -e '\033[0;32m================================\033[0m'
        "
        
        echo -e "${GREEN}PSK keys directory successfully deployed to the Raspberry Pi${NC}"
        echo -e "${GREEN}Active key ($active_key_name) is now ready for use on both systems${NC}"
    else
        echo -e "${RED}Failed to deploy PSK keys directory to the Raspberry Pi${NC}"
        echo -e "Check that rsync is installed and that your SSH configuration is correct"
        return 1
    fi
}

# Main command processing
case "$1" in
    generate)
        # Default to 256 bits if not specified
        bit_size=${2:-256}
        # Validate bit size
        if [[ "$bit_size" =~ ^(128|256|384|512)$ ]]; then
            generate_key "$bit_size"
        else
            echo -e "${RED}Invalid bit size. Please use 128, 256, 384, or 512.${NC}"
            exit 1
        fi
        ;;
    list)
        list_keys
        ;;
    activate)
        if [[ -z "$2" ]]; then
            echo -e "${RED}Error: No key specified${NC}"
            echo -e "Usage: ${YELLOW}$0 activate <key_filename>${NC}"
            exit 1
        fi
        activate_key "$2"
        ;;
    current)
        show_current_key
        ;;
    deploy)
        deploy_key
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

exit 0