#!/bin/bash

# ==============================================
# config_scenario.sh
# Network Emulation Configuration Script for CoAP Testing
# The tc parameters values must be changed manually 
# ==============================================

# Script directory and parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(dirname "$SCRIPT_DIR")

# Load environment configuration
if [ -f "${REPO_ROOT}/config.local.env" ]; then
    source "${REPO_ROOT}/config.local.env"
elif [ -f "${REPO_ROOT}/config.env" ]; then
    source "${REPO_ROOT}/config.env"
fi

# Mode configuration
LOCAL_MODE="${LOCAL_MODE:-false}"
LOCAL_INTERFACE="${LOCAL_CAPTURE_INTERFACE:-lo}"

# VM configuration (from config.env with fallbacks)
VM_USER="${VM_USER:-ubuntu}"
VM_HOST="${VM_IP:-192.168.0.172}"
VM_INTERFACE="${VM_INTERFACE:-ens3}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo -e "${BLUE}Network Emulation Configuration Script${NC}"
    echo
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo
    echo "Commands:"
    echo "  show              Show current network configuration"
    echo "  set <scenario>    Apply a network emulation scenario"
    echo "  reset             Reset to original configuration"
    echo "  test              Test connection (SSH or local sudo)"
    echo
    echo "Scenarios:"
    echo "  fiducial          No network emulation (baseline)"
    echo "  smart-factory     20ms delay, 1% loss, 50Mbps"
    echo "  smart-home        5ms delay, 0.1% loss, 10Mbps"
    echo "  public-transport  50ms delay, 2% loss, 5Mbps"
    echo
    echo "Options:"
    echo "  --local           Force local mode (loopback interface)"
    echo "  --remote          Force remote mode (SSH to VM)"
    echo "  -u, --user        VM username (default: $VM_USER)"
    echo "  -h, --host        VM hostname/IP (default: $VM_HOST)"
    echo "  -i, --interface   Network interface (default: auto)"
    echo "  -v, --verbose     Verbose output"
    echo "  --help            Show this help message"
    echo
    echo "Mode: Uses LOCAL_MODE from config.env, or --local/--remote flags"
    echo
    echo "Examples:"
    echo "  $0 show"
    echo "  $0 --local set smart-factory"
    echo "  $0 set public-transport --host 192.168.0.100"
    echo "  $0 reset"
    echo
}

# Function to log messages
log() {
    local level=$1
    local message=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    case "$level" in
        "INFO")
            echo -e "${BLUE}[$timestamp] [INFO] ${message}${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[$timestamp] [SUCCESS] ${message}${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[$timestamp] [WARNING] ${message}${NC}"
            ;;
        "ERROR")
            echo -e "${RED}[$timestamp] [ERROR] ${message}${NC}"
            ;;
        *)
            echo -e "[$timestamp] ${message}"
            ;;
    esac
}

# Function to execute command (local or remote)
exec_cmd() {
    local cmd=$1
    if [ "$VERBOSE" == "true" ]; then
        log "INFO" "Executing: $cmd"
    fi
    if [ "$USE_LOCAL" == "true" ]; then
        eval "sudo $cmd"
    else
        ssh "${VM_USER}@${VM_HOST}" "sudo $cmd"
    fi
}

# Get the active interface
get_interface() {
    if [ -n "$OVERRIDE_INTERFACE" ]; then
        echo "$OVERRIDE_INTERFACE"
    elif [ "$USE_LOCAL" == "true" ]; then
        echo "$LOCAL_INTERFACE"
    else
        echo "$VM_INTERFACE"
    fi
}

# Function to test connection
test_connection() {
    local iface=$(get_interface)
    if [ "$USE_LOCAL" == "true" ]; then
        log "INFO" "Testing local sudo permissions on interface $iface..."
        if sudo tc qdisc show dev "$iface" > /dev/null 2>&1; then
            log "SUCCESS" "Local mode: sudo and tc working"
            return 0
        else
            log "ERROR" "Failed to access tc on $iface. Check sudo permissions."
            return 1
        fi
    else
        log "INFO" "Testing SSH connection to ${VM_USER}@${VM_HOST}..."
        if ssh "${VM_USER}@${VM_HOST}" "echo 'Connection successful'" > /dev/null 2>&1; then
            log "SUCCESS" "SSH connection is working"
            return 0
        else
            log "ERROR" "Failed to connect to VM. Please check SSH configuration."
            return 1
        fi
    fi
}

# Function to show current configuration
show_current() {
    local iface=$(get_interface)
    log "INFO" "Fetching current network configuration..."
    
    local mode_str="Remote (${VM_USER}@${VM_HOST})"
    [ "$USE_LOCAL" == "true" ] && mode_str="Local"
    
    local result=$(exec_cmd "tc qdisc show dev ${iface}" 2>&1)
    
    echo -e "\n${CYAN}Mode: ${mode_str} | Interface: ${iface}${NC}"
    echo "$result"
    
    # Detect which scenario is active
    if echo "$result" | grep -q "netem"; then
        echo -e "\n${YELLOW}Network emulation is active:${NC}"
        
        # Extract parameters
        local delay=$(echo "$result" | grep -oP 'delay \K[0-9.]+ms' | head -1 || echo "N/A")
        local loss=$(echo "$result" | grep -oP 'loss \K[0-9.]+%' | head -1 || echo "N/A")
        local rate=$(echo "$result" | grep -oP 'rate \K[0-9.]+[KMG]?bit' | head -1 || echo "N/A")
        
        echo "  Delay: $delay"
        echo "  Loss: $loss"
        echo "  Rate: $rate"
        
        # Try to identify the scenario
        if [[ "$delay" == "20"* ]] && [[ "$loss" == "1"* ]]; then
            echo -e "  ${GREEN}Scenario: Smart Factory${NC}"
        elif [[ "$delay" == "5"* ]] && [[ "$loss" == "0.1"* ]]; then
            echo -e "  ${GREEN}Scenario: Smart Home${NC}"
        elif [[ "$delay" == "50"* ]] && [[ "$loss" == "2"* ]]; then
            echo -e "  ${GREEN}Scenario: Public Transport${NC}"
        else
            echo -e "  ${YELLOW}Scenario: Custom/Unknown${NC}"
        fi
    else
        echo -e "\n${GREEN}No network emulation active (Fiducial/Baseline)${NC}"
    fi
}

# Function to reset to original configuration
reset_config() {
    local iface=$(get_interface)
    log "INFO" "Resetting network emulation on $iface..."
    
    # Clear any existing emulation
    exec_cmd "tc qdisc del dev ${iface} root" 2>/dev/null
    
    if [ "$USE_LOCAL" == "true" ]; then
        # For loopback, just clear (kernel handles default)
        log "SUCCESS" "Network emulation cleared"
    else
        # For remote VM, restore original fq_codel configuration
        local cmd="tc qdisc add dev ${iface} root fq_codel limit 10240 flows 1024 quantum 1514 target 5ms interval 100ms memory_limit 32Mb ecn drop_batch 64"
        if exec_cmd "$cmd" 2>/dev/null; then
            log "SUCCESS" "Configuration reset to original fq_codel"
        else
            log "WARNING" "Could not restore fq_codel, but emulation is cleared"
        fi
    fi
}

# Function to apply network scenario
apply_scenario() {
    local scenario=$1
    local iface=$(get_interface)
    
    case "$scenario" in
        "fiducial")
            log "INFO" "Applying Fiducial scenario (no emulation)..."
            reset_config
            ;;
            
        "smart-factory")
            log "INFO" "Applying Smart Factory scenario..."
            exec_cmd "tc qdisc del dev ${iface} root" 2>/dev/null
            
            local cmd="tc qdisc add dev ${iface} root netem delay 20ms 5ms distribution normal loss 1% rate 50Mbit"
            if exec_cmd "$cmd"; then
                log "SUCCESS" "Smart Factory scenario applied"
            else
                log "ERROR" "Failed to apply Smart Factory scenario"
                return 1
            fi
            ;;
            
        "smart-home")
            log "INFO" "Applying Smart Home scenario..."
            exec_cmd "tc qdisc del dev ${iface} root" 2>/dev/null
            
            local cmd="tc qdisc add dev ${iface} root netem delay 5ms 1ms distribution normal loss 0.1% rate 10Mbit"
            if exec_cmd "$cmd"; then
                log "SUCCESS" "Smart Home scenario applied"
            else
                log "ERROR" "Failed to apply Smart Home scenario"
                return 1
            fi
            ;;
            
        "public-transport")
            log "INFO" "Applying Public Transport scenario..."
            exec_cmd "tc qdisc del dev ${iface} root" 2>/dev/null
            
            local cmd="tc qdisc add dev ${iface} root netem delay 50ms 10ms distribution normal loss 2.0% rate 5Mbit"
            if exec_cmd "$cmd"; then
                log "SUCCESS" "Public Transport scenario applied"
            else
                log "ERROR" "Failed to apply Public Transport scenario"
                return 1
            fi
            ;;
            
        *)
            log "ERROR" "Unknown scenario: $scenario"
            echo "Available scenarios: fiducial, smart-factory, smart-home, public-transport"
            return 1
            ;;
    esac
    
    # Show the new configuration
    echo
    show_current
}

# Parse command line arguments
COMMAND=""
SCENARIO=""
VERBOSE=false
USE_LOCAL="$LOCAL_MODE"
OVERRIDE_INTERFACE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_LOCAL="true"
            shift
            ;;
        --remote)
            USE_LOCAL="false"
            shift
            ;;
        -u|--user)
            VM_USER="$2"
            shift 2
            ;;
        -h|--host)
            VM_HOST="$2"
            shift 2
            ;;
        -i|--interface)
            OVERRIDE_INTERFACE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        show|test|reset)
            COMMAND="$1"
            shift
            ;;
        set)
            COMMAND="$1"
            SCENARIO="$2"
            shift 2
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                log "ERROR" "Unknown command: $1"
                usage
                exit 1
            else
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
            fi
            ;;
    esac
done

# Validate command
if [ -z "$COMMAND" ]; then
    log "ERROR" "No command specified"
    usage
    exit 1
fi

# Execute command
case "$COMMAND" in
    "test")
        test_connection
        ;;
        
    "show")
        if test_connection; then
            show_current
        fi
        ;;
        
    "reset")
        if test_connection; then
            reset_config
        fi
        ;;
        
    "set")
        if [ -z "$SCENARIO" ]; then
            log "ERROR" "No scenario specified for 'set' command"
            usage
            exit 1
        fi
        
        if test_connection; then
            apply_scenario "$SCENARIO"
        fi
        ;;
        
    *)
        log "ERROR" "Invalid command: $COMMAND"
        usage
        exit 1
        ;;
esac

exit 0