#!/bin/bash

# Import certificate configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(pwd)/certs/config_certs.sh"
BENCH_DIR="${REPO_ROOT}/libcoap-bench"
COAP_BIN="${REPO_ROOT}/libcoap/build/bin"
PSK_DIR="${REPO_ROOT}/pskeys"
ACTIVE_PSK="${PSK_DIR}/active_psk.txt"

rasp_option=""
cert_config="DEFAULT"
client_auth="no"  # Default to no client authentication

sudo rm -f "${REPO_ROOT}/cycles_output.txt"

# Function to display usage/help
show_usage() {
    echo "Usage: ${0} -sec-mode <pki|psk|nosec> [-rasp] [-cert-config <CONFIG>] [-client-auth <yes|no>]"
    echo ""
    echo "Options:"
    echo "  -sec-mode <pki|psk|nosec>    Security mode to use (required)"
    echo "  -rasp                        Enable Raspberry Pi mode"
    echo "  -cert-config <CONFIG>        Certificate configuration to use (for PKI mode)"
    echo "  -client-auth <yes|no>        Enable/disable client certificate authentication"
    echo "                               Default is 'no' (only server authentication)"
    echo "  -list-certs                  List available certificate configurations"
    echo "  -h, --help                   Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -sec-mode)
      SEC_MODE="$2"
      shift
      shift
      ;;
    -[rR][aA][sS][pP]) # Match -rasp, -RASP, -RAsP, etc.
      rasp_option="true"
      shift
      ;;
    -cert-config)
      cert_config="$2"
      shift
      shift
      ;;
    -client-auth)
      client_auth="$2"
      if [[ "$client_auth" != "yes" && "$client_auth" != "no" ]]; then
        echo "Error: -client-auth must be 'yes' or 'no'"
        show_usage
      fi
      shift
      shift
      ;;
    -list-certs)
      list_cert_configs
      exit 0
      ;;
    -h|--help)
      show_usage
      ;;
    *)
      # unknown option
      echo "Unknown option: $1"
      show_usage
      ;;
  esac
done

# Check if -sec-mode is provided
if [ -z "$SEC_MODE" ]; then
  echo "Please provide -sec-mode parameter."
  show_usage
fi

# Validate certificate configuration if using PKI mode
if [ "$SEC_MODE" == "pki" ]; then
  if ! validate_cert_files "$cert_config"; then
    echo "Certificate validation failed. Exiting."
    exit 1
  fi
  
  # Get certificate paths
  cert_paths=$(get_cert_paths "$cert_config")
  IFS=';' read -r cert_file key_file ca_file <<< "$cert_paths"
  
  # Save or update algorithm information
  if [ -f "${REPO_ROOT}/algorithm.txt" ]; then
    # Get first line (KEM algorithm) from the existing file
    kem_algorithm=$(head -n 1 "${REPO_ROOT}/algorithm.txt")
    # Write a two-line file with KEM on first line and signature on second
    echo "${kem_algorithm}" > "${REPO_ROOT}/algorithm.txt"
    echo "${cert_config}" >> "${REPO_ROOT}/algorithm.txt"
  else
    # First time, just write the certificate config as second line
    # Assuming first line (KEM) is added during build
    echo "UNKNOWN_KEM" > "${REPO_ROOT}/algorithm.txt"
    echo "${cert_config}" >> "${REPO_ROOT}/algorithm.txt"
  fi
  
  echo "Using certificate configuration: $cert_config"
  echo "  Certificate: $cert_file"
  echo "  Key: $key_file"
  echo "  CA: $ca_file"
  echo "  Client Authentication: $client_auth"
fi

# Check for active PSK key when in PSK mode
if [ "$SEC_MODE" == "psk" ]; then
  if [ ! -f "${ACTIVE_PSK}" ]; then
    echo "No active PSK key found. Please activate a key using:"
    echo "./pskeys/psk_key_manager.sh activate <key_filename>"
    exit 1
  fi
  
  # Get the key name for display purposes
  active_key_value=$(cat "${ACTIVE_PSK}")
  active_key_name=""
        
  for key_file in "${PSK_DIR}"/*.key; do
    if [[ -f "$key_file" ]] && [[ "$(cat "$key_file")" == "$active_key_value" ]]; then
      active_key_name=$(basename "$key_file")
      break
    fi
  done
  
  if [[ -n "$active_key_name" ]]; then
    echo "Using active PSK key: $active_key_name"
  else
    echo "Using active PSK key: Custom key"
  fi
  echo "Key value: $(cat ${ACTIVE_PSK})"
fi

# Use rasp_option as needed in your script
if [ -n "$rasp_option" ]; then
  echo "Rasp option is enabled."
fi

echo "Creating benchmark data directory in ${BENCH_DIR}/bench-data ..."
mkdir -p ${BENCH_DIR}/bench-data

# Determine if client authentication is disabled (add -n flag if yes)
client_auth_flag=""
if [ "$SEC_MODE" == "pki" ] && [ "$client_auth" == "no" ]; then
  client_auth_flag="-n"
fi

# Determine the command based on the value of -sec-mode
case "$SEC_MODE" in
  pki)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -A ::1 -c ${cert_file} -j ${key_file} ${client_auth_flag}"
    else
      # Add behavior when rasp_option is on for pki
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -A 192.168.0.157 -c ${cert_file} -j ${key_file} ${client_auth_flag}"
    fi
    ;;
  psk)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -k $(cat ${ACTIVE_PSK}) -h uc3m -A ::1"
    else
      # Add behavior when rasp_option is on for psk
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -k $(cat ${ACTIVE_PSK}) -h uc3m -A 192.168.0.157"
    fi
    ;;
  nosec)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -A ::1"
    else
      # Add behavior when rasp_option is on for nosec
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -A 192.168.0.157"
    fi
    ;;
  *)
    echo "Invalid -sec-mode value: $SEC_MODE"
    exit 1
    ;;
esac

# Run the determined command
echo "Running command: $CMD"
eval "$CMD"