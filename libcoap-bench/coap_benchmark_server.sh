#!/bin/bash

# Import certificate configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(pwd)/certs/config_certs.sh"
BENCH_DIR="${REPO_ROOT}/libcoap-bench"
COAP_BIN="${REPO_ROOT}/libcoap/build/bin"

rasp_option=""
cert_config="DEFAULT"

sudo rm -f "${REPO_ROOT}/cycles_output.txt"

# Function to display usage/help
show_usage() {
    echo "Usage: ${0} -sec-mode <pki|psk|nosec> [-rasp] [-cert-config <CONFIG>]"
    echo ""
    echo "Options:"
    echo "  -sec-mode <pki|psk|nosec>    Security mode to use (required)"
    echo "  -rasp                        Enable Raspberry Pi mode"
    echo "  -cert-config <CONFIG>        Certificate configuration to use (for PKI mode)"
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
    [rR][aA][sS][pP]) # Match -rasp, -RASP, -RAsP, etc.
      rasp_option="true"
      shift
      ;;
    -cert-config)
      cert_config="$2"
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
fi

# Use rasp_option as needed in your script
if [ -n "$rasp_option" ]; then
  echo "Rasp option is enabled."
fi

# Determine the command based on the value of -sec-mode
case "$SEC_MODE" in
  pki)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -A ::1 -c ${cert_file} -j ${key_file}"
    else
      # Add behavior when rasp_option is on for pki
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -A 192.168.0.157 -c ${cert_file} -j ${key_file}"
    fi
    ;;
  psk)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -k $(cat ${REPO_ROOT}/psk.txt) -h uc3m -A ::1"
    else
      # Add behavior when rasp_option is on for psk
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ${BENCH_DIR}/bench-data/auxiliary_server.txt -e cycles ${COAP_BIN}/coap-server -k $(cat ${REPO_ROOT}/psk.txt) -h uc3m -A 192.168.0.157"
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