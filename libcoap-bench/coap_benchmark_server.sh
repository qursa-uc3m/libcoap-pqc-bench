#!/bin/bash

# Usage: ./libcoap-bench/coap_benchmark_server.sh -sec-mode <pki|psk|nosec> [-rasp]

certs_path="./certs"
rasp_option=""

sudo rm ./cycles_output.txt

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
    *)
      # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if -sec-mode is provided
if [ -z "$SEC_MODE" ]; then
  echo "Please provide -sec-mode parameter."
  exit 1
fi

# Use rasp_option as needed in your script
if [ -n "$rasp_option" ]; then
  echo "Rasp option is enabled."
fi


# Determine the command based on the value of -sec-mode
case "$SEC_MODE" in
  pki)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ./libcoap-bench/bench-data/auxiliary_server.txt -e cycles ./libcoap/build/bin/coap-server -A ::1 -c ${certs_path}/server_cert.pem -j ${certs_path}/server_key.pem"
    else
      # Add behavior when rasp_option is on for pki
      # CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./libcoap/build/bin/coap-server -A 192.168.0.157 -c ${certs_path}/server_cert.pem -j ${certs_path}/server_key.pem > ./libcoap-bench/bench-data/auxiliary_server.txt"
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ./libcoap-bench/bench-data/auxiliary_server.txt -e cycles ./libcoap/build/bin/coap-server -A 192.168.0.157 -c ${certs_path}/server_cert.pem -j ${certs_path}/server_key.pem"
    fi
    ;;
  psk)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ./libcoap-bench/bench-data/auxiliary_server.txt -e cycles ./libcoap/build/bin/coap-server -k $(cat psk.txt) -h uc3m -A ::1"
    else
      # Add behavior when rasp_option is on for psk
      # CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./libcoap/build/bin/coap-server -k $(cat psk.txt) -h uc3m -A 192.168.0.157 > ./libcoap-bench/bench-data/auxiliary_server.txt"
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ./libcoap-bench/bench-data/auxiliary_server.txt -e cycles ./libcoap/build/bin/coap-server -k $(cat psk.txt) -h uc3m -A 192.168.0.157"
    fi
    ;;
  nosec)
    if [ -z "$rasp_option" ]; then
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf stat -o ./libcoap-bench/bench-data/auxiliary_server.txt -e cycles ./libcoap/build/bin/coap-server -A ::1"
    else
      # Add behavior when rasp_option is on for nosec
      # CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./libcoap/build/bin/coap-server -A 192.168.0.157 > ./libcoap-bench/bench-data/auxiliary_server.txt"
      CMD="sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH perf_5.10 stat -o ./libcoap-bench/bench-data/auxiliary_server.txt -e cycles ./libcoap/build/bin/coap-server -A 192.168.0.157"
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
