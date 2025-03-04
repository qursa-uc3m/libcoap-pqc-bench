#!/bin/bash

# Set the correct OPENSSL installation path

# I do this because my system has multiple versions of OpenSSL installed and the mappings are a mess

export OPENSSL_INSTALL="/opt/oqs_openssl3"

echo "... Configuring OpenSSL environment correctly ..."
export OPENSSL_DIR="${OPENSSL_INSTALL}/.local"
export OPENSSL_INCLUDE_DIR="$OPENSSL_DIR/include"
export OPENSSL_LIB_DIR="$OPENSSL_DIR/lib64"
export OPENSSL_BIN_DIR="$OPENSSL_DIR/bin"
export OPENSSL_CONF_DIR="$OPENSSL_DIR/ssl"
export OPENSSL_MODULES="${OPENSSL_INSTALL}/oqs-provider/_build/lib"
export OPENSSL_CONF="${OPENSSL_CONF_DIR}/openssl.cnf"

unset PATH
export PATH=$OPENSSL_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin

echo "Setting PATH to: $PATH"

echo -e "OpenSSL installation path set to: $OPENSSL_INSTALL"
echo -e "OpenSSL directory set to: $OPENSSL_DIR"
echo -e "OpenSSL configuration set to: $OPENSSL_CONF"
echo ""
openssl version
openssl list -providers