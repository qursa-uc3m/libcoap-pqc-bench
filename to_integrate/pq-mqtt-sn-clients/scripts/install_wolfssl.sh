#!/bin/bash
INSTALL_TAG="v5.6.4-stable"
#INSTALL_TAG="master"
# Installing missing packages for Raspberry Pi:
sudo apt-get update
sudo apt-get install -y autoconf automake libtool coreutils bsdmainutils

git clone https://github.com/wolfSSL/wolfssl.git
cd wolfssl
if [ "$INSTALL_TAG" != "master" ]; then
    git checkout $INSTALL_TAG
fi

./autogen.sh
mkdir build
cd build
../configure CFLAGS="-DHAVE_SECRET_CALLBACK" \
    --enable-opensslall \
    --enable-opensslextra \
    --enable-static \
    --enable-psk \
    --enable-alpn \
    --enable-aesccm \
    --enable-aesgcm \
    --enable-dtls-mtu \
    --enable-context-extra-user-data=yes \
    --enable-dtls \
    --enable-debug \
    --with-liboqs \
    --enable-dtls13 \
    --enable-tls13 \
    --enable-dtls-frag-ch \
    --enable-sp-math-all

make all
sudo make install
