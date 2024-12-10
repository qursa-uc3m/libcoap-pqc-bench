#!/bin/bash

custom_install=""
skip_clone=false
build_dir="$(pwd)/libcoap/build"
install_mode="default"
install_dir=$build_dir
groups_spec=false
libcoap_version="cb20c482b2bb857a2f06c342ecb8c8c6d5f387ce"

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        wolfssl)
            custom_install="wolfssl"
            ;;
        --skip-clone)
            skip_clone=true
            ;;
        --install-mode=*)
            install_mode="${arg#*=}"
            ;;
        --groups-spec)
            groups_spec=true
            ;;
        --sigalgs-spec)
            sigalgs_spec=true
            ;;
        --install-dir=*)
            install_dir="${arg#*=}"
            echo "Install dir: $install_dir"
            ;;
    esac
done

echo "---------------------------------"
# print install dir
echo "Install dir: $install_dir"
echo "---------------------------------"
# Update package lists
sudo apt-get update
# Install required dependencies
sudo apt-get install -y autoconf automake libtool make gcc

clean_coap_build() {
    if [ -d "./libcoap" ]; then
        echo "Cleaning existing libcoap directory..."
        cd libcoap
        make clean
        ./autogen.sh --clean
        sudo make uninstall
        cd ..
    else
        echo "libcoap directory does not exist, skipping clean..."
    fi
}

# Clean existing libcoap build
clean_coap_build

if [ "$skip_clone" = false ]; then
    sudo rm -rf ./libcoap
    git clone https://github.com/obgm/libcoap
    cd libcoap
    git checkout $libcoap_version
fi

# Configure based on custom_install flag
if [ "$custom_install" == "wolfssl" ]; then
    echo ".........."
    echo "Configuring with DTLS and WolfSSL"
    echo ".........."
    if [ "$install_mode" == "default" ]; then
        echo "Using autogen.sh...."
        ./autogen.sh
        # add --enable-server-mode flag to enable server mode
        #-DCOAP_WOLFSSL_GROUPS=\"P-384:P-256:KYBER_LEVEL1\"
        if [ "$groups_spec" = true ]; then
            if [ "$install_dir" == "default" ]; then
                CPPFLAGS="-DCOAP_WOLFSSL_GROUPS=\"\\\"P-256\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests
            else
                CPPFLAGS="-DCOAP_WOLFSSL_GROUPS=\"\\\"P-256\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
            fi
        elif [ "$sigalgs_spec" = true ]; then
            if [ "$install_dir" == "default" ]; then
                CPPFLAGS="-DCOAP_WOLFSSL_SIGALGS=\"\\\"RSA+SHA256\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests
            else
                CPPFLAGS="-DCOAP_WOLFSSL_SIGALGS=\"\\\"dilithium3\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
            fi
        else
            if [ "$install_dir" == "default" ]; then
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests
            else
                echo "Installing in custom directory"
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
            fi
        fi
    else
        echo "Using CMake...."
        mkdir -p $build_dir
        cd $build_dir
        cmake -DENABLE_DTLS=ON -DDTLS_BACKEND=wolfssl ..
    fi
else
    echo ".........."
    echo "Configuring with DTLS and OpenSSL"
    echo ".........."
    # Generate build scripts
    if [ "$install_mode" == "default" ]; then
        ./autogen.sh
        if [ "$groups_spec" = true ]; then
            CPPFLAGS="-DCOAP_OPENSSL_GROUPS=\"\\\"P-256\\\"\"" \
            ./configure --enable-dtls --with-openssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
        else
            ./configure --enable-dtls --with-openssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
        fi
    else
        mkdir -p $build_dir
        cd $build_dir
        cmake -DENABLE_DTLS=ON -DDTLS_BACKEND=openssl ..
    fi
fi

# Build and install
make
sudo make install