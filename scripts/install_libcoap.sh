#!/bin/bash

skip_clone=false
build_dir="$(pwd)/libcoap/build"
install_mode="default"
install_dir=$build_dir
libcoap_dir="$(pwd)/libcoap"
groups_spec=false
libcoap_version="cb20c482b2bb857a2f06c342ecb8c8c6d5f387ce"
algorithm="kyber768"  # Default groups including PQC. Use OPENSSL names here!
openssl_dir="/opt/openssl_dtls13/.local"  # Default OpenSSL DTLS 1.3 directory

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --skip-clone)
            skip_clone=true
            ;;
        --install-mode=*)
            install_mode="${arg#*=}"
            ;;
        --groups-spec)
            groups_spec=true
            ;;
        --algorithm=*)
            algorithm="${arg#*=}"
            ;;
        --install-dir=*)
            install_dir="${arg#*=}"
            ;;
        --openssl-dir=*)
            openssl_dir="${arg#*=}"
            ;;
    esac
done

echo "---------------------------------"
echo "Install dir: $install_dir"
echo "OpenSSL dir: $openssl_dir"
echo "Using groups/algorithms: $algorithm"
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
    #git checkout $libcoap_version
fi

# Replace client file if needed
sudo rm -f $libcoap_dir/examples/coap-client.c
if [ -f "$libcoap_dir/../libcoap-bench/coap-client.c" ]; then
    cp $libcoap_dir/../libcoap-bench/coap-client.c $libcoap_dir/examples/
fi

read -p "Do you want to install the library in a Raspberry Pi? (y/n):" is_rpi

if [ "$is_rpi" = "y" ] || [ "$is_rpi" = "Y" ]; then
    echo "Creating default server ..."
    sudo rm $libcoap_dir/examples/coap-server.c
    cp $libcoap_dir/../libcoap-bench/coap-server.c $libcoap_dir/examples/
fi

# Check if OpenSSL directory exists
if [ ! -d "$openssl_dir" ]; then
    echo "ERROR: OpenSSL DTLS 1.3 installation not found at $openssl_dir"
    echo "Please check the --openssl-dir parameter or run the install_openssl_dtls13.sh script first."
    exit 1
fi

# Check if liboqs is installed for OpenSSL
if [ ! -d "$openssl_dir/lib/ossl-modules" ] || [ ! -f "$openssl_dir/lib/ossl-modules/oqsprovider.so" ]; then
    echo "WARNING: OQS provider not found. PQC support may not be available."
    echo "You might want to run install_liboqs_for_openssl.sh first."
    read -p "Continue anyway? (y/n): " continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        exit 1
    fi
fi

echo "................................................................"
echo "Configuring libcoap with DTLS 1.3 and PQC support using OpenSSL"
echo "................................................................"

# Set environment variables to ensure correct OpenSSL is used
export PKG_CONFIG_PATH="$openssl_dir/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$openssl_dir/lib:$LD_LIBRARY_PATH"

# Configure with OpenSSL DTLS 1.3 and PQC
if [ "$install_mode" == "default" ]; then
    cd $libcoap_dir
    ./autogen.sh
    
    # Always configure with groups - this includes PQC algorithms
    # Added LDFLAGS and rpath to ensure correct linking
    CPPFLAGS="-I$openssl_dir/include -DCOAP_OPENSSL_GROUPS=\"\\\"$algorithm\\\"\" -DCOAP_DTLS_OPENSSL_BUILDER_OPTION=\"\\\"providers\\\"\"" \
    LDFLAGS="-L$openssl_dir/lib -Wl,-rpath,$openssl_dir/lib" \
    ./configure --enable-dtls --with-openssl --disable-manpages --disable-doxygen --enable-tests \
    --with-openssl-dir=$openssl_dir --prefix=$install_dir
else
    echo "Using CMake...."
    mkdir -p $build_dir
    cd $build_dir
    
    # Pass the groups via a definition to CMake
    # Added LDFLAGS and rpath to ensure correct linking
    LDFLAGS="-L$openssl_dir/lib -Wl,-rpath,$openssl_dir/lib" \
    cmake -DENABLE_DTLS=ON -DDTLS_BACKEND=openssl -DOPENSSL_ROOT_DIR=$openssl_dir \
    -DCOAP_OPENSSL_GROUPS="$algorithm" -DCOAP_DTLS_OPENSSL_BUILDER_OPTION="providers" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$openssl_dir/lib" ..
fi

echo $algorithm > "$libcoap_dir/../algorithm.txt"

# Build and install
make -j$(nproc)
sudo make install

echo "................................................................"
echo "libcoap with OpenSSL DTLS 1.3 and PQC support has been installed"
echo "................................................................"

# Verify the binary linking
echo "Verifying linked libraries for coap-client:"
ldd $build_dir/bin/coap-client | grep libssl
ldd $build_dir/bin/coap-client | grep libcrypto

echo "Verifying linked libraries for coap-server:"
ldd $build_dir/bin/coap-server | grep libssl
ldd $build_dir/bin/coap-server | grep libcrypto

# Check if libraries are correctly linked
if ldd $build_dir/bin/coap-client | grep -q "$openssl_dir/lib64/libssl"; then
    echo "SUCCESS: coap-client is correctly linked to OpenSSL DTLS 1.3"
else
    echo "WARNING: coap-client is NOT linked to OpenSSL DTLS 1.3"
    echo "You may need to use LD_LIBRARY_PATH when running the client"
fi
# Check if libraries are correctly linked
if ldd $build_dir/bin/coap-server | grep -q "$openssl_dir/lib64/libssl"; then
    echo "SUCCESS: coap-server is correctly linked to OpenSSL DTLS 1.3"
else
    echo "WARNING: coap-server is NOT linked to OpenSSL DTLS 1.3"
    echo "You may need to use LD_LIBRARY_PATH when running the server"
fi
echo "................................................................"
