#!/bin/bash

custom_install=""
skip_clone=false
build_dir="$(pwd)/libcoap/build"
install_mode="default"
install_dir=$build_dir
libcoap_dir="$(pwd)/libcoap"
groups_spec=false
libcoap_version="ce2057b7a91a6934aa7c0eb4fd3d899a476b025f" # Develop branch, last good commit
algorithm=""

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
        --groups-spec=*)
            groups_spec=true
            algorithm="${arg#*=}"
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
# sudo apt-get update
# Install required dependencies
# sudo apt-get install -y autoconf automake libtool make gcc

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

patch_coap_wolfssl() {
  local src_file="$libcoap_dir/src/coap_wolfssl.c"
  echo "Patching $src_file for runtime GROUP override..."
  sed -i '892,907c\
static void\
coap_set_user_prefs(WOLFSSL_CTX *ctx) {\
    (void)ctx;\
\
#ifdef COAP_WOLFSSL_SIGALGS\
    wolfSSL_CTX_set1_sigalgs_list(ctx, COAP_WOLFSSL_SIGALGS);\
#endif\
#ifdef COAP_WOLFSSL_GROUPS\
    /* Runtime override of GROUPS: use env var if set, else fallback */\
    const char *env = getenv("COAP_WOLFSSL_GROUPS");\
    const char *groups = (env && *env) ? env : COAP_WOLFSSL_GROUPS;\
    coap_log_debug("Using SSL groups: %s\\n", groups);\
    int ret = wolfSSL_CTX_set1_groups_list(ctx, (const char *)groups);\
    if (ret != WOLFSSL_SUCCESS)\
        coap_log_debug("Failed to set group list\\n");\
#endif\
}\
' "$src_file"
}


if [ "$skip_clone" = false ]; then
    sudo rm -rf ./libcoap
    git clone https://github.com/obgm/libcoap
    cd libcoap
    git checkout $libcoap_version
    patch_coap_wolfssl
else
    cd libcoap
fi

# default client or not
sudo rm $libcoap_dir/examples/coap-client.c
cp $libcoap_dir/../libcoap-bench/coap-client.c $libcoap_dir/examples/

read -p "Do you want to install CUSTOM SERVER (y|n):" is_custom_server

if [ "$is_custom_server" = "y" ] || [ "$is_custom_server" = "Y" ]; then
    echo "Creating custom server ..."
    sudo rm $libcoap_dir/examples/coap-server.c
    cp $libcoap_dir/../libcoap-bench/coap-server.c $libcoap_dir/examples/
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
                CPPFLAGS="-DCOAP_WOLFSSL_GROUPS=\"\\\"$algorithm\\\"\" -DDTLS_V1_3_ONLY=1" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests
            else
                CPPFLAGS="-DCOAP_WOLFSSL_GROUPS=\"\\\"$algorithm\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
            fi
        elif [ "$sigalgs_spec" = true ]; then
            if [ "$install_dir" == "default" ]; then
                CPPFLAGS="-DCOAP_WOLFSSL_SIGALGS=\"\\\"DILITHIUM_LEVEL3\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests
            else
                CPPFLAGS="-DCOAP_WOLFSSL_SIGALGS=\"\\\"DILITHIUM_LEVEL3\\\"\"" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests --prefix=$install_dir
            fi
        else
            if [ "$install_dir" == "default" ]; then
                CPPFLAGS="-DCOAP_WOLFSSL_GROUPS=\\\"\\\" -DDTLS_V1_3_ONLY=1" \
                ./configure --enable-dtls --with-wolfssl --disable-manpages --disable-doxygen --enable-tests
            else
                echo "Installing in custom directory"
                CPPFLAGS="-DCOAP_WOLFSSL_GROUPS=\\\"\\\" -DDTLS_V1_3_ONLY=1" \
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
            CPPFLAGS="-DCOAP_OPENSSL_GROUPS=\"\\\"$algorithm\\\"\"" \
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

echo $algorithm > "$libcoap_dir/../algorithm.txt"

# Build and install
make -j$(nproc)
sudo make install
