# Building liboqs provider for OpenSSL 3.0

## With `install_openssl3_ad_oqs_provider.sh`

### Usage

```bash
./install_openssl3_and_oqs_provider.sh -p <install_dir> -d <debug>
```

Options
-p `<install_dir>`: Optional. Set the installation directory. Default is /opt/oqs_openssl3.
-d `<debug>`: Optional. Set the debug mode. Default is 0 (disabled). If set to 1, the `rand_lib.c` file in the OpenSSL source will be replaced by a rand.c file located in the same directory as this script.

The `rand.c` file contains print statements that will keep track of the random number generation process. This is useful for debugging.

## Step by step

Follow the instructions in the official repository [open-quantum-safe/oqs-provider](https://github.com/open-quantum-safe/oqs-provider) to build the provider.

See also:

* [liboqs](https://github.com/open-quantum-safe/liboqs)

Build OpenSSL 3.*:

```bash
git clone git://git.openssl.org/openssl.git
cd openssl
./config --prefix=$(echo $(pwd)/../.local) && make && make install_sw
cd ..
```

Building the provider. Run this in the `<oqs-provider-test-dir>` directory:

```bash
git clone https://github.com/open-quantum-safe/oqs-provider.git
cd oqs-provider
cmake -DOPENSSL_ROOT_DIR=$(pwd)/../.local -DCMAKE_PREFIX_PATH=$(pwd)/../.local -S . -B _build
cmake --build _build
cd ..
```

Creating hard link:

```bash
sudo ln .local/bin/openssl /usr/local/bin/oqs_openssl3
```

Add this to `~/.bashrc`:

```text
export LD_LIBRARY_PATH="<oqs-provider-test-dir>/.local/lib64:$LD_LIBRARY_PATH"
```

Move the provider to the `/opt/oqs/providers/oqs-provider/lib` directory:

```bash
sudo mkdir -p /opt/oqs/providers/oqs-provider
sudo cp -r oqs-provider/_build/lib /opt/oqs/providers/oqs-provider
```

Checking provider info:

```bash
oqs_openssl3 list -providers -verbose -provider-path /opt/oqs/providers/oqs-provider/lib -provider oqsprovider 
```

## Compiling OpenSSL for debugging

.
In the `./openssl` directory run

```bash
CFLAGS="-g" ./config --prefix=$(echo $(pwd)/../.local) && make && make install_sw
```

## Adding `openssl.cnf` file enabling providers

First check the location of the OpenSSL openssl.cnf configuration file with

```bash
oqs_openssl3 version -d
```

You should get

```text
OPENSSLDIR: "/opt/oqs_openssl3/.local/ssl"
```

Then create the directory

```bash
sudo mkdir -p /opt/oqs_openssl3/.local/ssl
```

Make sure that the `openssl.cnf` file contains the following lines

```text
[openssl_init]
providers = provider_sect

# List of providers to load
[provider_sect]
oqsprovider = oqsprovider_section
default = default_sect

[oqsprovider_section]
activate = 1
module = /opt/oqs_openssl3/oqs-provider/_build/lib/oqsprovider.so
[default_sect]
activate = 1
[legacy_sect]
activate = 1
```

Notice that you have to point to the `oqsprovider.so` file location, in this case in the `oqs-provider` build directory.

Finally move the `openssl.cnf` sample file in this repository to that directory

```bash
sudo mkdir /opt/oqs_openssl3/.local/ssl
sudo cp ./openssl.cnf /opt/oqs_openssl3/.local/ssl
```

Check that the providers habe been corretly loaded with

```bash
oqs_openssl3 list -providers -verbose
```
