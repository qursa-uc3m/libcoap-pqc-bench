# Benchmarking Post-Quantum Cryptography in libcoap

Benchmarking post-quantum cryptographic algorithms in CoAP using liboqs, wolfSSL, and libcoap libraries (note that this is currently a work in progress).

## Installation

If you want PQC, first install the dependencies

```bash
./scripts/install_liboqs_for_wolfssl.sh
```

Then build wolfssl

```bash
./scripts/install_wolfssl.sh
```

Install libcoap dependencies

```bash
sudo apt-get install -y autoconf automake libtool make gcc
```

you may also need

```bash
sudo apt-get install autoconf-archive libwolfssl-dev libcunit1-dev pkg-config
```

And run the installation script with the desired options

```bash
./scripts/install_libcoap.sh [wolfssl] [--groups-spec]
```

Flags:

- `wolfssl`: This option indicates that you want to configure libcoap with WolfSSL as the underlying cryptographic library. If not provided, the script will configure libcoap with OpenSSL.
- `--groups-spec`: When provided, this option will set specific cryptographic groups during the configuration phase. Indicate the desired groups in the script. If not provided, the script will configure libcoap with the default groups.

## Network Emulation

For information on how to set up and run network emulation with a Kernel-based Virtual Machine (KVM) and NetEm, see the [network_emulation/README.md](network_emulation/README.md).

## Cleaning zombie processes

You can check if there are any zombie processes with the following command:

```bash
sudo netstat -tulnp | grep -E '5683|5684'
```

You can remove them with the following command (note: this is for development, you have to ensure that you are not killing any important processes):

```bash
sudo pgrep -f 'libcoap' | while read pid; do sudo kill -9 $pid; done
```



## Analyzing the traffic with Wireshark

See [OQS-wireshark](https://github.com/open-quantum-safe/oqs-demos/blob/main/wireshark/USAGE.md) for more details. Perhaps you need to run

```console
xhost +si:localuser:root
```

instead of

```console
xhost +si:localuser:$USER
```

if your user is not in the **docker** group. In that case

```console
sudo docker run --net=host --privileged --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" openquantumsafe/wireshark
```

Then you can filter by `udp.port==5684` for DTLS or `udp.port==5683` for CoAP (or `udp.port==5684 || udp.port==5683` for both).
