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

## Running benchmarks

### Dependencies installation

Install `perf` and `tshark`:

```bash
sudo apt install linux-tools-$(uname -r) tshark
```

Install the Python requirements:

```bash
conda create -n libcoap-bench python=3.10
conda activate libcoap-bench
pip install --no-cache-dir -r ./requirements_installation/requirements.txt
```

*Remark*: To perform the CPU cycles count in raspberry-pi without relying on perf (not recommended at this point), the instructions described in [this site](https://matthewarcus.wordpress.com/2018/01/27/using-the-cycle-counter-registers-on-the-raspberry-pi-3/) must be followed. Note that some modifications may be necessary due to particularities
of the raspberry-pi being used. In the raspberry-pi used for our experiments the following commands must be run every time it is rebooted:

```bash
cd enable_ccr_2024
make
sudo insmod enable_ccr.ko
dmesg | tail
gcc -Wall -O3 cycles.c -o cycles
time taskset 0x1 ./cycles
```

### Benchmark

In a terminal run,

```bash
./libcoap-bench/coap_benchmark_server.sh -sec-mode <pki|psk|nosec> [-rasp]
```

where:

- `-sec-mode`: is the security mode.
- `-rasp`: indicates whether the server is running in raspberry pi or not.

In another terminal run,

```bash
./libcoap-bench/coap_bench_client.sh -n <positive_integer> -sec-mode <pki|psk|nosec> -r <time|async> [-confirm <con|non>] [-s <integer>=1] [-rasp] [-parallelization <background|parallel>]
```

(don't forget to activate the python environment)

where:

- `-n`: is the number of clients that will do requests to the server.
- `-sec-mode`: is the security mode.
- `-r`: is the resource that the client asks for. The resource time corresponds to scenario A (resp. C) if -confirm is set to con (resp. non). The resource async corresponds to scenario B.
- `-confirm`: whether the messages betwen the client and the server are confirmable or not. Mandatory if -r is set to time. 
- `-s`: sets the clients in observer mode and it must be followed by a positive integer: the number of seconds the clients will observe.
- `-rasp`: indicates whether the server is running in raspberry pi or not.
- `-parallelization`: Only needed when the -s parameter is provided. Indicates whether the clients run in the same core (background, this is the default mode) or in different cores (parallel).

A csv file with relevant statistics will be created.

For instance, if libcoap was installed with KYBER_LEVEL5 algorithm:

```bash
./libcoap-bench/coap_benchmark_server.sh -sec-mode pki rasp
```

and

```bash
./libcoap-bench/coap_benchmark_client.sh -n 10 -sec-mode pki -s 30 -parallelization parallel -r time -confirm con -rasp
```

will create the file `udp_rasp_conv_stats_KYBER_LEVEL5_n10_s30_parallel_pki_scenarioA.csv`.

*Remark*: if the energy consumption wants to be included in the csv, these are the steps that must be followed:

1. Installing the GitHub repository <https://github.com/kolinger/rd-usb> in your computer.
2. Connecting the energy tester to your PC via bluetooth. This may depend on your machine, but

      ```bash
      sudo modprobe btusb
      sudo systemctl restart bluetooth
      sudo rfcomm connect hci0 00:15:A6:01:AA:21
      ```

    may work.
3. Running the command `python3 web.py` in the `rd-usb` directory, which will open the energy tester interface.
4. Running the server and client as explained before.
5. Exporting the energy csv generated from the `rd-usb` interface.
6. Running the script `energy_analysis.sh`, which takes the path of the `rd-usb` csv as first parameter and the metrics csv as second parameter. For instance:

      ```bash
      ./libcoap-bench/energy_analysis.sh ~/Downloads/2024-06-24.csv ./libcoap-bench/bench-data/udp_rasp_conv_stats_KYBER_LEVEL5_n10_s30_parallel_pki_scenarioA.csv
      ```

Once appropriate csv's have been created, plots can be drawn via:

```bash
cd libcoap-bench
python3 coap_benchmark_barplots.py <metric> <algorithms_list> <n> <rasp> <scenarios_list> [s] [p]
```

where:

- `metric`: is the metric we want to plot.
- `algorithms_list`: is a list with the algorithms that we want to consider.
- `-n`: is the number of clients considered in the csv's.
- `-rasp`: indicates whether the server was running in raspberry pi or not (True or False).
- `scenarios_list`: is the list of scenarios amongst A (if r=time and confirm=con), B (if r=async) and C (if r=time and confirm=non) that want to be considered.
- `-s`: is the number of seconds the clients observed.
- `-p`: is the parallelization mode (parallel or background).

For instance, we can run

```bash
cd libcoap-bench
python3 ./coap_benchmark_barplots.py 'duration' "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,P256_KYBER_LEVEL1,P384_KYBER_LEVEL3,P521_KYBER_LEVEL5" 500 True A,B,C
```

or

```bash
cd libcoap-bench
python3 ./coap_benchmark_barplots.py 'Wh' "KYBER_LEVEL1,KYBER_LEVEL3,KYBER_LEVEL5,P256_KYBER_LEVEL1,P384_KYBER_LEVEL3,P521_KYBER_LEVEL5" 20 True A,C 30 "background"
```

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
