# MQTT-SN Clients with wolfMQTT Library for PQC Testing

This repository contains MQTT-SN clients based on [wolfMQTT](https://github.com/wolfSSL/wolfMQTT)'s [wolfMQTT/examples/sn-client](https://github.com/wolfSSL/wolfMQTT/tree/master/examples/sn-client)

## Build wolfSSL and wolfMQTT

### wolfSSL

Clone this repository and navigate to the project directory.
Run the wolfSSL installation script:

```bash
./scripts/install_wolfssl.sh
```

By default, this script installs wolfSSL version `5.6.4-stable`. You can change the version by modifying the `INSTALL_TAG` variable in the `install_wolfssl.sh` script.

### wolfMQTT

After installing wolfSSL, install wolfMQTT:

```bash
./scripts/install_wolfMQTT.sh
```

By default, this script installs wolfMQTT version `1.19.0`. You can change the version by modifying the `INSTALL_TAG` variable in the `install_wolfMQTT.sh` script.

Note: Ensure that the versions of wolfSSL and wolfMQTT are compatible with each other and with the MQTT-SN clients in this repository.

## Configuration

### Key Exchange Groups (Runtime Selection)

The key exchange group for DTLS connections is configured via the `MQTT_WOLFSSL_GROUPS` environment variable at runtime. This allows switching between algorithms without recompilation.

If `MQTT_WOLFSSL_GROUPS` is not set, wolfSSL will use its default groups.

## Build the project

```bash
mkdir build
cd build
cmake ..
make
```

Alternatively, you can run the `./scripts/build_clients.sh` script.

## Running the clients

To run the MQTT-SN client:

```bash
./build/bin/sn-client -t 
```

To run the MQTT-SN publisher:

```bash
./build/bin/sn-pub -t 
```

To run the MQTT-SN subscriber:

```bash
./build/bin/sn-sub -t 
```

## Testing

The repository includes a test script (`test.sh`) to evaluate both publisher and subscriber clients.

### Usage

```bash
./test.sh [-n iterations] [-m pub|sub]
```

Options:

- `-n`: Number of iterations (default: 1)
- `-m`: Mode - 'pub' for publisher or 'sub' for subscriber (default: pub)

For example, to run the publisher 5 times:

```bash
./test.sh -n 5 -m pub
```