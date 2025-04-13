# Certificate Management for libcoap PQC Benchmarking

This directory contains tools and configurations for generating and managing cryptographic certificates used in the Post-Quantum Cryptography (PQC) benchmarking framework.

## Available Certificate Types

The following certificate types are supported:

- **RSA** - Traditional RSA certificates (RSA_2048)
- **Dilithium** - Post-quantum signature algorithm at various security levels:
  - DILITHIUM_LEVEL2 (equivalent to AES-128)
  - DILITHIUM_LEVEL3 (equivalent to AES-192)
  - DILITHIUM_LEVEL5 (equivalent to AES-256)
- **Falcon** - Post-quantum signature algorithm at various security levels:
  - FALCON_LEVEL1 (equivalent to AES-128)
  - FALCON_LEVEL5 (equivalent to AES-256)
- **Elliptic Curve** - Traditional elliptic curve certificates:
  - EC_P256 (NIST P-256 curve)
  - EC_ED25519 (Edwards curve 25519)

## Directory Structure

The certificates are organized in the following structure:

```
certs/
├── config_certs.sh       # Certificate configuration utility
├── generate_certs.sh     # Certificate generation script
├── dilithium/            # Dilithium certificates
├── falcon/               # Falcon certificates
├── rsa/                  # RSA certificates
├── ec/                   # Elliptic Curve certificates
└── oqs_prov_install/     # OpenSSL provider installation
```

## Certificate Generation

To generate a full set of certificates for all supported algorithms, run:

```bash
./generate_certs.sh
```

### Options

The certificate generation script supports the following options:

- `--rasp` - Sync generated certificates to a Raspberry Pi after creation
- `--rpi-address ADDR` - Specify Raspberry Pi IP address (default: 192.168.0.157)
- `--rpi-user USER` - Specify Raspberry Pi username (default: root)

Example with custom Raspberry Pi settings:

```bash
./generate_certs.sh --rasp --rpi-address 192.168.1.100 --rpi-user pi
```

## Certificate Configuration

The `config_certs.sh` script provides utilities for managing certificate configurations:

### Listing Available Configurations

To see all available certificate configurations:

```bash
./config_certs.sh --list
```

### Validating a Configuration

To check if a specific certificate configuration is valid and accessible:

```bash
./config_certs.sh --validate DILITHIUM_LEVEL3
```

### Setting Up a Configuration

To create symbolic links for a specific certificate configuration:

```bash
./config_certs.sh --setup DILITHIUM_LEVEL3
```

This creates symbolic links in the certs directory pointing to the specific certificate files, making them easier to reference in benchmark scripts.

## Installation Requirements

Certificate generation requires the [oqs-provider](https://github.com/open-quantum-safe/oqs-provider) for OpenSSL. Installation instructions can be found in the `certs/oqs_prov_install/` directory.

To install the provider:

```bash
cd certs/oqs_prov_install
./install_openssl3_and_oqs_provider.sh -p /opt/oqs_openssl3
```

## Testing Certificates

You can test the generated certificates to ensure they are properly signed and valid:

```bash
./test_certs.sh --test-all
```

To test a specific certificate configuration:

```bash
./test_certs.sh --test DILITHIUM_LEVEL3
```