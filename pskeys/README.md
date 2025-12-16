# PSK Key Management

Pre-Shared Key (PSK) management system for CoAP-DTLS benchmarks. This is essential for the PSK security mode.

## Overview

The `psk_manager.sh` script handles generation, activation, and deployment of PSK keys used in DTLS-PSK mode benchmarks.

## Usage

```bash
./pskeys/psk_manager.sh <command> [arguments]
```

## Commands

### Generate Keys

Create new keys with different bit strengths:

```bash
# Generate a new 256-bit key (default)
./pskeys/psk_manager.sh generate

# Generate keys with specific bit lengths
./pskeys/psk_manager.sh generate 128
./pskeys/psk_manager.sh generate 256
./pskeys/psk_manager.sh generate 384
./pskeys/psk_manager.sh generate 512
```

Generated keys are stored as `psk_<bits>_<timestamp>.key` files.

### List Keys

Show all available PSK keys:

```bash
./pskeys/psk_manager.sh list
```

### Show Current Key

Display the currently active key:

```bash
./pskeys/psk_manager.sh current
```

### Activate Key

Set a specific key as the active key for benchmarks:

```bash
./pskeys/psk_manager.sh activate psk_256_12345678.key
```

This creates/updates the `active_psk.txt` symlink.

### Deploy Keys

Synchronize keys to a remote server (e.g., Raspberry Pi). You may need to update the IP address in the script:

```bash
./pskeys/psk_manager.sh deploy
```

This ensures both client and server use the same cryptographic material.

## File Structure

```text
pskeys/
├── psk_manager.sh       # Management script
├── active_psk.txt       # Symlink to active key
├── psk_256_*.key        # 256-bit keys
├── psk_384_*.key        # 384-bit keys
└── ...
```

## Quick Start

```bash
# Generate and activate a 256-bit key
./pskeys/psk_manager.sh generate 256
./pskeys/psk_manager.sh activate $(ls pskeys/psk_256_*.key | head -1 | xargs basename)

# Verify
./pskeys/psk_manager.sh current
```

## Notes

- Keys are hex-encoded random bytes
- The `active_psk.txt` file is used by benchmark scripts to locate the current key
- When running benchmarks on a remote server, use `deploy` to sync keys
