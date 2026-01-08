#!/bin/bash

# Remove old build directory if exists
rm -rf ./build

# Create and enter new build directory
mkdir build
cd build

# Run cmake and make
cmake ..
make