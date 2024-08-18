#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 [Debug|Release]"
  exit 1
fi

# Set the build type based on the argument
BUILD_TYPE="$1"

# Validate the argument
if [ "$BUILD_TYPE" != "Debug" ] && [ "$BUILD_TYPE" != "Release" ]; then
  echo "Invalid build type: $BUILD_TYPE"
  echo "Usage: $0 [Debug|Release]"
  exit 1
fi

# Create the build directory if it doesn't exist
if [ ! -d "build/$BUILD_TYPE" ]; then
  mkdir -p "build/$BUILD_TYPE"
fi

# Configure and build the project
cmake -S . -B "build/$BUILD_TYPE" -DCMAKE_BUILD_TYPE="${BUILD_TYPE^}"
cmake --build "build/$BUILD_TYPE"

pytest tests/
