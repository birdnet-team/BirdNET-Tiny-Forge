#!/bin/bash
#   Copyright 2024 BirdNET-Team
#   Copyright 2024 fold ecosystemics
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
set -e

# Check if the path to tflite-micro is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-where-tflite-micro-will-be-cloned-and-built>"
    exit 1
fi

TFLITE_MICRO_PATH=$1
TFLITE_COMMIT="aeed46b"

# Check for python-dev
PYTHON_INCLUDE=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
if [ ! -f "${PYTHON_INCLUDE}/Python.h" ]; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "python${PYTHON_VERSION}-dev is not installed. Please install it first."
    exit 1
fi

# Check for bazel
if ! command -v bazel &> /dev/null; then
    echo "bazel is not installed. Please install it first."
    exit 1
fi

# Clone repo if not exists
# This is a patched fork which pins tensorflow version to the one used in this repo,
# and makes the signal library available to python
if [ ! -d "$TFLITE_MICRO_PATH" ]; then
    git clone git@github.com:gio-foldai/tflite-micro.git "$TFLITE_MICRO_PATH"
    cd "$TFLITE_MICRO_PATH"
    git checkout $TFLITE_COMMIT
    cd ..
fi

## Run bazel build
cd "$TFLITE_MICRO_PATH"
bazel build //python/tflite_micro:whl.dist

echo "Done."
