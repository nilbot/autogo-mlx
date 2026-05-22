#!/usr/bin/env bash
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building C++ Go/MCTS extension ==="
echo "Repo root: $REPO_DIR"

CPP_DIR="$REPO_DIR/third_party/autogo/src/alpha_go/cpp"
if [ ! -d "$CPP_DIR" ]; then
    echo "Error: Submodule directory not found at $CPP_DIR" >&2
    exit 1
fi

# Get Python executable path from uv virtual environment of the parent workspace
PYTHON_PATH=$(uv run which python)
echo "Using Python executable: $PYTHON_PATH"

cd "$CPP_DIR"

# Configure build with CMake
cmake -S . -B build \
    -DPython_EXECUTABLE="$PYTHON_PATH" \
    -DPython3_EXECUTABLE="$PYTHON_PATH" \
    -DCMAKE_BUILD_TYPE=Release

# Build targets
cmake --build build -j$(sysctl -n hw.ncpu || echo 2)

echo "=== Build Successful ==="
