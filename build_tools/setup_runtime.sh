#!/usr/bin/env bash
# Common script to setup the ChiSurf runtime environment using Micromamba.
# This is used by the AppImage build to ensure consistency.

set -euo pipefail

PREFIX="${1:-}"
if [[ -z "$PREFIX" ]]; then
    echo "Usage: $0 <prefix_path> [local_chisurf_path] [build_mode]"
    exit 1
fi
LOCAL_CHISURF_PATH="${2:-}"
BUILD_MODE="${3:-run}" # "run" or "build"

# Ensure micromamba is available
MICROMAMBA="micromamba"
if ! command -v micromamba &> /dev/null; then
    echo "micromamba not found. Attempting to download..."
    # Download micromamba if missing
    BIN_DIR="$(dirname "$(readlink -f "$0")")/bin"
    mkdir -p "$BIN_DIR"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$BIN_DIR" bin/micromamba
    MICROMAMBA="$BIN_DIR/bin/micromamba"
fi

echo "Using micromamba from: $(command -v "$MICROMAMBA" || echo "$MICROMAMBA")"
echo "=== Setting up ChiSurf Runtime at $PREFIX (Mode: $BUILD_MODE) ==="

CHANNELS="-c conda-forge -c bioconda --no-channel-priority"

# Dependencies list
DEPS=(
    "python=3.12"
    "numpy<2.0"
    "micromamba"
    "qtpy<2.0"
    "pyqtgraph=0.13.7"
    "mdtraj=1.11.1"
    "tttrlib=0.26.2"
    "scipy"
    "pandas"
    "matplotlib"
    "scikit-image"
    "pyqt"
    "pyqtwebengine"
    "numba"
    "guiqwt"
    "guidata"
    "typing-extensions"
    "pytools"
    "pyyaml"
    "markdown"
    "click"
    "click-didyoumean"
    "deprecation"
    "boost-cpp"
    "ipython"
    "notebook"
    "emcee"
    "pyopengl"
    "pytables"
    "python-docx"
    "qtconsole"
    "hmmlearn"
    "sympy"
    "zeus-mcmc"
    "pygments"
    "pyarrow"
    "boost-histogram"
    "fastmcp"
    "pymol-open-source"
)

if [[ "$BUILD_MODE" == "build" ]]; then
    DEPS+=(
        "cmake"
        "ninja"
        "cython<3"
        "swig"
        "eigen"
        "pybind11"
        "setuptools"
        "pip"
        "wheel"
        "gcc_linux-64"
        "gxx_linux-64"
        "libgomp"
    )
fi

# Create environment with ALL dependencies in one go to ensure solver consistency
"$MICROMAMBA" create -y --prefix "$PREFIX" $CHANNELS "${DEPS[@]}"

# If a local chisurf path (conda package) is provided, install it.
# Otherwise, we expect it to be installed later (e.g., from source via pip).
if [[ -n "$LOCAL_CHISURF_PATH" ]]; then
    echo "Installing local chisurf package from channel: $(dirname "$LOCAL_CHISURF_PATH")"
    "$MICROMAMBA" install -y --prefix "$PREFIX" \
        --channel "file://$(dirname "$(readlink -f "$LOCAL_CHISURF_PATH")")" \
        chisurf --no-deps
fi

echo "=== Runtime Setup Complete ==="

# Finalize Environment (Cleanup)
echo "Cleaning runtime environment..."
# Strip bloat
"$MICROMAMBA" clean -ay
rm -rf "$PREFIX/include" "$PREFIX/share/doc"
find "$PREFIX/" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
