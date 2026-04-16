#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
APP_NAME="ChiSurf"
DIST_DIR="dist"
LINUX_DIST_DIR="$DIST_DIR/linux"
# Use current directory as root if not provided
REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"

# Ensure we are running in a context where build tools are available (e.g., via pixi run)
if ! command -v micromamba &> /dev/null; then
    echo "ERROR: micromamba not found. Please run this script via 'pixi run' or ensure micromamba is in your PATH."
    exit 1
fi

# Use mamba as the solver
export MAMBA_ROOT_PREFIX="$HOME/miniforge3"

# --- Arguments ---
BUILD_PKG=1
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD_PKG=0; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== $APP_NAME Linux AppImage Build ==="

# 1. Handle Versioning
CHI_VERSION="${CHI_VERSION:-}"
if [[ -z "$CHI_VERSION" ]]; then
    CHI_VERSION=$(python3 "$REPO_ROOT/rattler-recipe/generate_version.py" --print)
fi
echo "Version: $CHI_VERSION"

OUTPUT_APPIMAGE="$DIST_DIR/$APP_NAME-$CHI_VERSION-x86_64.AppImage"

# 2. Build Conda Package (if requested)
if [[ "$BUILD_PKG" == "1" ]]; then
    echo "[1/4] Building conda package..."
    # We use rattlel-build from the environment
    rattler-build build --recipe "$REPO_ROOT/rattler-recipe" --output-dir "$REPO_ROOT/conda-bld" --channel conda-forge --channel bioconda --test skip
fi

# 3. Locate Package
PKG="$(ls -t "$REPO_ROOT/conda-bld/linux-64/chisurf-"*.conda | head -n 1)"
if [[ -z "$PKG" ]]; then
    echo "ERROR: No conda package found in $REPO_ROOT/conda-bld/linux-64/"
    exit 1
fi
echo "Using package: $PKG"

# 4. Create Distribution Prefix
PREFIX="$REPO_ROOT/$LINUX_DIST_DIR/runtime"
mkdir -p "$(dirname "$PREFIX")"
rm -rf "$PREFIX"

echo "[2/4] Creating runtime environment at $PREFIX..."
# Explicitly add eigen and specify python version
micromamba create -y -p "$PREFIX" \
    "python=3.12" chisurf tttrlib eigen pybind11 "cmake<3.27" zstd libarchive openblas \
    -c "$REPO_ROOT/conda-bld" -c conda-forge -c bioconda \
    --no-channel-priority

# Install submodules
echo "[2.5/4] Installing submodules from modules directory ..."
# Set CMAKE_ARGS to help submodules find the environment's eigen
export CMAKE_ARGS="-DEIGEN3_INCLUDE_DIR=$PREFIX/include/eigen3"
export CMAKE_PREFIX_PATH="$PREFIX"

# Patch submodules to use environment's Eigen (legacy bundled Eigen often fails on new compilers)
# We look specifically for labellib which is known to have this issue
if [[ -d "$REPO_ROOT/modules/labellib/thirdparty/eigen" ]]; then
    echo "Patching LabelLib to use environment Eigen..."
    rm -rf "$REPO_ROOT/modules/labellib/thirdparty/eigen"
    mkdir -p "$REPO_ROOT/modules/labellib/thirdparty/eigen"
    cp -r "$PREFIX/include/eigen3/Eigen" "$REPO_ROOT/modules/labellib/thirdparty/eigen/"
    # Patch pybind11 too
    rm -rf "$REPO_ROOT/modules/labellib/thirdparty/pybind11/include/pybind11"
    mkdir -p "$REPO_ROOT/modules/labellib/thirdparty/pybind11/include"
    cp -r "$PREFIX/include/pybind11" "$REPO_ROOT/modules/labellib/thirdparty/pybind11/include/"
    # Force C++14 as required by modern Eigen
    python3 -c "import sys; content = open(sys.argv[1]).read(); open(sys.argv[1], 'w').write(content.replace('set(CMAKE_CXX_STANDARD 11)', 'set(CMAKE_CXX_STANDARD 14)'))" "$REPO_ROOT/modules/labellib/CMakeLists.txt"
    # Fix missing include for assert
    python3 -c "import sys; content = open(sys.argv[1]).read(); open(sys.argv[1], 'w').write('#include <cassert>\n' + content)" "$REPO_ROOT/modules/labellib/FlexLabel/include/FlexLabel/FlexLabel.h"
fi

# Add environment bin to PATH for submodule builds (so they find cmake, etc.)
export PATH="$PREFIX/bin:$PATH"

for mod in "$REPO_ROOT/modules"/*; do
    if [[ -d "$mod" ]] && [[ -f "$mod/setup.py" || -f "$mod/pyproject.toml" ]]; then
        echo "Installing submodule $(basename "$mod") ..."
        "$PREFIX/bin/python" -m pip install "$mod" --no-deps
    fi
done

# 5. Finalize Environment (Cleanup)
echo "[3/4] Cleaning runtime environment..."

# Strip bloat to keep the AppImage small
rm -rf "$PREFIX/include"
rm -rf "$PREFIX/share/doc" "$PREFIX/share/man" "$PREFIX/share/info"
rm -rf "$PREFIX/conda-meta"
find "$PREFIX/lib" -name "*.a" -delete
find "$PREFIX/lib" -name "*.la" -delete
find "$PREFIX/" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove pip and wheel (keep setuptools as pkg_resources depends on it)
rm -rf "$PREFIX/lib/python3.12/site-packages/pip"
rm -rf "$PREFIX/lib/python3.12/site-packages/wheel"
# Keep metadata directories as many packages (prompt_toolkit, etc.) use importlib.metadata

# Keep tests and examples as some packages (like tables) import them at runtime

# 6. Bundle into AppImage
echo "[4/4] Bundling with linuxdeploy..."
APPDIR=$(mktemp -d)
trap 'rm -rf "$APPDIR"' EXIT

mkdir -p "$APPDIR/usr"
cp -r "$PREFIX" "$APPDIR/usr/"

# Create AppRun launcher
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
SELF="$(readlink -f "${0}")"
APPDIR="$(dirname "${SELF}")"
export PATH="$APPDIR/usr/runtime/bin:$PATH"
export QT_PLUGIN_PATH="$APPDIR/usr/runtime/plugins"
export LD_LIBRARY_PATH="$APPDIR/usr/runtime/lib:$LD_LIBRARY_PATH"
exec "$APPDIR/usr/runtime/bin/python3" -m chisurf "$@"
EOF
chmod +x "$APPDIR/AppRun"

# Copy metadata and icons
cp "$REPO_ROOT/build_tools/linuxdeploy/chisurf.desktop" "$APPDIR/"
if [[ -f "$REPO_ROOT/chisurf/gui/resources/icons/cs_logo.png" ]]; then
    cp "$REPO_ROOT/chisurf/gui/resources/icons/cs_logo.png" "$APPDIR/chisurf-logo.png"
fi

# Run linuxdeploy
LINUXDEPLOY_BIN="$DIST_DIR/linuxdeploy-x86_64.AppImage"
if [[ ! -f "$LINUXDEPLOY_BIN" ]]; then
    curl -L -o "$LINUXDEPLOY_BIN" "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
    chmod +x "$LINUXDEPLOY_BIN"
fi

export OUTPUT="$OUTPUT_APPIMAGE"
rm -f "$OUTPUT"

# Extract and run to avoid FUSE issues in WSL/containers
# Use COMP=xz for maximum AppImage compression if appimagetool supports it
export COMP=xz
APPIMAGE_EXTRACT_AND_RUN=1 \
"$LINUXDEPLOY_BIN" --appdir "$APPDIR" --output appimage

echo "=== Build Complete ==="
echo "AppImage: $OUTPUT"
