#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
APP_NAME="ChiSurf"
DIST_DIR="dist"
LINUX_DIST_DIR="$DIST_DIR/linux"
OUTPUT_APPIMAGE="$DIST_DIR/$APP_NAME-x86_64.AppImage"

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
echo "{\"version\": \"$CHI_VERSION\"}" > "$REPO_ROOT/rattler-recipe/version.json"

# 2. Build Conda Package (if requested)
if [[ "$BUILD_PKG" == "1" ]]; then
    echo "[1/4] Building conda package..."
    # We use rattlel-build from the environment
    rattler-build build --recipe "$REPO_ROOT/rattler-recipe" --output-dir "$REPO_ROOT/conda-bld" --test skip
fi

# 3. Locate Package
PKG="$(ls -t "$REPO_ROOT/conda-bld/linux-64/chisurf-"*.conda | head -n 1)"
if [[ -z "$PKG" ]]; then
    echo "ERROR: No conda package found in $REPO_ROOT/conda-bld/linux-64/"
    exit 1
fi
echo "Using package: $PKG"

# 4. Create Distribution Prefix
PREFIX="$LINUX_DIST_DIR/runtime"
mkdir -p "$(dirname "$PREFIX")"
rm -rf "$PREFIX"

echo "[2/4] Creating runtime environment at $PREFIX..."
bash "$REPO_ROOT/build_tools/setup_runtime.sh" "$PREFIX" "$PKG"

# 5. Finalize Environment (Cleanup)
echo "[3/4] Cleaning runtime environment..."
# Use the python in the prefix to ensure we are modifying the right env
PYTHON_BIN="$PREFIX/bin/python"

# Strip bloat to keep the AppImage small
rm -rf "$PREFIX/include" "$PREFIX/share/doc"
find "$PREFIX/" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

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
APPIMAGE_EXTRACT_AND_RUN=1 \
"$LINUXDEPLOY_BIN" --appdir "$APPDIR" --output appimage

echo "=== Build Complete ==="
echo "AppImage: $OUTPUT"
