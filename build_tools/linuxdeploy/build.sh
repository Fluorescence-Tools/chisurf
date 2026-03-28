#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "$SCRIPT_DIR/../.." > /dev/null
REPO_ROOT="$(pwd)"
popd > /dev/null

DIST_PATH="$REPO_ROOT/dist"
APP_PATH="$DIST_PATH/linux"
RATTLER_RECIPE_DIR="$REPO_ROOT/rattler-recipe"
OUTPUT_DIR="$REPO_ROOT/conda-bld"
APPDIR="$DIST_PATH/AppDir"
APP_NAME="ChiSurf"
BUILD_RATTLER_PACKAGE=1

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD_RATTLER_PACKAGE=0; shift ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "=== ChiSurf Linux Build ==="
echo "REPO_ROOT        = $REPO_ROOT"
echo "DIST_PATH        = $DIST_PATH"
echo "APP_PATH         = $APP_PATH"
echo "OUTPUT_DIR       = $OUTPUT_DIR"
echo ""

CHISURF_VERSION="${CHISURF_VERSION:-}"
if [[ -z "$CHISURF_VERSION" ]]; then
    CHISURF_VERSION="$(cd "$REPO_ROOT" && python rattler-recipe/generate_version.py --print)"
    if [[ -z "$CHISURF_VERSION" ]]; then
        echo "ERROR: Failed to generate version"
        exit 1
    fi
fi
echo "CHISURF_VERSION = $CHISURF_VERSION"
echo '{"version": "'"$CHISURF_VERSION"'"}' > "$RATTLER_RECIPE_DIR/version.json"

if [[ "$BUILD_RATTLER_PACKAGE" == "1" ]]; then
    echo ""
    echo "[1/4] Building conda package ..."
    rattler-build build --recipe "$RATTLER_RECIPE_DIR" --output-dir "$OUTPUT_DIR" --test skip
else
    echo "[1/4] Skipping package build (--no-build)"
fi

echo ""
echo "[2/4] Finding built conda package ..."

PLATFORM_SUBDIR="linux-64"
CHISURF_PKG=""
for f in "$OUTPUT_DIR/$PLATFORM_SUBDIR"/chisurf-*.conda; do
    if [[ -f "$f" ]]; then
        CHISURF_PKG="$f"
        break
    fi
done
if [[ -z "$CHISURF_PKG" ]]; then
    echo "ERROR: No chisurf conda package found in $OUTPUT_DIR/$PLATFORM_SUBDIR"
    exit 1
fi
echo "Found package: $CHISURF_PKG"

echo ""
echo "[3/4] Creating distribution environment at $APP_PATH ..."
rm -rf "$APP_PATH"
mkdir -p "$(dirname "$APP_PATH")"

micromamba create -y \
    --prefix "$APP_PATH" \
    python chisurf \
    "$CHISURF_PKG" \
    -c conda-forge -c bioconda \
    --no-channel-priority

if [[ ! -f "$APP_PATH/bin/python" ]]; then
    echo "ERROR: python not found in $APP_PATH"
    exit 1
fi

echo "Compiling .pyc files ..."
"$APP_PATH/bin/python" -m compileall -qq "$APP_PATH"

echo "Stripping dev-only bloat ..."
rm -rf "$APP_PATH/include" "$APP_PATH/share/doc" "$APP_PATH/share/IMP" "$APP_PATH/share/info" "$APP_PATH/share/man"
find "$APP_PATH/lib" -name "*.a" -delete 2>/dev/null || true
find "$APP_PATH/lib" -name "*.la" -delete 2>/dev/null || true
find "$APP_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$APP_PATH" -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "[4/4] Building AppImage with linuxdeploy ..."

rm -rf "$APPDIR"
mkdir -p "$APPDIR"

ln -s "$APP_PATH" "$APPDIR/usr"

cat > "$APPDIR/AppRun" << 'LAUNCHER'
#!/usr/bin/env bash
SELF="$(readlink -f "$0")"
APPDIR="$(dirname "$SELF")"
export PYTHONNOUSERS1
export PATH="$APPDIR/usr/bin:$PATH"
export QT_PLUGIN_PATH="$APPDIR/usr/plugins"
export LD_LIBRARY_PATH="$APPDIR/usr/lib:${LD_LIBRARY_PATH:-}"
cd "$APPDIR/usr"
exec "$APPDIR/usr/bin/python" -m chisurf "$@"
LAUNCHER
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/chisurf.desktop" << 'DESKTOP'
[Desktop Entry]
Version=1.0
Type=Application
Name=ChiSurf
Comment=Time-resolved fluorescence analysis
Exec=chisurf %F
Terminal=false
Icon=chisurf-logo
Categories=Science;Education;
StartupNotify=true
DESKTOP

if [[ -f "$REPO_ROOT/chisurf/gui/resources/icons/cs_logo.png" ]]; then
    cp "$REPO_ROOT/chisurf/gui/resources/icons/cs_logo.png" "$APPDIR/chisurf-logo.png"
fi

LINUXDEPLOY_BIN="$DIST_PATH/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_QT="$DIST_PATH/linuxdeploy-plugin-qt-x86_64.AppImage"

if [[ ! -f "$LINUXDEPLOY_BIN" ]]; then
    echo "Downloading linuxdeploy ..."
    curl -L -o "$LINUXDEPLOY_BIN" \
        "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
    chmod +x "$LINUXDEPLOY_BIN"
fi

if [[ ! -f "$LINUXDEPLOY_QT" ]]; then
    echo "Downloading linuxdeploy-plugin-qt ..."
    curl -L -o "$LINUXDEPLOY_QT" \
        "https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage"
    chmod +x "$LINUXDEPLOY_QT"
fi

export OUTPUT="$DIST_PATH/ChiSurf-x86_64.AppImage"
rm -f "$OUTPUT"

LINUXDEPLOY_PLUGIN_QT_PATH="$LINUXDEPLOY_QT" "$LINUXDEPLOY_BIN" \
    --appdir "$APPDIR" \
    --plugin qt \
    --output appimage

echo ""
echo "=== Build complete ==="
echo "Version:    $CHISURF_VERSION"
echo "AppImage:   $OUTPUT"
echo "Done."
