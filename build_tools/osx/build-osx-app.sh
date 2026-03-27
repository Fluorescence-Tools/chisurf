#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pushd "$SCRIPT_DIR/../.." > /dev/null
REPO_ROOT="$(pwd)"
popd > /dev/null

DIST_PATH="$REPO_ROOT/dist"
APP_PATH="$DIST_PATH/osx"
RATTLER_RECIPE_DIR="$REPO_ROOT/rattler-recipe"
OUTPUT_DIR="$REPO_ROOT/conda-bld"
APP_NAME="ChiSurf"
BUILD_RATTLER_PACKAGE=1

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD_RATTLER_PACKAGE=0; shift ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo.
echo "=== ChiSurf macOS Build ==="
echo "REPO_ROOT        = $REPO_ROOT"
echo "DIST_PATH        = $DIST_PATH"
echo "APP_PATH         = $APP_PATH"
echo "OUTPUT_DIR       = $OUTPUT_DIR"
echo "RATTLER_RECIPE   = $RATTLER_RECIPE_DIR"
echo.

if [[ ! -x "$(command -v pixi)" ]]; then
    if [[ -x "$HOME/.pixi/bin/pixi" ]]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    else
        echo "ERROR: pixi not found. Install from https://pixi.sh"
        exit 1
    fi
fi

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

if [[ ! -f "$RATTLER_RECIPE_DIR/entry_points.json" ]]; then
    echo "Generating entry_points.json ..."
    (cd "$REPO_ROOT" && python rattler-recipe/collect_entry_points.py) || {
        echo "WARNING: collect_entry_points.py failed, continuing"
    }
fi

if [[ "$BUILD_RATTLER_PACKAGE" == "1" ]]; then
    echo.
    echo "[1/4] Building conda package ..."
    (cd "$REPO_ROOT" && pixi run build-pkg)
else
    echo "[1/4] Skipping package build (--no-build)"
fi

echo.
echo "[2/4] Finding built conda package ..."

CHISURF_PKG=""
for f in "$OUTPUT_DIR/osx-64"/chisurf-*.conda; do
    if [[ -f "$f" ]]; then
        CHISURF_PKG="$f"
        break
    fi
done
if [[ -z "$CHISURF_PKG" ]]; then
    echo "ERROR: No chisurf conda package found in $OUTPUT_DIR/osx-64"
    exit 1
fi
echo "Found package: $CHISURF_PKG"

echo.
echo "[3/4] Creating distribution environment at $APP_PATH ..."
rm -rf "$APP_PATH"
mkdir -p "$(dirname "$APP_PATH")"

CHISURF_PKG_URL="${CHISURF_PKG//\//\\/}"
(cd "$REPO_ROOT" && pixi run micromamba create -y \
    --prefix "$APP_PATH" \
    python \
    chisurf \
    tttrlib \
    "file:///$CHISURF_PKG_URL" \
    -c conda-forge \
    -c bioconda \
    --no-channel-priority)

if [[ ! -f "$APP_PATH/bin/python" ]]; then
    echo "ERROR: python not found in $APP_PATH"
    exit 1
fi

echo "Compiling .pyc files ..."
"$APP_PATH/bin/python" -m compileall -qq "$APP_PATH"

echo "Stripping dev-only bloat ..."
rm -rf "$APP_PATH/include"
rm -rf "$APP_PATH/share/doc"
rm -rf "$APP_PATH/share/IMP"
rm -rf "$APP_PATH/share/info"
find "$APP_PATH/lib" -name "*.a" -delete 2>/dev/null || true
find "$APP_PATH/lib" -name "*.la" -delete 2>/dev/null || true
find "$APP_PATH" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$APP_PATH" -name "*.pyo" -delete 2>/dev/null || true
find "$APP_PATH" -name "*.pyc" -delete 2>/dev/null || true

echo.
echo "[4/4] Building .app bundle ..."

APP_BUNDLE="$DIST_PATH/$APP_NAME.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"

echo "Moving micromamba environment into bundle ..."
cp -a "$APP_PATH/." "$APP_BUNDLE/Contents/"

cat > "$APP_BUNDLE/Contents/MacOS/$APP_NAME" << LAUNCHER
#!/usr/bin/env bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
SCRIPT_DIR="\$(dirname "\$(cd "\$(dirname "\$0")" && pwd)")"
export PYTHONNOUSERSITE=1
export PATH="\$SCRIPT_DIR/Contents/bin:\$SCRIPT_DIR/Contents:\$PATH"
export QT_PLUGIN_PATH="\$SCRIPT_DIR/Contents/plugins"
export DYLD_LIBRARY_PATH="\$SCRIPT_DIR/Contents/lib:\${DYLD_LIBRARY_PATH:-}"
cd "\$SCRIPT_DIR/Contents"
exec "\$SCRIPT_DIR/Contents/bin/python" -m chisurf "\$@"
LAUNCHER
chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

if command -v python &> /dev/null; then
    (cd "$REPO_ROOT" && python build_tools/osx/create_app_plist.py \
        --module chisurf \
        --output "$APP_BUNDLE/Contents/Info.plist" \
        --executable "$APP_NAME" \
        -i "$REPO_ROOT/chisurf/gui/resources/icons/cs_logo.png" \
        -p "$SCRIPT_DIR/plist_template" \
        -t "$SCRIPT_DIR/launch_template") || {
        echo "WARNING: create_app_plist.py failed, generating minimal Info.plist"
        cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleDisplayName</key><string>$APP_NAME</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$CHISURF_VERSION</string>
    <key>CFBundleSignature</key><string>????</string>
    <key>CFBundleVersion</key><string>$CHISURF_VERSION</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
    }
else
    echo "WARNING: python not available for create_app_plist.py, generating minimal Info.plist"
    cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleDisplayName</key><string>$APP_NAME</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$CHISURF_VERSION</string>
    <key>CFBundleSignature</key><string>????</string>
    <key>CFBundleVersion</key><string>$CHISURF_VERSION</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
fi

echo.
echo "[4/4] Creating DMG installer ..."

STAGING_DIR="$DIST_PATH/${APP_NAME}-dmg"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp -a "$APP_BUNDLE" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

DMG_PATH="$DIST_PATH/ChiSurf-Installer.dmg"
rm -f "$DMG_PATH"

DMG_SIZE="5g"
hdiutil create \
    -volname "$APP_NAME Installer" \
    -srcfolder "$STAGING_DIR" \
    -ov -format UDRW \
    -size "$DMG_SIZE" \
    "$DMG_PATH"

hdiutil convert "$DMG_PATH" -format UDZO -o "${DMG_PATH%.dmg}-compressed.dmg"
mv "${DMG_PATH%.dmg}-compressed.dmg" "$DMG_PATH"

rm -rf "$STAGING_DIR"
rm -rf "$APP_BUNDLE"
rm -rf "$APP_PATH"

echo.
echo "=== Build complete ==="
echo "Version:    $CHISURF_VERSION"
echo "Installer:  $DMG_PATH"
echo "Done."
