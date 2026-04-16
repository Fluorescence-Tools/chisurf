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

if [[ "$BUILD_RATTLER_PACKAGE" == "1" ]]; then
    rattler-build build --recipe "$RATTLER_RECIPE_DIR" --output-dir "$OUTPUT_DIR" --channel conda-forge --channel bioconda --test skip
fi

CHISURF_PKG=$(find "$OUTPUT_DIR" -name "chisurf-*.conda" | head -n 1)

echo "[3/4] Creating distribution environment at $APP_PATH ..."
rm -rf "$APP_PATH"
# Explicitly include libomp (from llvm-openmp)
micromamba create -y --prefix "$APP_PATH" python=3.12 tttrlib libffi openblas libgfortran5 llvm-openmp chisurf -c file://$(dirname "$CHISURF_PKG") -c conda-forge -c bioconda --no-channel-priority

echo "[3.5/4] Installing submodules ..."
for mod in "$REPO_ROOT/modules"/*; do
    if [[ -d "$mod" ]] && [[ -f "$mod/setup.py" || -f "$mod/pyproject.toml" ]]; then
        echo "Installing submodule $(basename "$mod") ..."
        "$APP_PATH/bin/pip" install -e "$mod" --no-deps
    fi
done

# Build app bundle
APP_BUNDLE="$DIST_PATH/$APP_NAME.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/bin"
mkdir -p "$APP_BUNDLE/Contents/lib"
mkdir -p "$APP_BUNDLE/Contents/Resources"
cp -a "$APP_PATH/bin/"* "$APP_BUNDLE/Contents/bin/"
cp -a "$APP_PATH/lib/python3.12" "$APP_BUNDLE/Contents/lib/"
cp -a "$APP_PATH/lib/"*.dylib "$APP_BUNDLE/Contents/lib/"
# Remove libc++ - it is an OS-provided library on macOS; bundling it causes
# SIGBUS (EXC_ARM_DA_ALIGN) crashes when the conda version conflicts with the system one.
rm -f "$APP_BUNDLE/Contents/lib/libc++"*.dylib

# Add Icon
ICON_SRC="$REPO_ROOT/chisurf/gui/resources/icons/cs_logo.icns"
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$APP_BUNDLE/Contents/Resources/ChiSurf.icns"
fi

# Write PkgInfo (required for macOS bundle recognition)
echo -n "APPL????" > "$APP_BUNDLE/Contents/PkgInfo"

# Write Info.plist (required for Cocoa/CoreText initialization; missing plist causes CTFontDrawGlyphs SIGBUS crash)
CHISURF_VERSION=${CHISURF_VERSION:-$(cd "$REPO_ROOT" && git describe --tags --always 2>/dev/null || echo "26.0")}
cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIconFile</key>
    <string>ChiSurf</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>xyz.peulen.chisurf</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$CHISURF_VERSION</string>
    <key>CFBundleVersion</key>
    <string>$CHISURF_VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2026 Thomas-Otavio Peulen</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
PLIST

cat > "$APP_BUNDLE/Contents/MacOS/$APP_NAME" << 'LAUNCHER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYVER=python3.12
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCRIPT_DIR/Contents/lib/$PYVER/site-packages"
export PATH="$SCRIPT_DIR/Contents/bin:$PATH"
export QT_PLUGIN_PATH="$SCRIPT_DIR/Contents/lib/$PYVER/site-packages/PyQt5/Qt5/plugins"

# Workarounds for Qt5 crashes on macOS ARM64 (Apple Silicon)
# Enable CoreText (required for Ribbon) and use low-resolution compatibility mode for stability.
# Scaling is handled by a global application font override in chisurf/gui/__init__.py
export QT_MAC_DISABLE_APP_NAP=1
export QT_MAC_WANTS_BEST_RESOLUTION_OPENGL_SURFACE=0

cd "$HOME"
exec "$SCRIPT_DIR/Contents/bin/python" -m chisurf "$@"
LAUNCHER
chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

echo "[4/4] Creating DMG ..."
DMG_NAME="$DIST_PATH/$APP_NAME-$CHISURF_VERSION.dmg"
rm -f "$DMG_NAME"

# Create a temporary folder for the DMG content to add /Applications link
DMG_TMP="$DIST_PATH/dmg_tmp"
rm -rf "$DMG_TMP"
mkdir -p "$DMG_TMP"
cp -a "$APP_BUNDLE" "$DMG_TMP/"
ln -s /Applications "$DMG_TMP/Applications"

hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_TMP" -ov -format UDZO "$DMG_NAME"
rm -rf "$DMG_TMP"

# Also create a generic link for the CI artifact upload if needed
ln -sf "$(basename "$DMG_NAME")" "$DIST_PATH/$APP_NAME.dmg"

echo "Created artifact: $DMG_NAME"
echo "Created artifact: $DIST_PATH/$APP_NAME.dmg"
