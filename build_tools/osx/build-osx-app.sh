#!/usr/bin/env bash
# Build an OSX Application (.app) for ChiSurf

set -e  # Exit on error
set -u  # Treat unset variables as errors
set -o pipefail  # Catch errors in pipelines

# Ensure script runs from its own directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default Values
ICON_FILE="icon.png"
PYTHON_VERSION="3.10"
PYTHON_MODULE=""
APP_NAME="ChiSurf"
OUTPUT_PATH="$SCRIPT_DIR/dist"
PYTHON_MODULE_PATH=""

# Function to get absolute filename
get_abs_filename() {
  realpath "$1"
}

# Function to print usage
print_usage() {
    echo "Usage: build-osx-app.sh [options]"
    echo "Options:"
    echo "    -i, --icon          Path to the icon file (default: icon.png)"
    echo "    -n, --name          Name of the .app file (default: ChiSurf)"
    echo "    -t, --python        Python version (default: 3.10)"
    echo "    -m, --module        Python module to bundle"
    echo "    -p, --module_path   Path to the module’s parent directory"
    echo "    -o, --output_path   Output directory for the .app (default: ./dist)"
    echo "    -h, --help          Display this help message"
    exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i=*|--icon=*) ICON_FILE=$(get_abs_filename "${1#*=}") ;;
        -m=*|--module=*) PYTHON_MODULE="${1#*=}" ;;
        -n=*|--name=*) APP_NAME="${1#*=}" ;;
        -o=*|--output_path=*) OUTPUT_PATH=$(get_abs_filename "${1#*=}") ;;
        -p=*|--module_path=*) PYTHON_MODULE_PATH=$(get_abs_filename "${1#*=}") ;;
        -t=*|--python=*) PYTHON_VERSION="${1#*=}" ;;
        -h|--help) print_usage ;;
        *) echo "Unknown option: $1"; print_usage ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$PYTHON_MODULE" || -z "$PYTHON_MODULE_PATH" ]]; then
    echo "Error: Python module and module path must be specified."
    print_usage
fi

# Resolve paths
mkdir -p "$OUTPUT_PATH"
OUTPUT_PATH=$(cd "$OUTPUT_PATH" && pwd)
PYTHON_MODULE_PATH=$(cd "$PYTHON_MODULE_PATH" && pwd)

APP_FOLDER="$OUTPUT_PATH/$APP_NAME.app"
mkdir -p "$APP_FOLDER/Contents"

# Create Conda environment inside the bundle
echo "Creating Conda environment in $APP_FOLDER/Contents..."
mamba create --prefix "$APP_FOLDER/Contents" --force -y python="$PYTHON_VERSION"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$APP_FOLDER/Contents"

# Install Python module
echo "Installing module $PYTHON_MODULE..."
mamba install -y "$PYTHON_MODULE" --use-local

# Create necessary directories
mkdir -p "$APP_FOLDER/Contents/MacOS"
mkdir -p "$APP_FOLDER/Contents/Resources"

# Copy Python module
cd "$PYTHON_MODULE_PATH"
SITE_PACKAGE_PATH=$("$APP_FOLDER/Contents/bin/python" -c 'import site; print(site.getsitepackages()[0])')
cp -R "$PYTHON_MODULE" "$SITE_PACKAGE_PATH"

# Print values
function print_values() {
    echo "============================="
    echo " App Build Configuration"
    echo "-----------------------------"
    echo " App Name:           $APP_NAME"
    echo " Icon File:          $ICON_FILE"
    echo " Python Version:     $PYTHON_VERSION"
    echo " Python Module:      $PYTHON_MODULE"
    echo " Module Path:        $PYTHON_MODULE_PATH"
    echo " Output Path:        $OUTPUT_PATH"
    echo " Site Packages Path: $SITE_PACKAGE_PATH"
    echo " App Folder:         $APP_FOLDER"
    echo "============================="
}
print_values

# Generate icons
cd "$SCRIPT_DIR"
python generate-iconset.py "$SCRIPT_DIR/resources/AppIcon.png"
python generate-iconset.py "$SCRIPT_DIR/resources/VolumeIcon.png"

# Set app icon
"$SCRIPT_DIR/fileicon" set "$APP_FOLDER" "$SCRIPT_DIR/resources/AppIcon.icns"

# Generate Info.plist and executable
cd "$PYTHON_MODULE_PATH"
"$SCRIPT_DIR/create_app_plist.py" \
  --module "$PYTHON_MODULE" \
  --output "$APP_FOLDER/Contents/Info.plist" \
  --executable "$APP_NAME" \
  -i "$SCRIPT_DIR/resources/AppIcon.icns" \
  -p "$SCRIPT_DIR/plist_template" \
  -t "$SCRIPT_DIR/launch_template"

# Compile Python files
#cd "$APP_FOLDER/Contents"
#./bin/python -m compileall .

# --- Create DMG with Drag-and-Drop Installation ---
# Instead of creating a temporary folder in /tmp, we now create the staging folder in the output directory.

STAGING_DIR="$OUTPUT_PATH/${APP_NAME}-dmg"
rm -rf "$STAGING_DIR"         # Remove any previous staging folder
mkdir -p "$STAGING_DIR"
echo "Staging DMG content in: $STAGING_DIR"

# Move the .app bundle into the staging folder
mv "$APP_FOLDER" "$STAGING_DIR/"
echo "Moved app bundle to staging folder."

# Create a symlink to the /Applications folder for drag-and-drop installation
ln -s /Applications "$STAGING_DIR/Applications"
echo "Created symlink to /Applications in staging folder."

# Optionally, add a custom background image if available.
if [[ -f "$SCRIPT_DIR/resources/dmg-background.png" ]]; then
    mkdir -p "$STAGING_DIR/.background"
    cp "$SCRIPT_DIR/resources/dmg-background.png" "$STAGING_DIR/.background/"
    echo "Custom background image added."
else
    echo "No custom background image found. Skipping background image."
fi

# Debug: List staging folder contents
echo "Staging folder contents:"
ls -la "$STAGING_DIR"

# Define DMG parameters
DMG_NAME="$OUTPUT_PATH/$APP_NAME-Installer.dmg"
VOL_NAME="$APP_NAME Installer"
echo "Creating DMG file: $DMG_NAME with volume name: $VOL_NAME"

# Remove existing DMG if present
rm -f "$DMG_NAME"

# Increase size to 5 GB (adjust if needed)
DMG_SIZE="4.5g"

# Create a read/write DMG from the staging folder.
hdiutil create -volname "$VOL_NAME" -srcfolder "$STAGING_DIR" -ov -format UDRW -size "$DMG_SIZE" "$DMG_NAME"

# Convert the read/write DMG to a compressed, read-only DMG for distribution
hdiutil convert "$DMG_NAME" -format UDZO -o "${DMG_NAME%.dmg}-final.dmg"

# Replace the original DMG with the final version
mv "${DMG_NAME%.dmg}-final.dmg" "$DMG_NAME"

# Clean up the staging folder
rm -rf "$STAGING_DIR"
echo "Cleaned up staging folder."

echo "DMG created successfully at: $DMG_NAME"
