#!/usr/bin/env bash
# Build an OSX Application (.app) for ChiSurf
#
# Example:
#
#     $ build-osx-app.sh $HOME/Applications/ChiSurf.app
#

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}


export ICON_FILE=icon.png
export CONDA_ENVIRONMENT_YAML=env_osx.yml

# The directory of the build-osx-app.sh script
export SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

function print_usage() {
    echo "build-osx-app.sh [-i] [--template] $APP_NAME.app
Build python module as an $APP_NAME OSX application bundle ($APP_NAME.app).

This expects that the python module can be launched by:

python -m module_name

where module_name is the name of the module that is bundled as an osx app.

The bundle will include the python module and a conda environment that is
used to run the python module.

NOTE: this script should be run from build_tools in source root directory.
Options:
    -f --environment_file  Path to conda environment file (default: environment.yml)
    -i --icon              Path to the icon file (default: icon.png)
    -n --name              Name of .app file
    -m --module            Python module
    -p --module_path       Path to the parent of the python module
    -h --help              Display help
    -o --output_path       The path to which the .app is written

Example:
  ./build-osx-app.sh -f=../../environment.yml -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=../../ -o=../../dist
    "
    exit
}

for i in "$@"
do
case $i in
    -f=*|--environment_file=*)
    CONDA_ENVIRONMENT_YAML="${i#*=}"
    shift # past argument=value
    ;;
    -i=*|--icon=*)
    ICON_FILE=$(get_abs_filename "${i#*=}")
    shift # past argument=value
    ;;
    -m=*|--module=*)
    PYTHON_MODULE="${i#*=}"
    shift # past argument=value
    ;;
    -n=*|--name=*)
    APP_NAME="${i#*=}"
    shift # past argument=value
    ;;
    -o=*|--output_path=*)
    OUTPUT_PATH="${i#*=}"
    shift # past argument=value
    ;;
    -p=*|--module_path=*)
    PYTHON_MODULE_PATH="${i#*=}"
    shift # past argument=value
    ;;
    -h|--help)
    print_usage
    ;;
    *)
    print_usage
    exit
    ;;
    esac
    shift
done


# The target directory of the .app
mkdir "$OUTPUT_PATH"
OUTPUT_PATH=$(cd "$OUTPUT_PATH" && pwd)
PYTHON_MODULE_PATH=$(cd "$PYTHON_MODULE_PATH" && pwd)

export APP_FOLDER=${1:-$OUTPUT_PATH/$APP_NAME.app}
mkdir -p "$APP_FOLDER"
APP_FOLDER=$(cd "$APP_FOLDER" && pwd)

mamba env create -f $CONDA_ENVIRONMENT_YAML --prefix "$APP_FOLDER/Contents" --force
conda activate "$APP_FOLDER/Contents"
mamba install -y nomkl jinja2
mkdir -p "$APP_FOLDER/Contents/MacOS"
mkdir -p "$APP_FOLDER/Contents/Resources"

cd $PYTHON_MODULE_PATH
export SITE_PACKAGE_PATH=`$APP_FOLDER/Contents/bin/python -c 'import site; print(site.getsitepackages()[0])'`
cp -R $PYTHON_MODULE $SITE_PACKAGE_PATH

function print_values() {
    echo "Input values:"
    echo "-----------------------"
    echo "Conda Environment YAML: $CONDA_ENVIRONMENT_YAML"
    echo "Icon File:              $ICON_FILE"
    echo "Python Module:          $PYTHON_MODULE"
    echo "App Name:               $APP_NAME"
    echo "Output Path:            $OUTPUT_PATH"
    echo "Python Module Path:     $PYTHON_MODULE_PATH"
    echo "Site package Path:      $SITE_PACKAGE_PATH"
    echo "App folder:             $APP_FOLDER"
    echo "-----------------------"
}
print_values


# generate icons file
cd $SCRIPT_DIR
python generate-iconset.py $SCRIPT_DIR/resources/AppIcon.png
python generate-iconset.py resources/VolumeIcon.png

# also update the icon with fileicon
$SCRIPT_DIR/fileicon set "$APP_FOLDER" "$SCRIPT_DIR/resources/AppIcon.icns"

# update the Info.plist file and create a entry point
cd $PYTHON_MODULE_PATH
$SCRIPT_DIR/create_app_plist.py \
  --module $PYTHON_MODULE \
  --output $APP_FOLDER/Contents/Info.plist \
  --executable $APP_NAME \
  -i $SCRIPT_DIR/resources/AppIcon.icns \
  -p $SCRIPT_DIR/plist_template \
  -t $SCRIPT_DIR/launch_template
cd $APP_FOLDER/Contents
./bin/python -m compileall .

cd $SCRIPT_DIR
./create-dmg/create-dmg \
  --volname "$APP_NAME Installer" \
  --volicon "./resources/VolumeIcon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "$APP_NAME.app" 200 190 \
  --hide-extension "$APP_NAME.app" \
  --app-drop-link 600 185 \
  --skip-jenkins \
  --sandbox-safe \
  --no-internet-enable \
  "$OUTPUT_PATH/$APP_NAME-Installer.dmg" \
  "$APP_FOLDER/.."
