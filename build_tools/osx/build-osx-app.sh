#!/usr/bin/env bash
# Build an OSX Applicaiton (.app) for ChiSurf
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
export CONDA_ENVIRONMENT_YAML=environment.yml
# The directory of the build-osx-app.sh script
export SCRIPT_DIR="."
SCRIPT_DIR=$("pwd")
echo $SCRIPT_DIR

function print_usage() {
    echo "build-osx-app.sh [-i] [--template] $APP_NAME.app
Build python module as an $APP_NAME OSX application bundle ($APP_NAME.app).

This expects that the python module can be launed by:

python -m module_name

where module_name is the name of the module that is bundeled as an osx app.

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
  ./build-osx-app.sh -f=../environment.yml -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=.. -o=../dist
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
export APP_FOLDER=${1:-$OUTPUT_PATH/$APP_NAME.app}
APP_FOLDER="$(cd "$(dirname "$APP_FOLDER")" && pwd)/$(basename "$APP_FOLDER")"
echo "App folder: $APP_FOLDER"
mkdir "$APP_FOLDER"

source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f $CONDA_ENVIRONMENT_YAML --prefix "$APP_FOLDER/Contents" --force
conda activate "$APP_FOLDER/Contents"
conda install -y nomkl jinja2
mkdir "$APP_FOLDER/Contents/MacOS"
mkdir "$APP_FOLDER/Contents/Resources"

cd $PYTHON_MODULE_PATH
cp -R $PYTHON_MODULE "$APP_FOLDER/Contents"

# generate icns file
cd $SCRIPT_DIR
python generate-iconset.py $SCRIPT_DIR/resources/AppIcon.png
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
cd $SCRIPT_DIR

echo Remove files and folders from content folder: "$CONTENT_FOLDER"
while read p; do
  echo "removing: $APP_FOLDER/Contents/$p"
  rm -rf $CONTENT_FOLDER$p
done <"$SCRIPT_DIR/remove_list.txt"
