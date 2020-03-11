#!/usr/bin/env bash
# Build an OSX Applicaiton (.app) for ChiSurf
#
# Example:
#
#     $ build-osx-app.sh $HOME/Applications/ChiSurf.app
#

#export APP_NAME="ChiSurf"
#export PYTHON_MODULE=chisurf
#export SOFTWARE_NAME=ChiSurf

export ICON_FILE=icon.png
export CONDA_ENVIRONMENT_YAML=environment.yml
# The directory of the build-osx-app.sh script
export SCRIPT_DIR="."
SCRIPT_DIR=$("pwd")


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
    -p --module_path       Location of the python module
    -h --help              Display help

Example:
  ./build-osx-app.sh -f=../environment.yml -i=../chisurf/gui/resources/icons/cs_logo.png -n=ChiSurf -m=chisurf -p=..
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
    ICON_FILE="${i#*=}"
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
export APP_FOLDER=${1:-$APP_NAME.app}
# Create a MacOS app folder structure
mkdir "$APP_FOLDER"
# convert to absolute path
APP_FOLDER="$(cd "$(dirname "$APP_FOLDER")" && pwd)/$(basename "$APP_FOLDER")"

# Create a new conda environment in the target Resources
conda env create -f $CONDA_ENVIRONMENT_YAML --prefix "$APP_FOLDER/Contents" --force
# We will simpliy copy over the directory containing the code
# Hence make sure that all necessary extensions are up to data
#
export PATH="$APP_FOLDER/Contents/bin:$PATH"
# install possibility to hook into osx GUI event loop
conda install python.app --prefix "$APP_FOLDER/Contents"
cd $PYTHON_MODULE_PATH
export PYTHON_MODULE_PATH=$PYTHON_MODULE
export PYTHON_MODULE_PATH=$(python -c "import $PYTHON_MODULE; import pathlib; print(pathlib.Path($PYTHON_MODULE.__file__).parent.absolute())")
export SITE_PACKAGE_LOCATION=$APP_FOLDER/Contents/lib/python3.7/site-packages
# update SITE_PACKAGE_LOCATION
SITE_PACKAGE_LOCATION=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -R $PYTHON_MODULE_PATH $SITE_PACKAGE_LOCATION
cd $SCRIPT_DIR

# update the Info.plist file and create a entry point
mkdir "$APP_FOLDER/Contents/MacOS"
mkdir "$APP_FOLDER/Contents/Resources"
./create_app_plist.py --module $PYTHON_MODULE --output "$APP_FOLDER"/Contents/Info.plist -e "$APP_NAME"

# Update the icons
./fileicon set "$APP_FOLDER" "$ICON_FILE"
