@echo off
set "DIST_PATH=..\..\dist"
set "SCRIPT_PATH=%~dp0"
set "APP_PATH=..\..\dist\win"
set "SOURCE_PATH=..\.."
set "CONDA_RECIPE_FOLDER=..\..\conda-recipe"

rem Default behavior: Do not build conda package
set "BUILD_CONDA_PACKAGE=0"

rem Check for command-line arguments
if /I "%1"=="/build" (
    set "BUILD_CONDA_PACKAGE=1"
)

rem If /build flag is passed, build the Conda package
if %BUILD_CONDA_PACKAGE%==1 (
    echo Building Conda package...
    call conda mambabuild %CONDA_RECIPE_FOLDER%
) else (
    echo Skipping Conda package build...
)

rem Create necessary directories
md %DIST_PATH%
md %APP_PATH%

rem Create the conda environment
call mamba create -y --prefix %APP_PATH% chisurf -c local -c tpeulen --force

rem Generate Inno Setup script
python make_inno_setup.py

rem Optionally deactivate conda environment
rem call conda deactivate
rem call conda activate base

rem Create an installer with Inno Setup
"C:\Program Files (x86)\Inno Setup 6\Compil32.exe" /cc setup.iss

rem Cleaning step: Purge the APP_PATH folder
:: echo Cleaning up APP_PATH: %APP_PATH%...
rmdir /s /q %APP_PATH%

echo Script finished.

