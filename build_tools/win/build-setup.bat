@echo off
set "DIST_PATH=%CD%\..\..\dist"
set "SCRIPT_PATH=%~dp0"
set "APP_PATH=%CD%\..\..\dist\win"
set "SOURCE_PATH=%CD%\..\.."
set "CONDA_RECIPE_FOLDER=%CD%\..\..\conda-recipe"

:: Normalize paths to absolute paths
for %%I in ("%DIST_PATH%") do set "DIST_PATH=%%~fI"
for %%I in ("%APP_PATH%") do set "APP_PATH=%%~fI"
for %%I in ("%SOURCE_PATH%") do set "SOURCE_PATH=%%~fI"
for %%I in ("%CONDA_RECIPE_FOLDER%") do set "CONDA_RECIPE_FOLDER=%%~fI"

:: Default behavior: Do not build conda package
set "BUILD_CONDA_PACKAGE=0"

:: Check for command-line arguments
if /I "%1"=="/build" (
    set "BUILD_CONDA_PACKAGE=1"
)

:: If /build flag is passed, build the Conda package
if %BUILD_CONDA_PACKAGE%==1 (
    echo Building Conda package...
    call conda mambabuild %CONDA_RECIPE_FOLDER%
) else (
    echo Skipping Conda package build...
)

:: Create necessary directories
md %DIST_PATH%
md %APP_PATH%

:: Create the conda environment
call mamba create -y --prefix %APP_PATH% chisurf -c local -c tpeulen --force

:: Compile all Python source files into .pyc
python -m compileall -q %APP_PATH%

:: Remove unused files/directories
rmdir /s /q %APP_PATH%\include
rmdir /s /q %APP_PATH%\Library\share\doc
rmdir /s /q %APP_PATH%\Library\share\IMP
rmdir /s /q %APP_PATH%\Library\include
rmdir /s /q %APP_PATH%\etc\conda\test-files

:: Delete all .lib files from the Conda environment
echo Deleting all .lib files in %APP_PATH%...
for /r "%APP_PATH%\Library\lib" %%F in (*.lib) do del "%%F"

:: Generate Inno Setup script
python make_inno_setup.py

:: Create an installer with Inno Setup
"C:\Program Files (x86)\Inno Setup 6\Compil32.exe" /cc setup.iss

:: Cleaning step: Purge the APP_PATH folder
:: echo Cleaning up APP_PATH: %APP_PATH%...
:: rmdir /s /q %APP_PATH%

echo Script finished.

