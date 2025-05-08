@echo off
setlocal enabledelayedexpansion

:: Set Paths
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

:: Debugging: Print paths
echo DIST_PATH=%DIST_PATH%
echo APP_PATH=%APP_PATH%
echo SOURCE_PATH=%SOURCE_PATH%
echo CONDA_RECIPE_FOLDER=%CONDA_RECIPE_FOLDER%

:: Default: Build the Conda package
set "BUILD_CONDA_PACKAGE=1"

:: Check command-line arguments: If /nobuild is passed, skip building Conda package
if /I "%1"=="/nobuild" (
    set "BUILD_CONDA_PACKAGE=0"
)

:: If /build flag is passed, build the Conda package
if %BUILD_CONDA_PACKAGE%==1 (
    echo Building Conda package...
    call conda mambabuild %CONDA_RECIPE_FOLDER%
) else (
    echo Skipping Conda package build...
)

:: Create necessary directories
if not exist "%DIST_PATH%" mkdir "%DIST_PATH%"
if not exist "%APP_PATH%" mkdir "%APP_PATH%"

:: Create the conda environment
echo Creating Conda environment at %APP_PATH%...
call mamba create -y --prefix "%APP_PATH%" chisurf -c local --force

:: Verify chisurf installation
echo Checking chisurf installation...
"%APP_PATH%\python.exe" -c "import chisurf; print('chisurf installed successfully!')" || (
    echo ERROR: chisurf is not installed properly.
    exit /b 1
)

:: Compile all Python source files into .pyc
echo Compiling Python files...
"%APP_PATH%\python.exe" -m compileall -q "%APP_PATH%"

:: Remove unnecessary files/directories
echo Cleaning up unnecessary files...
rmdir /s /q "%APP_PATH%\include"
rmdir /s /q "%APP_PATH%\Library\share\doc"
rmdir /s /q "%APP_PATH%\Library\share\IMP"
rmdir /s /q "%APP_PATH%\Library\include"
rmdir /s /q "%APP_PATH%\etc\conda\test-files"

:: Delete all .lib files from the Conda environment
echo Deleting all .lib files in %APP_PATH%...
for /r "%APP_PATH%\Library\lib" %%F in (*.lib) do del "%%F"

:: Generate Inno Setup script
echo Generating Inno Setup script...
python make_inno_setup.py

:: Create an installer with Inno Setup
echo Running Inno Setup...
"C:\Program Files (x86)\Inno Setup 6\Compil32.exe" /cc setup.iss

:: Get the version number from chisurf
for /f "delims=" %%v in ('"%APP_PATH%\python.exe" -c "import chisurf.info; print(chisurf.info.__version__)"') do set "CHISURF_VERSION=%%v"

:: Optional: Clean up the extracted environment
echo Cleaning up APP_PATH: %APP_PATH%...
rmdir /s /q %APP_PATH%

del setup.iss

echo Script finished successfully.
exit /b 0
