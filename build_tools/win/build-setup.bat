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

:: Compute CHISURF_VERSION early for conda build (PEP 440 compatible)
:: If already set in the environment, keep it.
if "%CHISURF_VERSION%"=="" (
     for /f "delims=" %%v in ('python -c "import datetime,re,subprocess,sys;\
try:\
 d=subprocess.check_output(['git','describe','--tags','--long','--match','v[0-9]*'],stderr=subprocess.DEVNULL,text=True).strip();\
except Exception:\
 d='';\
m=re.match(r'^(v[0-9.]+)-(\\d+)-g[0-9a-f]+$', d);\
def norm(tag):\
 tag=tag.lstrip('v');\
 parts=tag.split('.');\
 out=[];\
 for p in parts:\
  if not p.isdigit():\
   return None;\
  out.append(str(int(p)));\
 return '.'.join(out);\
if m:\
 base=norm(m.group(1));\
 dist=int(m.group(2));\
 if base is None:\
  ver=None;\
 elif dist==0:\
  ver=base;\
 else:\
  # Extract year from base tag for dev version\
  base_parts=base.split('.');\
  if len(base_parts)>=1:\
   year=base_parts[0];\
   ver=f'{year}.dev{dist}';\
  else:\
   ver=None;\
else:\
 ver=None;\
if not ver:\
 ver=datetime.datetime.now().strftime('%y.dev0');\
print(ver)"') do set "CHISURF_VERSION=%%v"
)
echo CHISURF_VERSION=%CHISURF_VERSION%

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
call mamba create -y --prefix "%APP_PATH%" chisurf conda -c local --force

:: Verify chisurf installation
echo Checking chisurf installation...
"%APP_PATH%\python.exe" -c "import chisurf; print('chisurf installed successfully!')" || (
    echo ERROR: chisurf is not installed properly.
    exit /b 1
)

:: Create a conda configuration file to ensure it works properly
echo Creating conda configuration...
mkdir "%APP_PATH%\.condarc" 2>nul
echo channels:> "%APP_PATH%\.condarc\config"
echo   - conda-forge>> "%APP_PATH%\.condarc\config"
echo   - defaults>> "%APP_PATH%\.condarc\config"
echo ssl_verify: true>> "%APP_PATH%\.condarc\config"

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

:: Get the version number from chisurf (optional diagnostics)
for /f "delims=" %%v in ('"%APP_PATH%\python.exe" -c "import chisurf.info; print(chisurf.info.__version__)"') do set "CHISURF_VERSION_BUILT=%%v"
echo CHISURF_VERSION_BUILT=%CHISURF_VERSION_BUILT%

:: Optional: Clean up the extracted environment
echo Cleaning up APP_PATH: %APP_PATH%...
rmdir /s /q %APP_PATH%

del setup.iss

echo Script finished successfully.
exit /b 0
