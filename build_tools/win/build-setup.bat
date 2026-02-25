@echo off
setlocal enabledelayedexpansion

:: Set Paths
set "DIST_PATH=%TEMP%\chisurf_dist"
set "SCRIPT_PATH=%~dp0"
set "APP_PATH=%TEMP%\chisurf_dist\win"
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
if not "%CHISURF_VERSION%"=="" goto SkipVersion
call :ComputeVersion
:SkipVersion
echo CHISURF_VERSION=%CHISURF_VERSION%

:: Parse command-line arguments
set "CROOT_OPT="
set "LOCAL_CHANNEL=-c local"
:parse_args
if "%~1"=="" goto done_args
if /I "%~1"=="/nobuild" (
    set "BUILD_CONDA_PACKAGE=0"
    shift
    goto parse_args
)
if /I "%~1"=="--croot" (
    set "CROOT_OPT=--croot %~2"
    set "LOCAL_CHANNEL=-c %~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args
:done_args

:: Check for C++ compiler early
where cl.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: C++ compiler ^(cl.exe^) not found in PATH.
    echo Please install Visual Studio Build Tools and ensure they are available in your environment.
    echo See https://visualstudio.microsoft.com/visual-cpp-build-tools/
    exit /b 1
)

:: If /build flag is passed, build the Conda package
if %BUILD_CONDA_PACKAGE%==1 (
    echo Building Conda package...
    call conda mambabuild !CROOT_OPT! "%CONDA_RECIPE_FOLDER%"
) else (
    echo Skipping Conda package build...
)

:: Create necessary directories
if not exist "%DIST_PATH%" mkdir "%DIST_PATH%"
if not exist "%APP_PATH%" mkdir "%APP_PATH%"

:: Create the conda environment
echo Creating Conda environment at %APP_PATH%...
call mamba create -y --prefix "%APP_PATH%" chisurf conda !LOCAL_CHANNEL! --force --no-shortcuts

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
"%APP_PATH%\python.exe" -m compileall -qq "%APP_PATH%"

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
python create_installer_script.py

:: Create an installer with Inno Setup
echo Running Inno Setup...
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer_config.iss

:: Get the version number from chisurf (optional diagnostics)
for /f "delims=" %%v in ('"%APP_PATH%\python.exe" -c "import chisurf.info; print(chisurf.info.__version__)"') do set "CHISURF_VERSION_BUILT=%%v"
echo CHISURF_VERSION_BUILT=%CHISURF_VERSION_BUILT%

:: Optional: Clean up the extracted environment
echo Cleaning up APP_PATH: %APP_PATH%...
rmdir /s /q "%APP_PATH%"

del installer_config.iss

echo Script finished successfully.
exit /b 0

:ComputeVersion
set "TEMP_VER_PY=%TEMP%\chisurf_version_%RANDOM%.py"
(
echo import datetime, re, subprocess, sys, os
echo os.chdir^("%SOURCE_PATH%"^)
echo try:
echo     d = subprocess.check_output^(['git', 'describe', '--tags', '--long', '--match', 'v[0-9]*'], stderr=subprocess.DEVNULL, text=True^).strip^(^)
echo except Exception:
echo     d = ''
echo m = re.match^(r'^^(v[0-9.]+^)-(\d+^)-g[0-9a-f]+$', d^)
echo def norm^(tag^):
echo     tag = tag.lstrip^('v'^)
echo     parts = tag.split^('.'^)
echo     out = []
echo     for p in parts:
echo         if not p.isdigit^(^):
echo             return None
echo         out.append^(str^(int^(p^)^)^)
echo     return '.'.join^(out^)
echo if m:
echo     base = norm^(m.group^(1^)^)
echo     dist = int^(m.group^(2^)^)
echo     if base:
echo         if dist == 0:
echo             ver = base
echo         else:
echo             year = base.split^('.'^)[0]
echo             year = year[-2:]
echo             ver = f'{year}.dev{dist}'
echo     else:
echo         ver = None
echo else:
echo     ver = None
echo if not ver:
echo     today = datetime.datetime.now^(^)
echo     year = today.strftime^('%%y'^)
echo     try:
echo         dist = subprocess.check_output^(['git', 'rev-list', '--count', 'HEAD'], stderr=subprocess.DEVNULL, text=True^).strip^(^)
echo     except Exception:
echo         dist = '0'
echo     ver = f'{year}.dev{dist}'
echo print^(ver^)
) > "%TEMP_VER_PY%"

for /f "delims=" %%v in ('python "%TEMP_VER_PY%"') do set "CHISURF_VERSION=%%v"
del "%TEMP_VER_PY%"
goto :eof
