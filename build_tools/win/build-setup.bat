@echo off
setlocal enabledelayedexpansion

:: ---------------------------------------------------------------------------
:: build-setup.bat
::
:: Builds a self-contained Windows installer (via Inno Setup) from a staged
:: conda environment, using E:\miniforge3 only as the conda bootstrap root.
::
:: Prerequisites:
::   - Miniforge installation at E:\miniforge3 (or set BASE_CONDA_ROOT)
::   - conda available at %BASE_CONDA_ROOT%\Scripts\conda.exe
::   - Inno Setup 6 at "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
::   - MSVC build tools for Cython/CMake extensions
::
:: Usage (from project root):
::   build_tools\win\build-setup.bat
::
:: Or directly:
::   cd build_tools\win
::   build-setup.bat
::
:: ---------------------------------------------------------------------------

:: Resolve script and project root paths
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash from SCRIPT_DIR
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Project root is two levels up from build_tools\win
pushd "%SCRIPT_DIR%\..\.."
set "SOURCE_PATH=%CD%"
popd

:: Parse arguments
set "NO_CLEANUP="
set "USER_DIST_PATH="

:ArgLoop
if "%~1"=="" goto ArgDone
if /i "%~1"=="/nocleanup" (
    set "NO_CLEANUP=1"
    shift
    goto ArgLoop
)
if /i "%~1"=="/nobuild" (
    :: Existing flag from build_setup.py
    shift
    goto ArgLoop
)
if "%USER_DIST_PATH%"=="" (
    set "USER_DIST_PATH=%~1"
    shift
    goto ArgLoop
)
shift
goto ArgLoop
:ArgDone

set "DIST_PATH=%SOURCE_PATH%\dist"
if not "%USER_DIST_PATH%"=="" set "DIST_PATH=%USER_DIST_PATH%"

set "APP_PATH=%DIST_PATH%\win"
set "RUNTIME_ENV_PATH=%DIST_PATH%\runtime-base"
set "BUILDER_ENV_PATH=%DIST_PATH%\builder"
set "CONDA_PKGS_DIR=%DIST_PATH%\conda-pkgs"
set "BASE_CONDA_ROOT=%BASE_CONDA_ROOT%"
if not defined BASE_CONDA_ROOT set "BASE_CONDA_ROOT=E:\miniforge3"
set "RATTLER_RECIPE_FOLDER=%SOURCE_PATH%\rattler-recipe"
set "PIXI_MANIFEST=%SOURCE_PATH%\pixi.toml"


:: Normalize to absolute paths
for %%I in ("%DIST_PATH%")             do set "DIST_PATH=%%~fI"
for %%I in ("%APP_PATH%")              do set "APP_PATH=%%~fI"
for %%I in ("%RUNTIME_ENV_PATH%")      do set "RUNTIME_ENV_PATH=%%~fI"
for %%I in ("%BUILDER_ENV_PATH%")      do set "BUILDER_ENV_PATH=%%~fI"
for %%I in ("%CONDA_PKGS_DIR%")        do set "CONDA_PKGS_DIR=%%~fI"
for %%I in ("%BASE_CONDA_ROOT%")       do set "BASE_CONDA_ROOT=%%~fI"
for %%I in ("%SOURCE_PATH%")           do set "SOURCE_PATH=%%~fI"
for %%I in ("%RATTLER_RECIPE_FOLDER%") do set "RATTLER_RECIPE_FOLDER=%%~fI"
for %%I in ("%PIXI_MANIFEST%")         do set "PIXI_MANIFEST=%%~fI"

echo.
echo === ChiSurf Windows Build ===
echo SCRIPT_DIR        = %SCRIPT_DIR%
echo SOURCE_PATH       = %SOURCE_PATH%
echo DIST_PATH         = %DIST_PATH%
echo APP_PATH          = %APP_PATH%
echo RUNTIME_ENV_PATH  = %RUNTIME_ENV_PATH%
echo BUILDER_ENV_PATH  = %BUILDER_ENV_PATH%
echo CONDA_PKGS_DIR    = %CONDA_PKGS_DIR%
echo BASE_CONDA_ROOT   = %BASE_CONDA_ROOT%
echo RATTLER_RECIPE    = %RATTLER_RECIPE_FOLDER%
echo PIXI_MANIFEST     = %PIXI_MANIFEST%
echo.

if defined INNO_SETUP_EXE goto InnoCheck
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "INNO_SETUP_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "INNO_SETUP_EXE=C:\Program Files\Inno Setup 6\ISCC.exe"
:InnoCheck
if not defined INNO_SETUP_EXE (
    echo Inno Setup not found. Attempting to install via Chocolatey...
    choco install innosetup -y --no-progress
    if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "INNO_SETUP_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "INNO_SETUP_EXE=C:\Program Files\Inno Setup 6\ISCC.exe"
)
if not defined INNO_SETUP_EXE (
    echo ERROR: Inno Setup compiler not found even after install attempt. Set INNO_SETUP_EXE or install Inno Setup 6.
    exit /b 1
)

set "BASE_CONDA_EXE=%BASE_CONDA_ROOT%\Scripts\conda.exe"
if not exist "%BASE_CONDA_EXE%" (
    set "BASE_CONDA_EXE=%BASE_CONDA_ROOT%\Library\bin\micromamba.exe"
)
if not exist "%BASE_CONDA_EXE%" (
    set "BASE_CONDA_EXE=%BASE_CONDA_ROOT%-bin\micromamba.exe"
)
if not exist "%BASE_CONDA_EXE%" (
    echo ERROR: micromamba.exe not found at %BASE_CONDA_ROOT%
    exit /b 1
)

:: Clear any active conda state from a different installation before we invoke
:: the new base conda. Without this, an old base activation can leak stale
:: interpreter paths into solver/post-link steps.
set "PYTHONHOME="
set "PYTHONPATH="
set "CONDA_DEFAULT_ENV="
set "CONDA_EXE="
set "CONDA_PREFIX="
set "CONDA_PREFIX_1="
set "CONDA_PROMPT_MODIFIER="
set "CONDA_PYTHON_EXE="
set "CONDA_SHLVL="
set "_CONDA_EXE="
set "_CONDA_ROOT="
set "__CONDA_OPENSSL_CERT_DIR_SET="
set "__CONDA_OPENSSL_CERT_FILE_SET="
set "CONDA_NO_PLUGINS="
set "CONDA_OVERRIDE_CUDA="
set "CONDA_PKGS_DIRS=%CONDA_PKGS_DIR%"
:: Add Git to PATH (Git is typically in the user profile or Program Files)
set "PATH=%BASE_CONDA_ROOT%;%BASE_CONDA_ROOT%\Library\mingw-w64\bin;%BASE_CONDA_ROOT%\Library\usr\bin;%BASE_CONDA_ROOT%\Library\bin;%BASE_CONDA_ROOT%\Scripts;%BASE_CONDA_ROOT%\condabin;C:\Program Files\Git\cmd;%ProgramFiles%\Git\cmd;%UserProfile%\AppData\Local\Programs\Git\cmd;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0\"

:: ---------------------------------------------------------------------------
:: Ensure the reusable ChiSurf runtime environment exists.
:: This keeps the installer build off the system Python and gives us a stable
:: source env to clone into dist\win. The writable prefixes stay inside dist\
:: so the build does not depend on write access to E:\miniforge3\envs.
:: ---------------------------------------------------------------------------
if not exist "%DIST_PATH%" mkdir "%DIST_PATH%"
if not exist "%CONDA_PKGS_DIR%" mkdir "%CONDA_PKGS_DIR%"

if exist "%RUNTIME_ENV_PATH%" if not exist "%RUNTIME_ENV_PATH%\python.exe" (
    echo Removing incomplete base runtime environment ...
    rmdir /s /q "%RUNTIME_ENV_PATH%"
)
if not exist "%RUNTIME_ENV_PATH%\python.exe" (
    echo Creating base ChiSurf runtime environment at %RUNTIME_ENV_PATH% ...
    call "%BASE_CONDA_EXE%" create -y -p "%RUNTIME_ENV_PATH%" -c conda-forge --override-channels ^
        python=3.12.* ^
        pip ^
        "setuptools<81" ^
        wheel ^
        "cython>=0.29,<3.1" ^
        "numpy<2.0" ^
        openblas ^
        "cmake<3.27" ^
        ninja ^
        swig ^

        pythran ^
        "vs2022_win-64" ^
        "typing-extensions>=4.14" ^
        "pytools>=2024.0" ^
        pyyaml ^
        markdown ^
        click ^
        click-didyoumean ^
        "qtpy<2.0" ^
        pyqt ^
        "pyqtgraph=0.13.*" ^
        matplotlib ^
        scikit-image ^
        deprecation ^
        pandas ^
        scipy ^
        numba ^
        boost-cpp ^
        "mdtraj<1.10" ^
        ipython ^
        notebook ^
        emcee ^
        pyopengl ^
        pytables ^
        guidata ^
        guiqwt ^
        python-docx ^
        qtconsole ^
        hmmlearn ^
        sympy ^
        zeus-mcmc ^
        pygments ^
        pyarrow ^
        boost-histogram ^
        fastmcp ^
        hdf5
    if errorlevel 1 (
        echo ERROR: Failed to create base ChiSurf runtime environment
        exit /b 1
    )
)

:: ---------------------------------------------------------------------------
:: Bootstrap a small builder venv from the known-good ChiSurf runtime env so
:: setup generation never depends on system Python packages.
:: ---------------------------------------------------------------------------
set "BASE_ENV_PYTHON=%RUNTIME_ENV_PATH%\python.exe"
if not exist "%BASE_ENV_PYTHON%" (
    echo ERROR: Base env python.exe not found at %BASE_ENV_PYTHON%
    exit /b 1
)

echo Bootstrapping builder environment at %BUILDER_ENV_PATH% ...
if exist "%BUILDER_ENV_PATH%" (
    echo Removing existing builder environment ...
    rmdir /s /q "%BUILDER_ENV_PATH%"
)
"%BASE_ENV_PYTHON%" -m venv "%BUILDER_ENV_PATH%"
if errorlevel 1 (
    echo ERROR: Failed to create builder environment
    exit /b 1
)

set "BUILDER_PYTHON_EXE="
if exist "%BUILDER_ENV_PATH%\python.exe" set "BUILDER_PYTHON_EXE=%BUILDER_ENV_PATH%\python.exe"
if not defined BUILDER_PYTHON_EXE if exist "%BUILDER_ENV_PATH%\Scripts\python.exe" set "BUILDER_PYTHON_EXE=%BUILDER_ENV_PATH%\Scripts\python.exe"
if not defined BUILDER_PYTHON_EXE (
    echo ERROR: python.exe not found in %BUILDER_ENV_PATH%
    exit /b 1
)
call "%BUILDER_PYTHON_EXE%" -m pip install --upgrade pip >nul
call "%BUILDER_PYTHON_EXE%" -m pip install jinja2 >nul
if errorlevel 1 (
    echo ERROR: Failed to prepare builder environment
    exit /b 1
)
echo Builder environment ready.

:: -----------------------------------------------------------------------
:: Compute version (PEP 440 compatible) if not already set in environment
:: -----------------------------------------------------------------------
if not "%CHISURF_VERSION%"=="" goto SkipVersion
set "CHISURF_VERSION_FILE=%RATTLER_RECIPE_FOLDER%\version.generated.txt"
"%BUILDER_PYTHON_EXE%" "%SOURCE_PATH%\rattler-recipe\generate_version.py" --print > "%CHISURF_VERSION_FILE%"
if errorlevel 1 (
    echo ERROR: Failed to generate version using generate_version.py
    exit /b 1
)
set /p CHISURF_VERSION=<"%CHISURF_VERSION_FILE%"
del /q "%CHISURF_VERSION_FILE%" >nul 2>nul
if "%CHISURF_VERSION%"=="" (
    echo ERROR: Failed to generate version using generate_version.py
    exit /b 1
)
:SkipVersion
echo CHISURF_VERSION = %CHISURF_VERSION%

:: Skip version.json generation (handled by env vars)

:: -----------------------------------------------------------------------
:: entry_points.json is already generated by the pixi task dependency
:: (collect-entry-points runs before build-setup via depends-on).
:: Verify it exists; if not (direct bat invocation), generate it now.
:: -----------------------------------------------------------------------
if not exist "%RATTLER_RECIPE_FOLDER%\entry_points.json" (
    echo Generating entry_points.json ...
    "%BUILDER_PYTHON_EXE%" "%RATTLER_RECIPE_FOLDER%\collect_entry_points.py"
    if errorlevel 1 (
        echo ERROR: Failed to collect entry points
        exit /b 1
    )
)

:: -----------------------------------------------------------------------
:: Clone the known-good runtime env and install the current source into it.
:: -----------------------------------------------------------------------
echo.
echo [1/3] Staging runtime environment from %RUNTIME_ENV_PATH% ...

if exist "%APP_PATH%" (
    echo Removing existing environment ...
    rmdir /s /q "%APP_PATH%"
)
call "%BASE_CONDA_EXE%" create -y  -p "%APP_PATH%" --clone "%RUNTIME_ENV_PATH%"
if errorlevel 1 (
    echo ERROR: Failed to stage cloned runtime environment
    exit /b 1
)

set "APP_SITE_PACKAGES=%APP_PATH%\Lib\site-packages"
set "APP_CHISURF_DIR=%APP_SITE_PACKAGES%\chisurf"
set "APP_CONSTANTS_DIR=%APP_CHISURF_DIR%\settings\constants"
set "APP_PKG_RESOURCES_DIR=%APP_SITE_PACKAGES%\pkg_resources"
set "APP_PIP_VENDOR_PKG_RESOURCES_DIR=%APP_SITE_PACKAGES%\pip\_vendor\pkg_resources"
set "APP_DISTUTILS_HACK_DIR=%APP_SITE_PACKAGES%\_distutils_hack"
set "APP_PYTHONW_EXE=%APP_PATH%\pythonw.exe"
if not exist "%APP_CONSTANTS_DIR%" mkdir "%APP_CONSTANTS_DIR%"

set "PYTHON_EXE="
if exist "%APP_PATH%\python.exe" set "PYTHON_EXE=%APP_PATH%\python.exe"
if not defined PYTHON_EXE if exist "%APP_PATH%\Scripts\python.exe" set "PYTHON_EXE=%APP_PATH%\Scripts\python.exe"
if not defined PYTHON_EXE (
    echo ERROR: python.exe not found in %APP_PATH%
    exit /b 1
)
if not exist "%APP_PYTHONW_EXE%" (
    echo ERROR: pythonw.exe not found in staged runtime at %APP_PYTHONW_EXE%
    exit /b 1
)

echo Generating Python resources ...
if exist "%APP_PATH%\Scripts\pyrcc5.exe" (
    "%APP_PATH%\Scripts\pyrcc5.exe" "%SOURCE_PATH%\chisurf\gui\resources\resource.qrc" -o "%SOURCE_PATH%\chisurf\gui\resources\resource.py"
) else (
    echo WARNING: pyrcc5.exe not found, skipping resource generation
)

set "OLD_PATH=%PATH%"
set "PATH=%APP_PATH%;%APP_PATH%\Library\bin;%APP_PATH%\Scripts;%PATH%"

echo Installing ChiSurf submodules ...
call "%PYTHON_EXE%" -m pip install "%SOURCE_PATH%\modules\chinet" --no-deps --no-build-isolation
if errorlevel 1 (
    echo ERROR: Failed to install chinet submodule
    exit /b 1
)
call "%PYTHON_EXE%" -m pip install "%SOURCE_PATH%\modules\clsmview" --no-deps --no-build-isolation
call "%PYTHON_EXE%" -m pip install "%SOURCE_PATH%\modules\ndxplorer" --no-deps --no-build-isolation
call "%PYTHON_EXE%" -m pip install "%SOURCE_PATH%\modules\quest" --no-deps --no-build-isolation


echo Installing ChiSurf source into cloned environment ...
call "%PYTHON_EXE%" -m pip install "%SOURCE_PATH%" --no-deps --no-build-isolation
if errorlevel 1 (
    echo ERROR: Failed to install ChiSurf into staged environment
    exit /b 1
)

set "PATH=%OLD_PATH%"

echo Restoring setuptools runtime compatibility ...
call "%BASE_CONDA_EXE%" install -y  -p "%APP_PATH%" -c conda-forge --override-channels "setuptools<81"
if errorlevel 1 (
    echo ERROR: Failed to restore setuptools runtime compatibility
    exit /b 1
)

echo Syncing ChiSurf runtime files into staged environment ...
copy /y "%SOURCE_PATH%\chisurf\common.py" "%APP_CHISURF_DIR%\common.py" >nul
if errorlevel 1 (
    echo ERROR: Failed to sync common.py into staged environment
    exit /b 1
)
copy /y "%SOURCE_PATH%\chisurf\settings\constants\*.json" "%APP_CONSTANTS_DIR%\" >nul
if errorlevel 1 (
    echo ERROR: Failed to sync constants JSON files into staged environment
    exit /b 1
)
if not exist "%APP_PKG_RESOURCES_DIR%" (
    echo Restoring pkg_resources compatibility shim ...
    xcopy /e /i /y "%APP_PIP_VENDOR_PKG_RESOURCES_DIR%" "%APP_PKG_RESOURCES_DIR%" >nul
    if errorlevel 1 (
        echo ERROR: Failed to restore pkg_resources compatibility shim
        exit /b 1
    )
)
if not exist "%APP_DISTUTILS_HACK_DIR%" if exist "%RUNTIME_ENV_PATH%\Lib\site-packages\_distutils_hack" (
    echo Restoring _distutils_hack support files ...
    xcopy /e /i /y "%RUNTIME_ENV_PATH%\Lib\site-packages\_distutils_hack" "%APP_DISTUTILS_HACK_DIR%" >nul
    if errorlevel 1 (
        echo ERROR: Failed to restore _distutils_hack support files
        exit /b 1
    )
)
if not exist "%APP_SITE_PACKAGES%\distutils-precedence.pth" if exist "%RUNTIME_ENV_PATH%\Lib\site-packages\distutils-precedence.pth" (
    copy /y "%RUNTIME_ENV_PATH%\Lib\site-packages\distutils-precedence.pth" "%APP_SITE_PACKAGES%\distutils-precedence.pth" >nul
)

echo Installing tttrlib via pip (Windows) ...
call "%PYTHON_EXE%" -m pip install tttrlib --no-cache-dir --no-deps
if errorlevel 1 (
    echo WARNING: Could not install tttrlib
)

echo Installing labellib via pip ...
call "%PYTHON_EXE%" -m pip install labellib --no-cache-dir --no-deps
if errorlevel 1 (
    echo WARNING: Could not install labellib
)

echo Verifying chisurf installation ...
"%PYTHON_EXE%" -c "import sys; sys.path.insert(0, r'%APP_PATH%\Lib\site-packages'); import pkg_resources, tttrlib, chinet, LabelLib, chisurf;    print('chisurf OK:', chisurf.__version__)"
if errorlevel 1 (
    echo ERROR: Failed to verify packaged runtime dependencies
    exit /b 1
)

:: Fix Windows launchers for relocation and GUI launching in the staged environment
echo Fixing launchers in %APP_PATH% ...
if exist "%RATTLER_RECIPE_FOLDER%\fix_launchers.py" (
    "%PYTHON_EXE%" "%RATTLER_RECIPE_FOLDER%\fix_launchers.py" "%APP_PATH%"
)

:: Pre-compile Python files
echo Compiling .pyc files ...
"%PYTHON_EXE%" -m compileall -qq "%APP_PATH%"

:: Strip dev-only bloat
echo Stripping headers, docs, .lib files ...
if exist "%APP_PATH%\include"              rmdir /s /q "%APP_PATH%\include"
if exist "%APP_PATH%\Library\share\doc"   rmdir /s /q "%APP_PATH%\Library\share\doc"
if exist "%APP_PATH%\Library\share\man"   rmdir /s /q "%APP_PATH%\Library\share\man"
if exist "%APP_PATH%\Library\share\info"  rmdir /s /q "%APP_PATH%\Library\share\info"
if exist "%APP_PATH%\Library\share\IMP"   rmdir /s /q "%APP_PATH%\Library\share\IMP"
if exist "%APP_PATH%\Library\include"     rmdir /s /q "%APP_PATH%\Library\include"
if exist "%APP_PATH%\etc\conda"           rmdir /s /q "%APP_PATH%\etc\conda"
if exist "%APP_PATH%\conda-meta"          rmdir /s /q "%APP_PATH%\conda-meta"

:: Remove Python bytecode (safe), but keep tests/examples as some packages import them
echo Removing Python bytecode ...
powershell -Command "Get-ChildItem -Path '%APP_PATH%' -Filter '__pycache__' -Recurse | Remove-Item -Force -Recurse"

:: Remove pip, wheel (keep setuptools as pkg_resources depends on it)
echo Removing pip, wheel ...
rmdir /s /q "%APP_PATH%\Lib\site-packages\pip"
rmdir /s /q "%APP_PATH%\Lib\site-packages\wheel"
:: Keep metadata directories as many packages (prompt_toolkit, etc.) use importlib.metadata

for /r "%APP_PATH%\Library\lib" %%F in (*.lib) do del /q "%%F"
for /r "%APP_PATH%\Library\lib" %%F in (*.a) do del /q "%%F"

:: -----------------------------------------------------------------------
:: Generate Inno Setup script and build setup.exe
:: -----------------------------------------------------------------------
echo.
echo [3/3] Building Windows installer ...

:: create_installer_script.py must run from the build_tools\win directory
cd /d "%SCRIPT_DIR%"
call "%BUILDER_PYTHON_EXE%" create_installer_script.py
if errorlevel 1 (
    echo ERROR: Failed to generate Inno Setup script
    exit /b 1
)

if not defined INNO_SETUP_EXE (
    echo ERROR: INNO_SETUP_EXE is not set.
    exit /b 1
)

if not exist installer_config.iss (
    echo ERROR: installer_config.iss was not generated.
    exit /b 1
)

echo Compiling installer with Inno Setup ...
"%INNO_SETUP_EXE%" installer_config.iss
if errorlevel 1 (
    echo ERROR: Inno Setup compilation failed
    exit /b 1
)
del /q installer_config.iss

:InnoSkip
echo Inno Setup step complete.
goto Done


:Done
:: -----------------------------------------------------------------------
:: Report version and clean up
:: -----------------------------------------------------------------------
set "CHISURF_VERSION_BUILT=%CHISURF_VERSION%"
echo.
echo === Build complete ===
echo Version: %CHISURF_VERSION_BUILT%
echo Installer: %DIST_PATH%\ChiSurf-Windows-Setup-%CHISURF_VERSION_BUILT%.exe

:: Smoke-test installer by performing a clean silent install into dist\install-smoke
echo Running installer smoke test ...
set "SMOKE_INSTALL_DIR=%DIST_PATH%\install-smoke"
set "SMOKE_INSTALLER=%DIST_PATH%\ChiSurf-Windows-Setup-%CHISURF_VERSION_BUILT%.exe"
if exist "%SMOKE_INSTALL_DIR%" rmdir /s /q "%SMOKE_INSTALL_DIR%"
"%SMOKE_INSTALLER%" /CURRENTUSER /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR="%SMOKE_INSTALL_DIR%"
if errorlevel 1 (
    echo ERROR: Installer smoke test failed to install
    exit /b 1
)
if not exist "%SMOKE_INSTALL_DIR%\python.exe" (
    echo ERROR: Smoke install missing python.exe
    exit /b 1
)
if not exist "%SMOKE_INSTALL_DIR%\pythonw.exe" (
    echo ERROR: Smoke install missing pythonw.exe
    exit /b 1
)
"%SMOKE_INSTALL_DIR%\python.exe" -c "import pkg_resources,tttrlib,chinet,LabelLib,chisurf; print('smoke-ok', chisurf.__version__)"
if errorlevel 1 (
    echo ERROR: Smoke install failed runtime import verification
    exit /b 1
)
if exist "%SMOKE_INSTALL_DIR%" rmdir /s /q "%SMOKE_INSTALL_DIR%"
echo Smoke test passed.

:: Remove the staging environment (unless NO_CLEANUP is set)
if "%NO_CLEANUP%"=="1" (
    echo NO_CLEANUP set, preserving staging environments.
    goto FinalExit
)

echo Removing staging environment ...
rmdir /s /q "%APP_PATH%"
if exist "%BUILDER_ENV_PATH%" (
    echo Removing builder environment ...
    rmdir /s /q "%BUILDER_ENV_PATH%"
)

:FinalExit
cd /d "%SOURCE_PATH%"
echo Done.
exit /b 0

