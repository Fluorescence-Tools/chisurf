@echo off
setlocal

:: Enable long paths in git to avoid errors with deeply nested submodules
git config --global core.longpaths true

:: Generate Python resources
call pyrcc5 chisurf\gui\resources\resource.qrc -o chisurf\gui\resources\resource.py

:: Update submodules (Note: this Requires git in host/build and internet access if not pre-fetched)
git submodule sync --recursive
git submodule update --init --recursive --force


:: Build Burbulator C++ shared library
set BURB_SRC=src\csrc\burbulator
set BURB_BUILD=build\burbulator_cmake
set BURB_OUT_DIR=chisurf\plugins\core\acq\tcspc_devices\simulation
mkdir %BURB_BUILD%
cmake -S %BURB_SRC% -B %BURB_BUILD% -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --build %BURB_BUILD% --config Release
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
copy %BURB_BUILD%\bin\*.dll %BURB_OUT_DIR%\ /Y
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Build chinet (pure Python package)
cd %ROOT_DIR%
%PYTHON% -m pip install .\modules\chinet --no-deps --prefix=%PREFIX%
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Install sub-modules
cd %ROOT_DIR%
%PYTHON% -m pip install .\modules\clsmview --no-deps --prefix=%PREFIX%
%PYTHON% -m pip install .\modules\ndxplorer --no-deps --prefix=%PREFIX%
%PYTHON% -m pip install .\modules\quest --no-deps --prefix=%PREFIX%

:: Install imp-tricks (external IMP mixin, pure Python): use local checkout if
:: present (dev builds), otherwise clone from GitLab (CI builds).
if not defined IMP_TRICKS_REPO set IMP_TRICKS_REPO=https://gitlab.peulen.xyz/tpeulen/imp-tricks.git
if not defined IMP_TRICKS_REF set IMP_TRICKS_REF=main
if exist modules\imp-tricks\pyproject.toml (
    %PYTHON% -m pip install .\modules\imp-tricks --no-deps --prefix=%PREFIX%
) else (
    git clone --depth 1 --branch %IMP_TRICKS_REF% %IMP_TRICKS_REPO% %TEMP%\imp-tricks
    if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
    %PYTHON% -m pip install %TEMP%\imp-tricks --no-deps --prefix=%PREFIX%
)
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Clean pre-compiled Cython files to force regeneration
if exist chisurf\fluorescence\simulation\simulation_.cpp del /q chisurf\fluorescence\simulation\simulation_.cpp
if exist chisurf\structure\av\fps_.cpp del /q chisurf\structure\av\fps_.cpp
if exist chisurf\structure\potential\cPotentials_.cpp del /q chisurf\structure\potential\cPotentials_.cpp
if exist chisurf\math\reaction\reaction_.cpp del /q chisurf\math\reaction\reaction_.cpp

:: Replace dynamic version in chisurf/info.py
echo Building ChiSurf version: %PKG_VERSION%
copy chisurf\info.py chisurf\info.py.bak
powershell -Command "(Get-Content chisurf\info.py) -replace \"__version__ = .*\", \"__version__ = '%PKG_VERSION%'\" | Set-Content chisurf\info.py"

:: Main module install
%PYTHON% -m pip install . --no-deps --no-build-isolation -vv --prefix=%PREFIX%

:: Fix Windows launchers for relocation and GUI launching
if exist rattler-recipe\fix_launchers.py (
    echo Fixing Windows launchers...
    %PYTHON% rattler-recipe\fix_launchers.py
)

:: Restore info.py
move /Y chisurf\info.py.bak chisurf\info.py
