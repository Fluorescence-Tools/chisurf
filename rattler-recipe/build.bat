@echo off
setlocal

:: Enable long paths in git to avoid errors with deeply nested submodules
git config --global core.longpaths true

:: Generate Python resources
call pyrcc5 chisurf\gui\resources\resource.qrc -o chisurf\gui\resources\resource.py

:: Update submodules (Note: this Requires git in host/build and internet access if not pre-fetched)
git submodule sync --recursive
git submodule update --init --recursive --force


:: Build chinet (pure Python package)
cd %ROOT_DIR%
%PYTHON% -m pip install .\modules\chinet --no-deps --prefix=%PREFIX%
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Install sub-modules
cd %ROOT_DIR%
%PYTHON% -m pip install .\modules\clsmview --no-deps --prefix=%PREFIX%
%PYTHON% -m pip install .\modules\ndxplorer --no-deps --prefix=%PREFIX%
%PYTHON% -m pip install .\modules\quest --no-deps --prefix=%PREFIX%

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
