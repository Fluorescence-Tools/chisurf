@echo off
setlocal

:: Enable long paths in git to avoid errors with deeply nested submodules
git config --global core.longpaths true

:: Generate Python resources
call pyrcc5 chisurf\gui\resources\resource.qrc -o chisurf\gui\resources\resource.py

:: Update submodules (Note: this Requires git in host/build and internet access if not pre-fetched)
git submodule sync --recursive
git submodule update --init --recursive --force

:: Build labellib
set "ROOT_DIR=%cd%"
cd modules\labellib
git fetch --tags
git checkout tags/2020.10.05

:: Set pybind11 version
pushd thirdparty\pybind11
git checkout v2.13
git pull
popd

:: Configure the build using CMake
cmake -S . -B build -G Ninja ^
    -DPYTHON_EXECUTABLE="%PYTHON%" ^
    -DPYTHON_LIBRARY_OUTPUT_DIRECTORY="%SP_DIR%" ^
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="%SP_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --install build --prefix %PREFIX%
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Build chinet
cd %ROOT_DIR%\modules\chinet
if exist build rmdir /s /q build
mkdir build
cd build
cmake .. -G "Ninja" ^
 -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
 -DCMAKE_PREFIX_PATH="%PREFIX%" ^
 -DBUILD_PYTHON_INTERFACE=ON ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="%SP_DIR%" ^
 -DCMAKE_SWIG_OUTDIR="%SP_DIR%" ^
 -DPython_ROOT_DIR="%PREFIX%\bin" ^
 -DBUILD_LIBRARY=OFF ^
 -DBUILD_PYTHON_DOCS=OFF ^
 -DWITH_AVX=OFF ^
 -DWITH_MONGODB=OFF ^
 -Wno-dev ^
 -DBoost_USE_STATIC_LIBS=OFF
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cmake --build . --config Release --target install
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

:: Build tttrlib
cd %ROOT_DIR%\modules\tttrlib
:: git fetch --all
:: git checkout development
:: git pull origin development
:: git submodule update --init --recursive

if exist b2 rmdir b2 /s /q
mkdir b2
cd b2
cmake .. -G "Ninja" ^
 -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
 -DCMAKE_PREFIX_PATH="%PREFIX%" ^
 -DBUILD_PYTHON_INTERFACE=ON ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="%SP_DIR%" ^
 -DCMAKE_SWIG_OUTDIR="%SP_DIR%" ^
 -DPython_ROOT_DIR="%PREFIX%\bin" ^
 -DBUILD_LIBRARY=OFF ^
 -DBUILD_PYTHON_DOCS=ON ^
 -DWITH_AVX=OFF ^
 -DBoost_USE_STATIC_LIBS=OFF
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
ninja install
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

:: Restore info.py
move /Y chisurf\info.py.bak chisurf\info.py
