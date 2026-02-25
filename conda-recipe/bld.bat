@echo off
setlocal

:: Generate Python resources
call pyrcc5 chisurf\gui\resources\resource.qrc -o chisurf\gui\resources\resource.py

:: Update submodules
git submodule sync --recursive
git submodule update --init --recursive --force

:: Build labellib
:: Set to specific Labellib version
cd modules\labellib
git fetch --tags
git checkout tags/2020.10.05

cd thirdparty\pybind11
git checkout v2.13
git pull
cd ..\..

:: Configure the build using CMake
cmake -S . -B build -A x64 ^
    -DPYTHON_EXECUTABLE="%PYTHON%" ^
    -DPYTHON_LIBRARY_OUTPUT_DIRECTORY="%SP_DIR%" ^
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="%SP_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release
:: Build the project
cmake --build build --config Release --parallel
:: Install the built library
cmake --install build --prefix %PREFIX%
cd ..

:: Build chinet
cd chinet

git fetch --all
git checkout development
git pull origin development
git submodule update --init --recursive

if exist build rmdir /s /q build
mkdir build
cd build
:: Configure the build using CMake
cmake .. -G "Visual Studio 17 2022" -A x64 ^
 -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
 -DCMAKE_PREFIX_PATH="%PREFIX%" ^
 -DBUILD_PYTHON_INTERFACE=ON ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="%SP_DIR%" ^
 -DCMAKE_SWIG_OUTDIR="%SP_DIR%" ^
 -DPython_ROOT_DIR="%PREFIX%\bin" ^
 -DBUILD_LIBRARY=OFF ^
 -DBUILD_PYTHON_DOCS=ON ^
 -DWITH_AVX=OFF ^
 -DWITH_MONGODB=OFF ^
 -Wno-dev ^
 -DBoost_USE_STATIC_LIBS=OFF
cmake --build . --config Release --target install
cd ..\..

:: Build tttrlib
cd tttrlib

git fetch --all
git checkout development
git pull origin development
git submodule update --init --recursive

rmdir b2 /s /q
mkdir b2
cd b2

cmake .. -G "NMake Makefiles" ^
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

nmake install
cd ..\..

:: Install Python modules
pip install .\clsmview --no-deps --prefix=%PREFIX%
pip install .\ndxplorer --no-deps --prefix=%PREFIX%
pip install .\quest --no-deps --prefix=%PREFIX%
cd ..

:: Clean pre-compiled Cython C++ files to force regeneration with current NumPy
:: This ensures compatibility with the NumPy version in the build environment
if exist chisurf\fluorescence\simulation\simulation_.cpp del chisurf\fluorescence\simulation\simulation_.cpp
if exist chisurf\structure\av\fps_.cpp del chisurf\structure\av\fps_.cpp
if exist chisurf\structure\potential\cPotentials_.cpp del chisurf\structure\potential\cPotentials_.cpp
if exist chisurf\math\reaction\reaction_.cpp del chisurf\math\reaction\reaction_.cpp

:: Use the version from conda's PKG_VERSION environment variable
:: This is automatically set by conda-build from meta.yaml
echo Building ChiSurf version: %PKG_VERSION%

:: Replace dynamic version in chisurf/info.py with the build version
:: Save original file
copy chisurf\info.py chisurf\info.py.bak

:: replace __version__ line with the version from conda
powershell -Command "(Get-Content chisurf\info.py) -replace \"__version__ = .*\", \"__version__ = '%PKG_VERSION%'\" | Set-Content chisurf\info.py"

:: Install main module using pip with pyproject.toml
:: Use --no-deps to avoid dependency issues, setuptools will create entry points
%PYTHON% -m pip install . --no-deps --no-build-isolation -vv --prefix=%PREFIX%

:: Restore original info.py
move /Y chisurf\info.py.bak chisurf\info.py