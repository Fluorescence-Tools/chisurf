:: Generate Python resources
call pyrcc5 chisurf\gui\resources\resource.qrc -o chisurf\gui\resources\resource.py

:: Update submodules
git submodule sync --recursive
git submodule update --init --recursive --force

:: Install Python modules

:: Set to specific Labellib version
cd modules\labellib
git fetch --tags
git checkout tags/2020.10.05
cd thirdparty\pybind11
git checkout v2.13
git pull
cd ..\..\

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

echo %CD%
pip install .\scikit-fluorescence --no-deps --prefix=%PREFIX%
pip install .\clsmview --no-deps --prefix=%PREFIX%
pip install .\k2dist --no-deps --prefix=%PREFIX%
pip install .\ndxplorer --no-deps --prefix=%PREFIX%
pip install .\tttrconvert --no-deps --prefix=%PREFIX%
pip install .\quest --no-deps --prefix=%PREFIX%
cd ..

:: Build chinet module
cd modules\chinet

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
 -Wno-dev ^
 -DBoost_USE_STATIC_LIBS=OFF
:: Build and install the project
cmake --build . --config Release --target install
cd ..\..\..

:: Build tttrlib module
cd modules\tttrlib

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

cd ..\..\..

:: Build fit2x module
cd modules\fit2x
git switch master
if exist build rmdir /s /q build
md build
cd build

:: Call Python with the --version flag to get the version information
for /f "tokens=2 delims= " %%v in ('%PYTHON% --version 2^>^&1') do set PYTHON_VERSION=%%v
:: Extract only the numeric part of the version
for /f "tokens=1-3 delims=." %%a in ("%PYTHON_VERSION%") do set PYTHON_VERSION_NUMERIC=%%a.%%b.%%c

REM Configure the build using CMake
cmake .. -G "Visual Studio 17 2022" ^
 -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
 -DCMAKE_PREFIX_PATH="%PREFIX%" ^
 -DBUILD_PYTHON_INTERFACE=ON ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="%SP_DIR%" ^
 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="%SP_DIR%" ^
 -DCMAKE_SWIG_OUTDIR="%SP_DIR%" ^
 -DPython_ROOT_DIR="%PREFIX%\bin" ^
 -DBUILD_LIBRARY=OFF ^
 -DBUILD_PYTHON_DOCS=ON ^
 -DWITH_AVX=OFF ^
 -Wno-dev ^
 -DBoost_USE_STATIC_LIBS=OFF

:: Build and install the project
cmake --build . --config Release --target install
cd ..\..\..

:: Install main module
%PYTHON% setup.py build_ext --force --inplace
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt