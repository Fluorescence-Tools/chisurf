@echo off

:: Generate Python resources
call pyrcc5 chisurf\gui\resources\resource.qrc -o chisurf\gui\resources\resource.py

:: Update submodules
git submodule sync --recursive
git submodule update --init --recursive --force

:: Install Python modules

:: Labellib Module
cd modules\labellib
git fetch --tags
git checkout tags/2020.10.05
cd thirdparty\pybind11
git checkout v2.13
git pull
cd ..\..\..
pip install modules\labellib --no-deps --prefix=%PREFIX%

:: Install other Python modules (same as build.sh)
pip install modules/scikit-fluorescence --no-deps --prefix=%PREFIX%
pip install modules/clsmview --no-deps --prefix=%PREFIX%
pip install modules/k2dist --no-deps --prefix=%PREFIX%
pip install modules/ndxplorer --no-deps --prefix=%PREFIX%
pip install modules/tttrconvert --no-deps --prefix=%PREFIX%

:: Build chinet module (same as build.sh)
cd modules\chinet
if exist build rmdir /s /q build
mkdir build
cd build
cmake -S .. -B . ^
  -DCMAKE_CXX_COMPILER="%CXX%" ^
  -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
  -DBUILD_PYTHON_INTERFACE=ON ^
  -DWITH_AVX=OFF ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBoost_USE_STATIC_LIBS=OFF ^
  -DCMAKE_SWIG_OUTDIR="%PREFIX%" ^
  -DBUILD_PYTHON_DOCS=ON ^
  -DPython_ROOT_DIR="%PREFIX%\bin" ^
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="%PREFIX%" ^
  -G "Ninja"
ninja install -j %CPU_COUNT%
cd ..\..\..

:: Build fit2x module (same as build.sh)
cd modules\fit2x
git switch master
if exist build rmdir /s /q build
mkdir build
cd build
cmake ^
  -DCMAKE_INSTALL_PREFIX="%PREFIX%" ^
  -DCMAKE_PREFIX_PATH="%PREFIX%" ^
  -DBUILD_PYTHON_INTERFACE=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="%SP_DIR%" ^
  -DCMAKE_SWIG_OUTDIR="%SP_DIR%" ^
  -DPython_ROOT_DIR="%PREFIX%\bin" ^
  ..
cmake --build . --target install
cd ..\..\..

:: Install main module
%PYTHON% setup.py build_ext --force --inplace
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
