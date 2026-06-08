@echo off

REM Update git submodules recursively, initializing and fetching remote updates
git submodule update --init --recursive --remote

REM Remove and recreate the build directory
rmdir b2 /s /q
mkdir b2
cd b2

REM Configure the build using CMake
cmake .. -G "Visual Studio 17 2022" ^
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
 -Wno-dev ^
 -DBoost_USE_STATIC_LIBS=OFF

REM Set compilation to use multiple processes
set CL=/MP

REM Build and install the project
cmake --build . --config Release --target install
