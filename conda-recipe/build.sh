if [[ "${target_platform}" == osx-* ]]; then
  # See https://conda-forge.org/docs/maintainer/knowledge_base.html#newer-c-features-with-old-sdk
  CXXFLAGS="${CXXFLAGS} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

pyrcc5 chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py

# Install modules
#################
# Update submodules
git submodule sync --recursive
git submodule update --init --recursive --force

# Install python modules
########################

# Labellib
cd modules/labellib
git fetch --tags && git checkout tags/2020.10.05
cd thirdparty/pybind11 && git checkout v2.13 && git pull && cd ../..
cd thirdparty && rm -rf eigen
git clone https://gitlab.com/libeigen/eigen
cd eigen && git switch 3.4
cd ../../../..
pip install modules/labellib --no-deps --prefix="$PREFIX"

pip install modules/clsmview --no-deps --prefix="$PREFIX"
pip install modules/ndxplorer --no-deps --prefix="$PREFIX"
pip install modules/tttrconvert --no-deps --prefix="$PREFIX"
pip install modules/quest --no-deps --prefix="$PREFIX"
pip install modules/lltf --no-deps --prefix="$PREFIX"

# Build chinet
cd modules/chinet
rm -rf build && mkdir build && cd build
cmake -S .. -B . \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_PYTHON_INTERFACE=ON \
  -DWITH_AVX=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBoost_USE_STATIC_LIBS=OFF \
  -DCMAKE_SWIG_OUTDIR="${PREFIX}" \
  -DBUILD_PYTHON_DOCS=ON \
  -DPython_ROOT_DIR="${PREFIX}/bin" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${PREFIX}" \
  -DWITH_MONGODB=OFF \
  -G Ninja
ninja install -j ${CPU_COUNT}
cd ../../..

# Build fit2x
cd modules/fit2x
git switch master
rm -rf build && mkdir build && cd build
cmake \
 -DCMAKE_INSTALL_PREFIX="$PREFIX" \
 -DCMAKE_PREFIX_PATH="$PREFIX" \
 -DBUILD_PYTHON_INTERFACE=ON \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$SP_DIR" \
 -DCMAKE_SWIG_OUTDIR="$SP_DIR" \
 -DPython_ROOT_DIR="${PREFIX}/bin" \
 ..
make && make install
cd ../../..

# Build tttrlib
cd modules/tttrlib
git fetch --all
git checkout development
git pull origin development
git submodule update --init --recursive

rm -rf build && mkdir build && cd build
cmake -S .. -B . \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_PYTHON_INTERFACE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIBRARY=OFF \
  -DWITH_AVX=OFF \
  -DBoost_USE_STATIC_LIBS=OFF \
  -DPython_ROOT_DIR="${PREFIX}/bin" \
  -DBUILD_PYTHON_DOCS=ON \
  -G Ninja \
  ${CONFIG_ARGS}
make install -j ${CPU_COUNT}
cd ../../..


# Install main module
#####################
# Compile cython code
$PYTHON setup.py build_ext --force --inplace
# Install python code
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
