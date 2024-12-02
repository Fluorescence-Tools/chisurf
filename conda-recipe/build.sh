if [[ "${target_platform}" == osx-* ]]; then
  # See https://conda-forge.org/docs/maintainer/knowledge_base.html#newer-c-features-with-old-sdk
  CXXFLAGS="${CXXFLAGS} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

pyrcc5 chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py

# Install modules
#################
# Update submodules
git submodule update --recursive --init --remote

# Install python modules
pip install modules/labellib --no-deps --prefix="$PREFIX"
pip install modules/clsmview --no-deps --prefix="$PREFIX"
pip install modules/k2dist --no-deps --prefix="$PREFIX"
pip install modules/ndxplorer --no-deps --prefix="$PREFIX"
pip install modules/tttrconvert --no-deps --prefix="$PREFIX"

## Build chinet
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
  -DCMAKE_INSTALL_RPATH="${PREFIX}/lib" \
  -G Ninja
ninja install -j ${CPU_COUNT}
cd ../../..

# Install chisurf
###################

## Compile cython code
$PYTHON setup.py build_ext --force --inplace
# Install python code
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
