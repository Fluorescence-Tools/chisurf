git submodule update --recursive --init --remote

if [[ "${target_platform}" == osx-* ]]; then
  # See https://conda-forge.org/docs/maintainer/knowledge_base.html#newer-c-features-with-old-sdk
  CXXFLAGS="${CXXFLAGS} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

rm -rf build && mkdir build && cd build
cmake -S .. -B . \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_PYTHON_INTERFACE=ON \
  -DWITH_AVX=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBoost_USE_STATIC_LIBS=OFF \
  -DCMAKE_SWIG_OUTDIR="${SP_DIR}" \
  -DBUILD_PYTHON_DOCS=ON \
  -DPython_ROOT_DIR="${PREFIX}/bin" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${PREFIX}" \
  -G Ninja

ninja install -j ${CPU_COUNT}
