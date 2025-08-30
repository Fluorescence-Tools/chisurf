set -euo pipefail

# macOS SDK note (avoid availability errors)
if [[ "${target_platform}" == osx-* ]]; then
  export CXXFLAGS="${CXXFLAGS:-} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

# Ensure we use the right Python/pip
PY="$PYTHON"

# 1) Compile Qt resources (needs pyqt in host env)
#    On Linux/macOS, pyrcc5 is in $PREFIX/bin
"$PREFIX/bin/pyrcc5" chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py

# 2) Prepare labellib third-party deps exactly as requested
prep_labellib() {
  pushd modules/labellib

    # Pin labellib repo to the tag
    git fetch --tags --force || true
    # Accept both '2020.10.05' and 'tags/2020.10.05'
    if git rev-parse -q --verify "refs/tags/2020.10.05" >/dev/null; then
      git checkout -f "2020.10.05"
    else
      git checkout -f "tags/2020.10.05" || true
    fi

    # pybind11 to v2.13
    if [[ -d thirdparty/pybind11/.git ]]; then
      pushd thirdparty/pybind11
        git fetch --tags --force || true
        git checkout -f "v2.13"
      popd
    fi

    # Eigen to 3.4 (fresh, shallow clone)
    rm -rf thirdparty/eigen
    git clone --depth 1 --branch "3.4" https://gitlab.com/libeigen/eigen thirdparty/eigen

  popd
}

prep_labellib

# 3) Install local submodules via pip into $PREFIX (no Conda resolution)
"$PY" -m pip install ./modules/labellib     --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/clsmview     --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/ndxplorer    --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/tttrconvert  --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/quest        --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/lltf         --no-deps -vv --prefix="$PREFIX"

# 4) Build & install chinet (CMake+SWIG)
pushd modules/chinet
  rm -rf build && mkdir build && cd build
  cmake -S .. -B . \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=ON \
    -DWITH_AVX=OFF \
    -DBoost_USE_STATIC_LIBS=OFF \
    -DCMAKE_SWIG_OUTDIR="${PREFIX}" \
    -DBUILD_PYTHON_DOCS=ON \
    -DPython_ROOT_DIR="${PREFIX}/bin" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${PREFIX}" \
    -DWITH_MONGODB=OFF \
    -G Ninja
  ninja -j "${CPU_COUNT}"
  ninja install
popd

# 5) Finally install the top-level chisurf package itself
"$PY" -m pip install . --no-deps -vv --prefix="$PREFIX"
