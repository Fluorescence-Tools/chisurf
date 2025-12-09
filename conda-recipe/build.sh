set -euo pipefail

# macOS SDK note
if [[ "${target_platform}" == osx-* ]]; then
  export CXXFLAGS="${CXXFLAGS:-} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

PY="$PYTHON"

# 1) Qt resources
"$PREFIX/bin/pyrcc5" chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py

# 2) Prepare & install labellib exactly as you require
pushd modules/labellib
  git fetch --tags --force || true
  git checkout -f 2020.10.05 || git checkout -f tags/2020.10.05 || true
  (cd thirdparty/pybind11 && git fetch --tags --force || true; git checkout -f v2.13)
  rm -rf thirdparty/eigen
  git clone --depth 1 --branch 3.4 https://gitlab.com/libeigen/eigen thirdparty/eigen
popd

# 4) Install your local modules
"$PY" -m pip install ./modules/labellib    --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/clsmview    --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/ndxplorer   --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/quest       --no-deps -vv --prefix="$PREFIX"

# 5) Build & install chinet (CMake+SWIG)
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

# 6) Use the version from conda's PKG_VERSION environment variable
# This is automatically set by conda-build from meta.yaml
echo "Building ChiSurf version: $PKG_VERSION"

# Replace dynamic version in chisurf/info.py with the build version
cp chisurf/info.py chisurf/info.py.bak
sed -i.tmp "s/__version__ = .*/__version__ = '$PKG_VERSION'/" chisurf/info.py

# 7) Install top-level chisurf
"$PY" -m pip install . --no-deps -vv --prefix="$PREFIX"

# Restore original info.py
mv chisurf/info.py.bak chisurf/info.py
