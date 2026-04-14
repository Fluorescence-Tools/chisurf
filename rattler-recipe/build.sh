#!/bin/bash
set -euo pipefail

# Enable long paths in git to avoid errors with deeply nested submodules (mostly for Windows but good practice)
git config --global core.longpaths true

# macOS SDK note
if [[ "${target_platform}" == osx-* ]]; then
  export CXXFLAGS="${CXXFLAGS:-} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

PY="$PYTHON"

find_cmake_package_dir() {
  local package_name="$1"
  shift
  local root
  for root in "$@"; do
    [[ -d "$root" ]] || continue
    local match
    match=$(find "$root" -type f -name "${package_name}Config.cmake" -print -quit 2>/dev/null || true)
    if [[ -n "$match" ]]; then
      dirname "$match"
      return 0
    fi
  done
  return 1
}

# 1) Qt resources
"$PREFIX/bin/pyrcc5" chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py

# 2) Prepare & install labellib without network fetches
pushd modules/labellib
  if [[ ! -d .git ]]; then
    git init -q
    git config user.email "build@example.invalid"
    git config user.name "ChiSurf Build"
    git add -A
    git commit -qm "build metadata"
    git tag 2020.10.05
  fi
popd

export CMAKE_PREFIX_PATH="${PREFIX}:${BUILD_PREFIX:-}:${CMAKE_PREFIX_PATH:-}"
export PIP_NO_BUILD_ISOLATION=1

EIGEN3_DIR="$(find_cmake_package_dir Eigen3 "$PREFIX" "${BUILD_PREFIX:-}")" || true
PYBIND11_DIR="$(find_cmake_package_dir pybind11 "$PREFIX" "${BUILD_PREFIX:-}")" || true

export CMAKE_ARGS="${CMAKE_ARGS:-}"
if [[ -n "${EIGEN3_DIR:-}" ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS} -DEigen3_DIR=${EIGEN3_DIR}"
fi
if [[ -n "${PYBIND11_DIR:-}" ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS} -Dpybind11_DIR=${PYBIND11_DIR}"
fi
if [[ "${target_platform}" == linux-* ]] && [[ -x /usr/bin/gcc ]] && [[ -x /usr/bin/g++ ]]; then
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
  export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX}"
fi

# 4) Install local modules
"$PY" -m pip install ./modules/labellib    --no-deps --no-build-isolation -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/clsmview    --no-deps --no-build-isolation -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/ndxplorer   --no-deps --no-build-isolation -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/quest       --no-deps --no-build-isolation -vv --prefix="$PREFIX"

# 5) Install chinet (pure Python, no CMake/SWIG needed)
"$PY" -m pip install ./modules/chinet --no-deps --no-build-isolation -vv --prefix="$PREFIX"

# 6) Versioning
echo "Building ChiSurf version: $PKG_VERSION"
cp chisurf/info.py chisurf/info.py.bak
sed -i.tmp "s/__version__ = .*/__version__ = '$PKG_VERSION'/" chisurf/info.py

# 7) Install main module
"$PY" -m pip install . --no-deps --no-build-isolation -vv --prefix="$PREFIX"

# Restore original info.py
mv chisurf/info.py.bak chisurf/info.py
