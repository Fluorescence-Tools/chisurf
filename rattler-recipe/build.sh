#!/bin/bash
set -euo pipefail

# Enable long paths in git to avoid errors with deeply nested submodules (mostly for Windows but good practice)
git config --global core.longpaths true

# macOS SDK note
if [[ "${target_platform}" == osx-* ]]; then
  export CXXFLAGS="${CXXFLAGS:-} -D_LIBCPP_DISABLE_AVAILABILITY"
fi

PY="$PYTHON"

# 1) Qt resources
"$PREFIX/bin/pyrcc5" chisurf/gui/resources/resource.qrc -o chisurf/gui/resources/resource.py

# 2) Prepare & install labellib
pushd modules/labellib
  git fetch --tags --force || true
  git checkout -f 2020.10.05 || git checkout -f tags/2020.10.05 || true
  (cd thirdparty/pybind11 && git fetch --tags --force || true; git checkout -f v2.13)
  rm -rf thirdparty/eigen
  git clone --depth 1 --branch 3.4 https://gitlab.com/libeigen/eigen thirdparty/eigen
popd

# 4) Install local modules
"$PY" -m pip install ./modules/labellib    --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/clsmview    --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/ndxplorer   --no-deps -vv --prefix="$PREFIX"
"$PY" -m pip install ./modules/quest       --no-deps -vv --prefix="$PREFIX"

# 5) Install chinet (pure Python, no CMake/SWIG needed)
"$PY" -m pip install ./modules/chinet --no-deps -vv --prefix="$PREFIX"

# 6) Versioning
echo "Building ChiSurf version: $PKG_VERSION"
cp chisurf/info.py chisurf/info.py.bak
sed -i.tmp "s/__version__ = .*/__version__ = '$PKG_VERSION'/" chisurf/info.py

# 7) Install main module
"$PY" -m pip install . --no-deps -vv --prefix="$PREFIX"

# Restore original info.py
mv chisurf/info.py.bak chisurf/info.py
