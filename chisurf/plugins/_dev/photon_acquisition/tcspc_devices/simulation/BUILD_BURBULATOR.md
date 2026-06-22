# Building the Burbulator DLL from the SM Acquisition plugin

This directory contains a CMake configuration and the full set of C/C++
sources required to build the `burbulator` shared library (DLL on Windows,
`.so`/`.dylib` on other platforms) used by the simulation code.

All relevant sources (`*.cpp`, `*.h`, and `burbulator.def`) have been copied
into this `simulation` folder, so you can build the DLL entirely from here
without depending on the `playground` tree.

## Prerequisites

- CMake ≥ 3.12
- A C++ compiler toolchain
  - Windows: MSVC (Visual Studio) or compatible
  - Linux/macOS: GCC or Clang

## Build steps (Windows / MSVC)

From this directory (`chisurf/plugins/sm_acquisition/tcspc_devices/simulation`):

```powershell
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release
```

The resulting `burbulator.dll` and import library will be written directly into
this `simulation` folder (not into `build/`), so the Python wrappers can load
it without additional configuration.

## Build steps (Linux / macOS)

From the same directory:

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

This will produce a shared library (e.g. `libburbulator.so` or
`libburbulator.dylib`) in the `simulation` folder.

## Notes

- The CMake configuration (`CMakeLists.txt`) builds directly from the local
  `*.cpp` files in this folder and includes headers from the same location.

- The Python wrapper `burbulator_dll_wrapper.py` currently looks for
  `burbulator_x64.dll` / `burbulator.dll` in this folder on Windows. On
  non-Windows platforms you may need to extend the loader to also look for the
  appropriate `.so` / `.dylib` name.
