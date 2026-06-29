import os
import shutil
import subprocess
import sys
import pathlib
from distutils.command.build import build as _build
from setuptools import setup, Command, find_packages

# ---------------------------------------------------------------------------
# Burbulator C++ shared-library build helpers
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).parent.resolve()
SIMULATION_DIR = HERE / "chisurf" / "plugins" / "core" / "acq" / "tcspc_devices" / "simulation"
CSRC_DIR = HERE / "src" / "csrc" / "burbulator"
CMAKE_BUILD_DIR = HERE / "build" / "burbulator_cmake"


def _get_lib_name():
    if sys.platform == "win32":
        return "burbulator.dll"
    elif sys.platform == "darwin":
        return "libburbulator.dylib"
    else:
        return "libburbulator.so"


def _build_burbulator():
    # If the shared library was already built and placed next to the wrapper
    # (e.g. the conda recipe builds it once before `pip install .`), skip the
    # rebuild. Avoids a redundant CMake invocation that fails inside the conda
    # Windows build env (bogus CMAKE_GENERATOR).
    if SIMULATION_DIR.exists() and any(SIMULATION_DIR.glob("*burbulator*")):
        print("Burbulator library already present; skipping rebuild")
        return
    if not shutil.which("cmake"):
        print("CMake not found -- skipping Burbulator C++ build", file=sys.stderr)
        return

    CSRC_DIR.mkdir(parents=True, exist_ok=True)
    CMAKE_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([
        "cmake", "-S", str(CSRC_DIR), "-B", str(CMAKE_BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ])
    subprocess.check_call([
        "cmake", "--build", str(CMAKE_BUILD_DIR), "--config", "Release",
    ])

    lib_name = _get_lib_name()
    src_lib = CMAKE_BUILD_DIR / "lib" / lib_name
    if not src_lib.exists():
        candidates = list(CMAKE_BUILD_DIR.rglob("*burbulator*"))
        src_lib = candidates[0] if candidates else None

    if src_lib is None or not src_lib.exists():
        print(f"WARNING: burbulator library not found under {CMAKE_BUILD_DIR}", file=sys.stderr)
        return

    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
    dst = SIMULATION_DIR / lib_name
    shutil.copy2(str(src_lib), str(dst))
    print(f"Burbulator library installed -> {dst}")


# ---------------------------------------------------------------------------
# Custom setuptools commands
# ---------------------------------------------------------------------------

class build_burbulator(Command):
    description = "Build the Burbulator C++ shared library"
    user_options: list = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _build_burbulator()


class build(_build):
    """Build + C++ lib."""

    def run(self):
        self.run_command("build_burbulator")
        super().run()


# Override `develop` so that ``pip install -e .`` also compiles the C++ lib.
try:
    from setuptools.command.develop import develop as _develop

    class develop(_develop):
        def run(self):
            self.run_command("build_burbulator")
            super().run()

except ImportError:
    develop = None  # fallback for very old setuptools


# ---------------------------------------------------------------------------
setup(
    name="chisurf",
    version=os.environ.get("CHISURF_VERSION", "26.dev0"),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        "build": build,
        "build_burbulator": build_burbulator,
        **({} if develop is None else {"develop": develop}),
    },
)
