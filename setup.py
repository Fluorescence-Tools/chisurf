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
CSRC_DIR = SIMULATION_DIR / "csrc"
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
# Build-time static version
# ---------------------------------------------------------------------------
# Resolve the version once at build time and freeze it into
# ``chisurf/core/_version.py`` so the *installed* app reads a static string
# instead of spawning git on every ``import chisurf`` (see chisurf/core/info.py).
# Editable/develop installs deliberately skip this, keeping the dev version
# git-derived and live.

def _resolve_version() -> str:
    env = os.environ.get("CHISURF_VERSION")
    if env:
        return env.strip()
    # Remove any stale generated file so info.py falls through to the git path
    # rather than reading a previous build's frozen version.
    version_file = HERE / "chisurf" / "core" / "_version.py"
    try:
        version_file.unlink()
    except FileNotFoundError:
        pass
    info_path = HERE / "chisurf" / "core" / "info.py"
    ns: dict = {"__file__": str(info_path)}
    exec(compile(info_path.read_text(), str(info_path), "exec"), ns)
    return ns.get("__version__", "26.dev0")


def _write_version_file(version: str) -> None:
    target = HERE / "chisurf" / "core" / "_version.py"
    target.write_text(
        "# Generated at build time -- do not edit or commit.\n"
        f'__version__ = "{version}"\n'
    )
    print(f"Wrote static version {version!r} -> {target}")


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
        _write_version_file(_resolve_version())
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
