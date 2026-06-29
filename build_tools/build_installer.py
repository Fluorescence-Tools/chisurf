#!/usr/bin/env python3
"""Unified cross-platform installer builder for ChiSurf.

Produces a native installer for the current platform:

    Linux   -> AppImage   (linuxdeploy)
    macOS   -> DMG        (.app bundle + hdiutil)
    Windows -> setup.exe  (Inno Setup)

One shared pipeline does the heavy lifting on every platform; only the final
"wrap" differs. This replaces the old per-platform scripts (build-osx-app.sh,
linuxdeploy/build.sh, build-setup.bat and its helpers).

The conda recipe (`pip install .`) produces a package containing only chisurf
+ the burbulator lib, so the installer assembles a full runtime env on top of
it: chisurf (conda) + runtime libs + tttrlib + labellib + latexify-py +
imp-tricks + the local ``modules/*`` (chinet/clsmview/ndxplorer/quest). It then
slims the env aggressively (strip unused Qt, debug symbols, build tools, test
suites) before wrapping it.

Usage:
    python build_tools/build_installer.py [--no-build] [--platform PLAT]
                                          [--output-dir DIR] [--audit]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE_DIR = REPO_ROOT / "rattler-recipe"
WIN_DIR = REPO_ROOT / "build_tools" / "win"
LINUX_DIR = REPO_ROOT / "build_tools" / "linuxdeploy"
MODULES_DIR = REPO_ROOT / "modules"
CONDA_BLD = REPO_ROOT / "conda-bld"
DIST = REPO_ROOT / "dist"
PYVER = "3.12"
APP_NAME = "ChiSurf"

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"

IMP_TRICKS_REPO = os.environ.get("IMP_TRICKS_REPO", "https://gitlab.peulen.xyz/tpeulen/imp-tricks.git")
IMP_TRICKS_REF = os.environ.get("IMP_TRICKS_REF", "main")

# Qt feature tokens ChiSurf never imports (verified: only Core/Gui/Widgets +
# Multimedia/OpenGL/Svg are used). Dropping these C++ libs + PyQt bindings is
# the single biggest size win.
QT_DROP_TOKENS = (
    "WebEngine", "WebKit", "WebChannel", "WebSockets", "WebView",
    "3DAnimation", "3DCore", "3DExtras", "3DInput", "3DLogic", "3DRender", "3DQuick",
    "Quick", "Qml", "Designer", "Charts", "DataVisualization", "Pdf",
    "Bluetooth", "Positioning", "Sensors", "NetworkAuth", "Location",
    "RemoteObjects", "Gamepad", "SerialPort", "SerialBus", "Nfc",
)

# Packages whose bundled "tests" dirs are safe to drop (NOT tables/pytables,
# which imports its tests at runtime).
TEST_PKGS = ("numpy", "scipy", "pandas", "numba", "skimage", "mdtraj", "matplotlib", "sklearn")

# Build tools pulled in only to compile modules/* during assembly; removed after.
BUILD_TOOLS_TO_REMOVE = ("cmake", "ninja", "swig", "cython", "pythran", "vs2022_win-64", "doxygen")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def run(cmd, **kw) -> None:
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, **kw)


def rmtree(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
    elif p.exists():
        shutil.rmtree(p, ignore_errors=True)


# NOTE: all traversal uses os.walk(followlinks=False). Path.rglob("**") follows
# symlinked directories on Python 3.12, and conda envs can contain symlink cycles
# -> infinite recursion -> C-stack overflow -> SIGSEGV. os.walk is iterative and
# does not follow symlinks here.
def _walk_files(root: Path):
    for dp, _dns, fns in os.walk(root, followlinks=False):
        for fn in fns:
            yield Path(dp) / fn


def _walk_dirs(root: Path):
    for dp, dns, _fns in os.walk(root, followlinks=False):
        for dn in dns:
            yield Path(dp) / dn


def du_mb(p: Path) -> float:
    total = 0
    for f in _walk_files(p):
        try:
            if not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total / (1024 * 1024)


def _render_template(text: str, params: dict) -> str:
    """Minimal {{ VAR }} substitution (no jinja2 dependency). Single pass, so
    substituted values containing braces (e.g. the Inno {{GUID}} AppId) are not
    re-scanned, and Inno's own {app}/{group} single-brace tokens are untouched."""
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: str(params[m.group(1)]), text)


def load_info() -> dict:
    """Exec chisurf/core/info.py in isolation (stdlib-only) to read static app
    metadata without importing the Qt-heavy chisurf package."""
    info_path = REPO_ROOT / "chisurf" / "core" / "info.py"
    ns: dict = {"__file__": str(info_path)}
    exec(compile(info_path.read_text(), str(info_path), "exec"), ns)
    return ns


def get_version() -> str:
    env = os.environ.get("CHISURF_VERSION")
    if env:
        return env.strip()
    try:
        out = subprocess.check_output(
            [sys.executable, str(RECIPE_DIR / "generate_version.py"), "--print"], text=True
        ).strip()
        if out:
            return out
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: generate_version failed ({exc}); using fallback", file=sys.stderr)
    return "26.dev0"


# --------------------------------------------------------------------------- #
# Conda package + runtime env (shared)
# --------------------------------------------------------------------------- #
def build_conda_package() -> None:
    channels = ["--channel", "conda-forge"]
    if not IS_WIN:
        channels += ["--channel", "bioconda"]
    run(["rattler-build", "build", "--recipe", RECIPE_DIR, "--output-dir", CONDA_BLD,
         *channels, "--test", "skip"])


def find_conda_pkg() -> Path:
    matches = sorted(CONDA_BLD.rglob("chisurf-*.conda"), key=lambda p: p.stat().st_mtime)
    if not matches:
        sys.exit(f"ERROR: no chisurf-*.conda under {CONDA_BLD}. Run without --no-build.")
    print(f"Using conda package: {matches[-1]}")
    return matches[-1]


def site_packages(prefix: Path) -> Path:
    return prefix / ("Lib" if IS_WIN else f"lib/python{PYVER}") / "site-packages"


def make_runtime(prefix: Path, *, conda_extras: list[str], pip_nodeps: list[str],
                 pip_withdeps: list[str]) -> Path:
    """Create a slimmed, self-contained runtime env at ``prefix`` from the
    locally-built chisurf conda package plus the extra packages that the recipe
    does not bundle. Returns the python executable path."""
    rmtree(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    channels = ["-c", CONDA_BLD.as_uri(), "-c", "conda-forge"]
    if not IS_WIN:
        channels += ["-c", "bioconda"]
    run(["micromamba", "create", "-y", "-p", prefix,
         f"python={PYVER}", "chisurf", *conda_extras, *channels, "--no-channel-priority"])

    py = prefix / ("python.exe" if IS_WIN else "bin/python")

    # Env so modules/* (ndxplorer compiles C++/Eigen) and pip find the prefix.
    env = dict(os.environ)
    env["PATH"] = str(prefix / ("Scripts" if IS_WIN else "bin")) + os.pathsep + env.get("PATH", "")
    env["CMAKE_PREFIX_PATH"] = str(prefix)
    eigen_inc = prefix / ("Library/include/eigen3" if IS_WIN else "include/eigen3")
    env["CMAKE_ARGS"] = f"-DEIGEN3_INCLUDE_DIR={eigen_inc}"

    for pkg in pip_nodeps:
        run([py, "-m", "pip", "install", pkg, "--no-deps"], env=env)
    for pkg in pip_withdeps:
        run([py, "-m", "pip", "install", pkg], env=env)

    _install_imp_tricks(py, env)

    for mod in sorted(MODULES_DIR.iterdir()):
        if mod.name in ("imp-tricks", "tttrconvert"):  # imp-tricks installed above; tttrconvert is retired
            continue
        if (mod / "setup.py").exists() or (mod / "pyproject.toml").exists():
            run([py, "-m", "pip", "install", mod, "--no-deps"], env=env)

    # Slimming is optional for correctness — isolate it in a child process so a
    # crash (e.g. a native tool segfaulting) can't fail the installer build.
    rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--strip-only", str(prefix)])
    if rc != 0:
        print(f"WARNING: slimming step exited {rc}; shipping un-slimmed env", file=sys.stderr, flush=True)
    return py


def _install_imp_tricks(py: Path, env: dict) -> None:
    local = MODULES_DIR / "imp-tricks"
    if (local / "pyproject.toml").exists():
        src = local
        run([py, "-m", "pip", "install", src, "--no-deps"], env=env)
        return
    tmp = Path(tempfile.mkdtemp(prefix="imp-tricks-")) / "imp-tricks"
    run(["git", "clone", "--depth", "1", "--branch", IMP_TRICKS_REF, IMP_TRICKS_REPO, tmp])
    run([py, "-m", "pip", "install", tmp, "--no-deps"], env=env)


# --------------------------------------------------------------------------- #
# Slimming (shared): measure -> trim proven offenders -> re-measure
# --------------------------------------------------------------------------- #
def _step(msg: str) -> None:
    print(f"[strip] {msg}", flush=True)


def strip_bloat(prefix: Path) -> None:
    sp = site_packages(prefix)
    _step("measuring size")
    before = du_mb(prefix)

    _step("removing build tools")
    try:
        subprocess.run(["micromamba", "remove", "-y", "-p", str(prefix), *BUILD_TOOLS_TO_REMOVE],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=300)
    except Exception as exc:
        print(f"[strip] build-tool removal skipped: {exc}", flush=True)

    _step("removing generic cruft")
    for rel in ("include", "share/doc", "share/man", "share/info", "conda-meta", "man"):
        rmtree(prefix / rel)
    for f in _walk_files(prefix):
        if f.suffix in (".a", ".la") or (IS_WIN and f.suffix == ".lib"):
            f.unlink(missing_ok=True)
    for d in [d for d in _walk_dirs(prefix) if d.name == "__pycache__"]:
        rmtree(d)
    rmtree(sp / "pip")
    rmtree(sp / "wheel")

    _step("stripping unused Qt")
    _strip_qt(prefix, sp)

    _step("removing test suites")
    for pkg in TEST_PKGS:
        pkg_dir = sp / pkg
        if pkg_dir.exists():
            for d in [d for d in _walk_dirs(pkg_dir) if d.name == "tests"]:
                rmtree(d)

    if not IS_WIN:
        _step("stripping debug symbols")
        _strip_symbols(prefix)

    _step("measuring size (after)")
    after = du_mb(prefix)
    print(f"[strip] env: {before:.0f} MB -> {after:.0f} MB (saved {before - after:.0f} MB)", flush=True)


def _strip_qt(prefix: Path, sp: Path) -> None:
    drop = tuple(t.lower() for t in QT_DROP_TOKENS)
    roots = [prefix / "lib", prefix / "Library" / "bin", prefix / "Library" / "lib",
             prefix / "Library" / "plugins", prefix / "plugins", sp / "PyQt5"]
    for root in roots:
        if not root.exists():
            continue
        for f in list(_walk_files(root)):
            low = f.name.lower()
            if any(tok in low for tok in drop):
                f.unlink(missing_ok=True)
    for root in (prefix / "lib", prefix / "Library", sp / "PyQt5"):
        if not root.exists():
            continue
        for d in [d for d in _walk_dirs(root) if d.name in ("qml", "translations")]:
            rmtree(d)
    for d in [d for d in _walk_dirs(prefix)
              if d.name.startswith("QtWebEngineProcess") or d.name.startswith("qtwebengine_")]:
        rmtree(d)


def _strip_symbols(prefix: Path) -> None:
    strip_bin = shutil.which("strip")
    if not strip_bin:
        return
    args = ["-x"] if IS_MAC else ["--strip-unneeded"]
    count = 0
    for f in _walk_files(prefix):
        if not f.is_symlink() and f.suffix in (".so", ".dylib"):
            subprocess.run([strip_bin, *args, str(f)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            count += 1
    print(f"[strip] symbols stripped from {count} shared libraries")


# --------------------------------------------------------------------------- #
# Platform packagers
# --------------------------------------------------------------------------- #
def package_macos(version: str, info: dict) -> Path:
    prefix = DIST / "osx"
    make_runtime(prefix,
                 conda_extras=["tttrlib", "cmake<3.27", "zstd", "libarchive", "libffi",
                               "openblas", "libgfortran5", "llvm-openmp"],
                 pip_nodeps=["labellib"], pip_withdeps=["latexify-py"])

    app = DIST / f"{APP_NAME}.app"
    rmtree(app)
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir(parents=True)
    shutil.copytree(prefix / "bin", contents / "bin", symlinks=True)
    shutil.copytree(prefix / "lib" / f"python{PYVER}", contents / "lib" / f"python{PYVER}", symlinks=True)
    for dylib in (prefix / "lib").glob("*.dylib"):
        shutil.copy2(dylib, contents / "lib" / dylib.name, follow_symlinks=False)
    for f in (contents / "lib").glob("libc++*.dylib"):  # OS-provided; bundling => SIGBUS
        f.unlink()

    icon = REPO_ROOT / "chisurf" / "gui" / "resources" / "icons" / "cs_logo.icns"
    if icon.exists():
        shutil.copy2(icon, contents / "Resources" / "ChiSurf.icns")
    (contents / "PkgInfo").write_text("APPL????")
    (contents / "Info.plist").write_text(_MAC_PLIST.format(app=APP_NAME, version=version))
    launcher = contents / "MacOS" / APP_NAME
    launcher.write_text(_MAC_LAUNCHER)
    launcher.chmod(0o755)

    dmg = DIST / f"{APP_NAME}-{version}.dmg"
    dmg.unlink(missing_ok=True)
    dmg_tmp = DIST / "dmg_tmp"
    rmtree(dmg_tmp)
    dmg_tmp.mkdir()
    run(["cp", "-a", app, dmg_tmp / f"{APP_NAME}.app"])
    (dmg_tmp / "Applications").symlink_to("/Applications")
    run(["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", dmg_tmp,
         "-ov", "-format", "ULFO", dmg])  # lzfse: smaller than UDZO zlib
    rmtree(dmg_tmp)
    return dmg


def package_linux(version: str, info: dict) -> Path:
    prefix = DIST / "linux" / "runtime"
    make_runtime(prefix,
                 conda_extras=["tttrlib", "cmake<3.27", "zstd", "libarchive", "openblas"],
                 pip_nodeps=["labellib"], pip_withdeps=["latexify-py"])

    appdir = Path(tempfile.mkdtemp(prefix="chisurf-appdir-"))
    try:
        (appdir / "usr").mkdir(parents=True)
        run(["cp", "-a", prefix, appdir / "usr" / "runtime"])
        (appdir / "AppRun").write_text(_LINUX_APPRUN)
        (appdir / "AppRun").chmod(0o755)
        desktop = LINUX_DIR / "chisurf.desktop"
        if desktop.exists():
            shutil.copy2(desktop, appdir / "chisurf.desktop")
        png = REPO_ROOT / "chisurf" / "gui" / "resources" / "icons" / "cs_logo.png"
        if png.exists():
            shutil.copy2(png, appdir / "chisurf-logo.png")

        linuxdeploy = DIST / "linuxdeploy-x86_64.AppImage"
        if not linuxdeploy.exists():
            run(["curl", "-L", "-o", linuxdeploy,
                 "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"])
            linuxdeploy.chmod(0o755)

        out = DIST / f"{APP_NAME}-{version}-x86_64.AppImage"
        out.unlink(missing_ok=True)
        env = dict(os.environ, OUTPUT=str(out), COMP="xz", APPIMAGE_EXTRACT_AND_RUN="1")
        run([linuxdeploy, "--appdir", appdir, "--output", "appimage"], env=env)
        return out
    finally:
        rmtree(appdir)


def package_windows(version: str, info: dict) -> Path:
    prefix = DIST / "win"
    make_runtime(prefix,
                 conda_extras=["cmake<3.27", "ninja", "eigen", "pybind11", "swig", "cython",
                               "pythran", "vs2022_win-64", "hdf5", "zstd", "libarchive", "openblas"],
                 pip_nodeps=["labellib", "tttrlib"],  # no win-64 bioconda tttrlib
                 pip_withdeps=["latexify-py"])

    icon = REPO_ROOT / "chisurf" / str(info["setup_icon"]).lstrip("/")
    params = {
        "AppId": info["__app_id__"],
        "AppName": info["__name__"],
        "AppVerName": f"{info['__name__']} {version}" + (" (Dev)" if info.get("__status__") == "Dev" else ""),
        "AppVersion": version,
        "AppPublisher": info["__author__"],
        "AppURL": info["__url__"],
        "LicenseFile": str(REPO_ROOT / "LICENSE"),
        "Output_dir": str(DIST),
        "App_dir": str(prefix),
        "SetupIconFile": str(icon),
    }
    iss = _render_template((WIN_DIR / "setup_template.jinja2").read_text(), params)
    iss_path = WIN_DIR / "installer_config.iss"
    iss_path.write_text(iss)
    try:
        run([_find_iscc(), str(iss_path)], cwd=WIN_DIR)
    finally:
        iss_path.unlink(missing_ok=True)

    out = DIST / f"ChiSurf-Windows-Setup-{version}.exe"
    if not out.exists():
        sys.exit(f"ERROR: expected installer not found: {out}")
    return out


def _find_iscc() -> str:
    found = shutil.which("ISCC") or shutil.which("iscc")
    if found:
        return found
    for c in (r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
              r"C:\Program Files\Inno Setup 6\ISCC.exe"):
        if Path(c).exists():
            return c
    sys.exit("ERROR: Inno Setup (ISCC.exe) not found. Install via 'choco install innosetup'.")


# --------------------------------------------------------------------------- #
# Static templates
# --------------------------------------------------------------------------- #
_MAC_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIconFile</key><string>ChiSurf</string>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleDisplayName</key><string>{app}</string>
    <key>CFBundleExecutable</key><string>{app}</string>
    <key>CFBundleIdentifier</key><string>xyz.peulen.chisurf</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>{app}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict>
</plist>
"""

_MAC_LAUNCHER = """#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYVER=python3.12
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SCRIPT_DIR/Contents/lib/$PYVER/site-packages"
export PATH="$SCRIPT_DIR/Contents/bin:$PATH"
export QT_PLUGIN_PATH="$SCRIPT_DIR/Contents/lib/$PYVER/site-packages/PyQt5/Qt5/plugins"
export QT_MAC_DISABLE_APP_NAP=1
export QT_MAC_WANTS_BEST_RESOLUTION_OPENGL_SURFACE=0
cd "$HOME"
exec "$SCRIPT_DIR/Contents/bin/python" -m chisurf "$@"
"""

_LINUX_APPRUN = """#!/bin/bash
SELF="$(readlink -f "${0}")"
APPDIR="$(dirname "${SELF}")"
export PATH="$APPDIR/usr/runtime/bin:$PATH"
export QT_PLUGIN_PATH="$APPDIR/usr/runtime/plugins"
export LD_LIBRARY_PATH="$APPDIR/usr/runtime/lib:$LD_LIBRARY_PATH"
exec "$APPDIR/usr/runtime/bin/python3" -m chisurf "$@"
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    # Hidden mode: run only the (isolated) slimming pass on an existing prefix.
    if len(sys.argv) == 3 and sys.argv[1] == "--strip-only":
        strip_bloat(Path(sys.argv[2]))
        return

    ap = argparse.ArgumentParser(description="Build the ChiSurf installer for this platform.")
    ap.add_argument("--no-build", action="store_true", help="reuse an existing conda package")
    ap.add_argument("--platform", choices=["macos", "linux", "windows"], help="override target")
    ap.add_argument("--output-dir", help="conda-bld dir (default ./conda-bld)")
    ap.add_argument("--audit", action="store_true", help="print the largest dirs in the runtime env")
    args = ap.parse_args()

    global CONDA_BLD
    if args.output_dir:
        CONDA_BLD = Path(args.output_dir).resolve()

    target = args.platform or ("macos" if IS_MAC else "windows" if IS_WIN else "linux")
    info = load_info()
    version = get_version()
    print(f"=== Building {APP_NAME} {version} installer for {target} ===")

    if not args.no_build:
        build_conda_package()
    find_conda_pkg()

    artifact = {"macos": package_macos, "linux": package_linux, "windows": package_windows}[target](version, info)

    if args.audit:
        _audit(target)
    print(f"\n=== Created {artifact}  ({artifact.stat().st_size / 1024 / 1024:.0f} MB) ===")


def _audit(target: str) -> None:
    prefix = {"macos": DIST / "osx", "linux": DIST / "linux" / "runtime", "windows": DIST / "win"}[target]
    if not prefix.exists():
        return
    sizes = [(du_mb(d), d) for d in _walk_dirs(prefix)]
    sizes.sort(reverse=True)
    print("\n[audit] largest directories:")
    for mb, d in sizes[:30]:
        print(f"  {mb:8.1f} MB  {d.relative_to(prefix)}")


if __name__ == "__main__":
    main()
