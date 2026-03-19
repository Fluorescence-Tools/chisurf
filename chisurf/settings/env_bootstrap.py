from __future__ import annotations

import os
import sys
import glob
import ctypes
import pathlib
import logging
import site
from typing import Iterable, Optional

log = logging.getLogger(__name__)

PLAT = sys.platform
IS_WIN = PLAT.startswith("win")
IS_MAC = PLAT == "darwin"
IS_LNX = PLAT.startswith("linux")


def _norm(p: pathlib.Path) -> pathlib.Path:
    try:
        return p.resolve()
    except Exception:
        return p


def _first_existing(paths: Iterable[pathlib.Path]) -> Optional[pathlib.Path]:
    for p in paths:
        try:
            if p and p.is_dir():
                return _norm(p)
        except Exception:
            continue
    return None


def _get_approot() -> pathlib.Path:
    # 1) Explicit override via env var
    env = os.environ.get("CHISURF_APPROOT") or os.environ.get("NAPARI_APPROOT")
    if env:
        p = pathlib.Path(env)
        if p.is_dir():
            return _norm(p)

    # 2) Environment prefix (e.g. CONDA_PREFIX)
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("MAMBA_ROOT_PREFIX")
    if conda and pathlib.Path(conda).is_dir():
        return _norm(pathlib.Path(conda))

    # 3) Frozen app (PyInstaller/Briefcase/etc.)
    try:
        if getattr(sys, "frozen", False):
            return _norm(pathlib.Path(sys.executable).parent)
    except Exception:
        pass

    # 4) Venv/sys.prefix
    cands = [
        pathlib.Path(sys.prefix),
        pathlib.Path(getattr(sys, "base_prefix", sys.prefix)),
        pathlib.Path(sys.executable).parent,
    ]
    found = _first_existing(cands)
    if found:
        return found

    # 5) Fallback: project root relative to this file
    return _norm(pathlib.Path(__file__).resolve().parents[3])


APPROOT = _get_approot()


def _ensure_prepend_env_path(var: str, path: pathlib.Path) -> None:
    if not path or not path.is_dir():
        return
    cur = os.environ.get(var, "")
    parts = [p for p in cur.split(os.pathsep) if p]
    pstr = str(path)
    if pstr not in parts:
        os.environ[var] = pstr + (os.pathsep + cur if cur else "")


def _add_path(path: pathlib.Path) -> None:
    if not path or not path.is_dir():
        return
    _ensure_prepend_env_path("PATH", path)
    if IS_WIN and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(path))
        except Exception:
            pass


def _add_libpath(path: pathlib.Path) -> None:
    if not path or not path.is_dir():
        return
    if IS_LNX:
        _ensure_prepend_env_path("LD_LIBRARY_PATH", path)
    elif IS_MAC:
        # DYLD variables: best-effort; SIP can restrict these for system binaries
        _ensure_prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", path)
        _ensure_prepend_env_path("DYLD_LIBRARY_PATH", path)
    _add_path(path)


def _glob_one(patterns: Iterable[str]) -> Optional[str]:
    for pat in patterns:
        try:
            hits = glob.glob(pat, recursive=True)
            if hits:
                return hits[0]
        except Exception:
            continue
    return None


def _init_paths() -> None:
    # Common candidate folders (environment layouts)
    cand_bin = [
        APPROOT / "bin",
        APPROOT / "Scripts",               # Windows
        APPROOT / "Library" / "bin",      # Windows
        APPROOT / "Library" / "usr" / "bin",
    ]
    cand_lib = [
        APPROOT / "lib",
        APPROOT / "Library" / "lib",     # Windows
        APPROOT / "DLLs",                  # Windows
    ]

    for p in cand_bin:
        _add_path(p)
    for p in cand_lib:
        _add_libpath(p)


def _init_qt_plugins() -> None:
    # Known Qt plugin layouts
    candidates: list[pathlib.Path] = [
        APPROOT / "Qt6" / "plugins",
        APPROOT / "Library" / "Qt6" / "plugins",
        APPROOT / "plugins",
        APPROOT / "Library" / "plugins",  # conda/Win
    ]

    # Pip site-packages locations
    try:
        sp = []
        try:
            sp.extend(site.getsitepackages() or [])
        except Exception:
            pass
        try:
            usp = site.getusersitepackages()
            if usp:
                sp.append(usp)
        except Exception:
            pass
        for s in sp:
            base = pathlib.Path(s)
            candidates.extend([
                base / "PyQt6" / "Qt6" / "plugins",
                base / "PyQt5" / "Qt5" / "plugins",
            ])
    except Exception:
        pass

    qt_plugins = _first_existing(candidates)
    if qt_plugins:
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
        plat_dir = qt_plugins / "platforms"
        if plat_dir.is_dir():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plat_dir))

    if IS_MAC:
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")


def _init_vispy() -> None:
    # Prefer a Qt backend for vispy; user can override
    os.environ.setdefault("VISPY_APP", "PyQt6")


def _preload_freetype() -> None:
    # Preload FreeType to avoid lazy loader issues in some environments
    if os.environ.get("CHISURF_SKIP_PRELOAD_FREETYPE", "").lower() in {"1", "true", "yes"}:
        return
    if IS_WIN:
        cand_dirs = [
            APPROOT / "Library" / "bin",
            APPROOT / "bin",
        ]
        patterns = [str(d / "freetype*.dll") for d in cand_dirs]
    elif IS_LNX:
        cand_dirs = [
            APPROOT / "lib",
            APPROOT / "Library" / "lib",
        ]
        patterns = []
        for d in cand_dirs:
            patterns.extend([
                str(d / "libfreetype.so"),
                str(d / "libfreetype.so.*"),
            ])
    else:  # macOS
        cand_dirs = [
            APPROOT / "lib",
            APPROOT / "Library" / "lib",
        ]
        patterns = []
        for d in cand_dirs:
            patterns.extend([
                str(d / "libfreetype.dylib"),
                str(d / "libfreetype*.dylib"),
            ])

    ft = _glob_one(patterns)
    if not ft:
        return
    try:
        mode = ctypes.RTLD_GLOBAL if hasattr(ctypes, "RTLD_GLOBAL") else None
        ctypes.CDLL(ft, mode=mode)  # type: ignore[arg-type]
        log.debug("Preloaded FreeType: %s", ft)
    except Exception as e:
        log.debug("Could not preload FreeType (%s): %s", ft, e)


def _apply_thread_env_from_settings() -> None:
    """Apply thread-related environment variables from user settings.

    Reads the user's settings YAML (settings_chisurf.yaml) and respects the
    `threads` section with the following keys:
      - numba_num_threads (maps to NUMBA_NUM_THREADS)
      - numba_threading_layer (maps to NUMBA_THREADING_LAYER)
      - mkl_num_threads (maps to MKL_NUM_THREADS)
      - omp_num_threads (maps to OMP_NUM_THREADS)
      - mkl_threading_layer (maps to MKL_THREADING_LAYER)
      - override_existing_env (bool): if True, overwrite existing variables

    Best-effort: failures are swallowed and only logged at debug level.
    """
    try:
        # Local imports to avoid any potential cycle during early bootstrap
        from .path_utils import get_path
        from .settings_utils import get_chisurf_settings
        import sys as _sys
        import os as _os

        settings_dir = get_path('settings')
        # YAML source (existing behavior)
        settings_file = settings_dir / 'settings_chisurf.yaml'
        cs_settings = get_chisurf_settings(settings_file, use_source_folder=False)
        yaml_threads = cs_settings.get('threads', {}) if isinstance(cs_settings, dict) else {}

        # Optional JSON source: ~/.chisurf/settings.json with a top-level "threads" object
        json_threads = {}
        try:
            import json as _json
            json_file = settings_dir / 'settings.json'
            if json_file.is_file():
                with open(json_file, 'r', encoding='utf-8') as fh:
                    data = _json.load(fh)
                    if isinstance(data, dict):
                        jt = data.get('threads', {})
                        if isinstance(jt, dict):
                            json_threads = jt
        except Exception:
            # ignore malformed or missing JSON
            pass

        # Merge strategy: YAML base, JSON overrides
        threads = {}
        if isinstance(yaml_threads, dict):
            threads.update(yaml_threads)
        if isinstance(json_threads, dict):
            threads.update(json_threads)

        defaults = {
            'numba_num_threads': "1",
            'numba_threading_layer': "workqueue",
            'mkl_num_threads': "1",
            'omp_num_threads': "1",
            'mkl_threading_layer': "SEQUENTIAL",
        }
        override = bool(threads.get('override_existing_env', False))

        def _set_env(var_name: str, value: str):
            if override or var_name not in _os.environ or _os.environ.get(var_name, "") == "":
                _os.environ[var_name] = str(value)

        heavy_loaded = any(m in _sys.modules for m in ("numpy", "numba", "umap"))
        if heavy_loaded:
            # Changing env vars may not take effect if modules already imported
            log.debug("Thread env applied after heavy modules import; might not take full effect.")

        _set_env("NUMBA_NUM_THREADS", threads.get('numba_num_threads', defaults['numba_num_threads']))
        _set_env("NUMBA_THREADING_LAYER", threads.get('numba_threading_layer', defaults['numba_threading_layer']))
        _set_env("MKL_NUM_THREADS", threads.get('mkl_num_threads', defaults['mkl_num_threads']))
        _set_env("OMP_NUM_THREADS", threads.get('omp_num_threads', defaults['omp_num_threads']))
        _set_env("MKL_THREADING_LAYER", threads.get('mkl_threading_layer', defaults['mkl_threading_layer']))
    except Exception as e:
        # Fail silently; settings application is best-effort and should not break startup
        log.debug("Could not apply thread env from settings: %s", e)


def _init_local_python_modules() -> None:
    """Best-effort: add in-tree third-party modules to sys.path for dev.

    Specifically ensure the SARibbon-pyqt5 submodule (if present) is importable
    without a separate installation by appending its ``src`` folder to
    ``sys.path``. This enables ``import PySARibbon`` during development.
    """
    try:
        # Project root is two levels up from this file: chisurf/settings/...
        project_root = pathlib.Path(__file__).resolve().parents[2]
        saribbon_src = project_root / "modules" / "SARibbon-pyqt5" / "src"
        if saribbon_src.is_dir():
            p = str(saribbon_src)
            if p not in sys.path:
                sys.path.append(p)
                try:
                    log.debug("Added SARibbon-pyqt5 src to sys.path: %s", p)
                except Exception:
                    pass
    except Exception:
        # Never fail startup because of path tweaks
        pass


def _apply_custom_env_from_settings() -> None:
    """Apply environment variables defined in settings_chisurf.yaml.

    YAML shape:
      env:
        KEY: value
      env_override_existing: false
    """
    try:
        from .path_utils import get_path
        from .settings_utils import get_chisurf_settings
        import json as _json

        settings_dir = get_path('settings')
        settings_file = settings_dir / 'settings_chisurf.yaml'
        cs_settings = get_chisurf_settings(settings_file, use_source_folder=False)
        yaml_env_cfg = cs_settings.get('env', {}) if isinstance(cs_settings, dict) else {}

        json_env_cfg = {}
        try:
            json_file = settings_dir / 'settings.json'
            if json_file.is_file():
                with open(json_file, 'r', encoding='utf-8') as fh:
                    data = _json.load(fh)
                    if isinstance(data, dict):
                        je = data.get('env', {})
                        if isinstance(je, dict):
                            json_env_cfg = je
        except Exception:
            pass

        env_cfg = {}
        if isinstance(yaml_env_cfg, dict):
            env_cfg.update(yaml_env_cfg)
        if isinstance(json_env_cfg, dict):
            env_cfg.update(json_env_cfg)

        if not isinstance(env_cfg, dict) or not env_cfg:
            return

        override = bool(cs_settings.get('env_override_existing', True)) if isinstance(cs_settings, dict) else True

        for key, val in env_cfg.items():
            if key is None or val is None:
                continue
            if override or key not in os.environ or os.environ.get(key, "") == "":
                os.environ[str(key)] = str(val)
    except Exception as e:
        log.debug("Could not apply custom env from settings: %s", e)


# Execute at import time (idempotent and best-effort)
try:
    _apply_thread_env_from_settings()
    _apply_custom_env_from_settings()
    _init_paths()
    _init_local_python_modules()
    _init_qt_plugins()
    _init_vispy()
    _preload_freetype()
except Exception as _e:
    # Never fail application startup because of best-effort environment tweaks
    log.debug("env_bootstrap encountered a non-fatal error: %s", _e)
