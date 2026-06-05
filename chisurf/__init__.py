from __future__ import annotations

import importlib
import logging
import os

# Map chisurf.logging to the standard logging module to support
# "import chisurf.logging" throughout the codebase.
import sys
sys.modules['chisurf.logging'] = logging
import pathlib
import sys
import typing

import chisurf.core.info

# --- DISTUTILS SHIM FOR PYTHON 3.12 ---
import sys
import importlib.util as _importlib_util
if 'distutils' not in sys.modules and _importlib_util.find_spec('distutils') is None:
    try:
        from types import ModuleType
        import packaging.version as _pkg_version
        import shutil as _shutil

        # Create dummy distutils and distutils.version
        du = ModuleType('distutils')
        sys.modules['distutils'] = du
        duv = ModuleType('distutils.version')
        sys.modules['distutils.version'] = duv
        dus = ModuleType('distutils.spawn')
        sys.modules['distutils.spawn'] = dus

        # Provide StrictVersion and LooseVersion backed by packaging.version.Version
        class _StrictVersion(_pkg_version.Version):
            def __init__(self, vstring):  # keep signature similar to distutils
                super().__init__(vstring)

        duv.StrictVersion = _StrictVersion
        duv.LooseVersion = _pkg_version.Version
        duv.__all__ = ['StrictVersion', 'LooseVersion']

        # Provide spawn helpers expected by some packages (e.g., mdtraj)
        def _find_executable(file, path=None):
            return _shutil.which(file, path=path)

        dus.find_executable = _find_executable
        dus.__all__ = ['find_executable']

        # Some packages check if distutils is really there (e.g., by looking for __version__)
        du.__version__ = "3.12-shim"
    except Exception:
        # Best-effort ultra-minimal fallback without packaging present
        try:
            from types import ModuleType
            du = ModuleType('distutils')
            sys.modules['distutils'] = du
            duv = ModuleType('distutils.version')
            sys.modules['distutils.version'] = duv
            dus = ModuleType('distutils.spawn')
            sys.modules['distutils.spawn'] = dus

            class _SV(str):
                pass

            duv.StrictVersion = _SV
            duv.LooseVersion = _SV
            duv.__all__ = ['StrictVersion', 'LooseVersion']
            dus.find_executable = lambda file, path=None: None
            dus.__all__ = ['find_executable']
            du.__version__ = "3.12-shim-nopackaging"
        except Exception:
            # If even this fails, we leave things untouched; import will error as before.
            pass
# ----------------------------------------

__version__ = chisurf.core.info.__version__

fits: typing.List["chisurf.core.fitting.fit.FitGroup"] = list()
imported_datasets: typing.List["chisurf.core.data.DataGroup"] = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = None         # The current instance of ChiSurf
console = None
experiment: typing.Dict[str, "chisurf.core.experiments.core.experiment.Experiment"] = dict()
working_path = pathlib.Path().home()
verbose = False  # Updated lazily when settings are loaded

__jupyter_process__ = None
__jupyter_address__ = None

_SETTINGS_MODULE = None
_LOGGING_SETTINGS_APPLIED = False


def _load_settings_module():
    global _SETTINGS_MODULE
    if _SETTINGS_MODULE is None:
        _SETTINGS_MODULE = importlib.import_module("chisurf.core.settings")
    return _SETTINGS_MODULE


def _apply_logging_settings(settings_module) -> None:
    global _LOGGING_SETTINGS_APPLIED, verbose
    if _LOGGING_SETTINGS_APPLIED:
        return
    log_file = getattr(settings_module, "session_log", None)
    level = getattr(settings_module, "log_level", None)
    if not isinstance(level, int):
        level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    has_file = False
    for handler in list(root.handlers):
        try:
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == (
                str(log_file) if log_file else None
            ):
                has_file = True
        except Exception:
            continue
    if log_file and not has_file:
        try:
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
            root.addHandler(fh)
        except Exception:
            pass

    verbose = getattr(settings_module, "verbose", verbose)
    _LOGGING_SETTINGS_APPLIED = True


def _initialize_logging() -> None:
    if globals().get("__logging_initialized__", False):
        return

    env_level = os.environ.get("CHISURF_LOG_LEVEL")
    level = logging.INFO
    if env_level:
        if env_level.isdigit():
            level = int(env_level)
        else:
            level = getattr(logging, env_level.upper(), logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(level)

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    globals()["__logging_initialized__"] = True
    logging.getLogger(__name__).debug("Logging initialized with level=%s (env)", level)


_initialize_logging()


def __getattr__(name: str):
    """Lazily resolve selected subpackages or settings on first access."""
    if name == "plots":
        mod = importlib.import_module("chisurf.gui.plots")
        globals()["plots"] = mod
        return mod
    if name == "settings":
        settings_module = _load_settings_module()
        _apply_logging_settings(settings_module)
        globals()["settings"] = settings_module
        return settings_module
    if name == "verbose":
        settings_module = _load_settings_module()
        _apply_logging_settings(settings_module)
        value = getattr(settings_module, "verbose", verbose)
        globals()["verbose"] = value
        return value
    if name == "history":
        mod = importlib.import_module("chisurf.history")
        value = mod.OperationHistory()
        globals()["history"] = value
        return value
    if name == "actions":
        mod = importlib.import_module("chisurf.core.actions")
        globals()["actions"] = mod
        return mod
    if name == "action_dispatcher":
        mod = importlib.import_module("chisurf.core.actions._infra")
        value = mod.build_default_dispatcher(history_provider=lambda: getattr(sys.modules[__name__], "history", None))
        globals()["action_dispatcher"] = value
        return value
    if name == "action_registry":
        dispatcher = __getattr__("action_dispatcher")
        value = getattr(dispatcher, "registry", None)
        globals()["action_registry"] = value
        return value
    if name == "action_catalog":
        mod = importlib.import_module("chisurf.core.actions._infra")
        value = mod.get_action_catalog
        globals()["action_catalog"] = value
        return value
    if name == "action_execute":
        mod = importlib.import_module("chisurf.core.actions._infra")
        value = mod.invoke_action
        globals()["action_execute"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
