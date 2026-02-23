from __future__ import annotations

import importlib
import logging
import os
import pathlib
import sys

try:
    if sys.version_info >= (3, 8):
        import typing
    elif sys.version_info >= (3, 7):
        import typing_extensions
        import typing
        for key in typing_extensions.__dict__.keys():
            f = typing_extensions.__dict__[key]
            if callable(f):
                typing.__dict__[key] = f
    else:
        import typing_extensions as typing
except ModuleNotFoundError:
    print("WARNING typing_extensions not found", file=sys.stderr)
    typing = None

import chisurf.info

__version__ = chisurf.info.__version__

fits: typing.List["chisurf.fitting.fit.FitGroup"] = list()
imported_datasets: typing.List["chisurf.data.DataGroup"] = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = object         # The current instance of ChiSurf
console = object
experiment: typing.Dict[str, "chisurf.experiments.core.experiment.Experiment"] = dict()
working_path = pathlib.Path().home()
verbose = False  # Updated lazily when settings are loaded

__jupyter_process__ = None
__jupyter_address__ = None

_SETTINGS_MODULE = None
_LOGGING_SETTINGS_APPLIED = False


def _load_settings_module():
    global _SETTINGS_MODULE
    if _SETTINGS_MODULE is None:
        _SETTINGS_MODULE = importlib.import_module("chisurf.settings")
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
        mod = importlib.import_module("chisurf.plots")
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
    if name == "action_dispatcher":
        mod = importlib.import_module("chisurf.runtime.actions")
        value = mod.build_default_dispatcher(history_provider=lambda: getattr(sys.modules[__name__], "history", None))
        globals()["action_dispatcher"] = value
        return value
    if name == "action_registry":
        dispatcher = __getattr__("action_dispatcher")
        value = getattr(dispatcher, "registry", None)
        globals()["action_registry"] = value
        return value
    if name == "action_catalog":
        mod = importlib.import_module("chisurf.runtime.actions")
        value = mod.get_action_catalog
        globals()["action_catalog"] = value
        return value
    if name == "action_execute":
        mod = importlib.import_module("chisurf.runtime.actions")
        value = mod.invoke_action
        globals()["action_execute"] = value
        return value
    if name == "action_controller":
        mod = importlib.import_module("chisurf.controllers.action_controller")
        value = mod.ActionController()
        globals()["action_controller"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
