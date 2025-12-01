from __future__ import annotations

import sys
import logging
import pathlib

try:
    if sys.version_info >= (3, 8):
        import typing
    elif sys.version_info >= (3, 7):
        # monkey patch the 3.7 typing system as
        # TypedDict etc. is missing
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

import chisurf.settings
import chisurf.info

__version__ = chisurf.info.__version__

fits: typing.List[chisurf.fitting.fit.FitGroup] = list()
imported_datasets: typing.List[chisurf.data.DataGroup] = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = object         # The current instance of ChiSurf
console = object
experiment: typing.Dict[str, chisurf.experiments.experiment.Experiment] = dict()
working_path = pathlib.Path().home()
verbose = chisurf.settings.verbose

# Jupyter
__jupyter_process__ = None
__jupyter_address__ = None

# Initialize logging early and idempotently so we capture startup issues before GUI widgets exist.
try:
    if not globals().get("__logging_initialized__", False):
        # Determine log level from settings; fall back to INFO
        try:
            _level = chisurf.settings.cs_settings.get('log_level', None)
        except Exception:
            _level = None
        if not isinstance(_level, int):
            _level = getattr(chisurf.settings, 'log_level', None)
        if not isinstance(_level, int):
            _level = logging.INFO

        # Determine session log file
        log_file = getattr(chisurf.settings, 'session_log', None)

        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        root = logging.getLogger()
        root.setLevel(_level)

        # Attach file handler if not already attached
        has_file = False
        for h in list(root.handlers):
            try:
                if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == (str(log_file) if log_file else None):
                    has_file = True
            except Exception:
                pass
        if log_file and not has_file:
            try:
                fh = logging.FileHandler(str(log_file), encoding='utf-8')
                fh.setLevel(_level)
                fh.setFormatter(fmt)
                root.addHandler(fh)
            except Exception:
                # If file handler fails (e.g., path issues), continue with console only
                pass

        # Attach stderr stream handler if not already attached
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers)
        if not has_stream:
            sh = logging.StreamHandler(stream=sys.stderr)
            sh.setLevel(_level)
            sh.setFormatter(fmt)
            root.addHandler(sh)

        __logging_initialized__ = True
        logging.getLogger(__name__).debug("Early logging initialized (level=%s, file=%s)", _level, log_file)
except Exception:
    # Last resort: basic stderr logging
    logging.basicConfig(level=logging.INFO)


def __getattr__(name: str):
    """Lazily resolve selected subpackages on first attribute access.

    This allows references such as ``chisurf.plots`` in modules that are
    imported during doctest collection or other partial initialization stages
    before the submodule has been attached explicitly.
    """
    if name == "plots":
        import importlib
        mod = importlib.import_module("chisurf.plots")
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
