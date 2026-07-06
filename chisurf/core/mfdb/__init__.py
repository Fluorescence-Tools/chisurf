"""Compatibility facade for the vendored :mod:`mfdb` package.

MFDB now lives under ``modules/mfdb/src/mfdb`` so it can be split from
ChiSurf cleanly. Existing ``chisurf.core.mfdb`` imports are kept during the
prerelease cutover, but new code should import ``mfdb`` directly.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from typing import Any

_VENDORED_SRC = Path(__file__).resolve().parents[3] / "modules" / "mfdb" / "src"
if str(_VENDORED_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDORED_SRC))

_mfdb = importlib.import_module("mfdb")
__path__ = list(getattr(_mfdb, "__path__", []))
__all__ = list(getattr(_mfdb, "__all__", []))


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve every ``chisurf.core.mfdb.X`` to the real ``mfdb.X`` module.

    Without this, a submodule imported through the facade *before* the real
    ``mfdb.X`` is loaded would be created as a **duplicate** module object (the
    ``__path__`` above points at the vendored package, so the default finder would
    load a second copy). Duplicate modules mean duplicate module-level singletons —
    e.g. two event buses — so a subscriber reached through the facade and a
    publisher inside ``mfdb`` would never meet. This finder aliases the facade name
    to the already-canonical ``mfdb`` module instead, guaranteeing one object.
    """

    _prefix = __name__ + "."

    def find_spec(self, name, path=None, target=None):
        if not name.startswith(self._prefix):
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        real_name = "mfdb." + spec.name[len(self._prefix):]
        module = importlib.import_module(real_name)
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module):  # already executed as the real mfdb.* module
        pass


sys.meta_path.insert(0, _AliasFinder())

for _module_name, _module in list(sys.modules.items()):
    if _module_name == "mfdb" or _module_name.startswith("mfdb."):
        sys.modules.setdefault(__name__ + _module_name.removeprefix("mfdb"), _module)

for _name in __all__:
    if hasattr(_mfdb, _name):
        globals()[_name] = getattr(_mfdb, _name)


def __getattr__(name: str) -> Any:
    """Delegate unresolved attributes to the vendored package."""
    return getattr(_mfdb, name)
