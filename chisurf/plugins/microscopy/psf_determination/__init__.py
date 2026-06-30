"""PSF Determination plugin — new-style, AutoForm-based.

The plugin is split into Qt-free and GUI layers (like ``burst_selection`` /
``flc_2d`` / ``clsm``):

- :mod:`.api`      — pure PSF computation (``gaussian_3d``, ``extract_roi``,
  ``fit_3d_gaussian``, ``detect_beads``, ``fit_all_beads``) + dataclasses + contract.
- :mod:`.backend`  — ZMQ/JSON-RPC service handlers.
- :mod:`.cli`      — ``psf-determination`` command-line interface.
- :mod:`.gui`      — the interactive tool: a Qt-free :class:`~.gui.view_model.PsfViewModel`
  rendered by :class:`~chisurf.gui.autoform.AutoForm` from ``gui/psf.view.json``.

Qt is only imported lazily through the :class:`PsfDeterminationTool` gate below,
so importing this package stays headless-safe.
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path

from .api import gaussian_3d  # noqa: F401  (re-exported for back-compat)

name = "Imaging:PSF Determination"

_manifest_path = _Path(__file__).parent / "manifest.json"
if _manifest_path.exists():
    _manifest = _json.loads(_manifest_path.read_text())
    name = _manifest.get("display_name", name)

cli_entrypoint = "psf-determination=chisurf.plugins.microscopy.psf_determination.cli:cli"


def __getattr__(attr_name: str):
    """Lazy Qt gate for the new-style GUI entrypoint."""
    if attr_name == "PsfDeterminationTool":
        from .gui.tool import PsfDeterminationTool as _cls

        globals()["PsfDeterminationTool"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


__all__ = ["PsfDeterminationTool", "gaussian_3d"]
