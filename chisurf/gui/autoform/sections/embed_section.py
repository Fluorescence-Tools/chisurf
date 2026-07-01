"""A general ``embed`` custom section: host any existing widget inside a form.

Lets a ``.view.json`` drop a ready-made widget (a plugin editor, a tool panel)
straight into an AutoForm — no per-widget glue code. This is what attaches the
FCS channel and detector-definition editors *inside* the onboarding wizard
(instead of popping them as detached floating windows), and it is reusable by any
wizard/tool that needs to embed an existing widget.

Authoring::

    {"type": "custom", "key": "embed",
     "options": {"widget": "chisurf.plugins.fcs.fcs_channel_preset.gui.tool:FCSChannelWidget"}}

``options`` keys:

* ``widget`` (required) — dotted import path ``"pkg.module:ClassName"`` or
  ``"pkg.module.ClassName"``.
* ``kwargs`` — mapping passed to the widget constructor.
* ``pass_model`` — when ``True``, the bound model is passed as ``model=`` kwarg.
* ``expanding`` — mark the widget to take spare vertical space (default ``True``).
"""

from __future__ import annotations

import importlib

from qtpy import QtWidgets

from chisurf import logging

from .registry import register_section


def _resolve(path: str):
    """Import and return the object named by ``"pkg.module:Attr"`` or ``"pkg.module.Attr"``."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


@register_section("embed")
def _embed_section_factory(model, target=None, **options):
    """Instantiate and return the widget named by ``options['widget']``."""
    path = options.get("widget", "")
    if not path:
        logging.warning("embed section: no 'widget' path given")
        return None
    try:
        cls = _resolve(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(f"embed section: could not import {path!r}: {exc}")
        return None

    kwargs = dict(options.get("kwargs", {}) or {})
    if options.get("pass_model", False):
        kwargs.setdefault("model", model)
    try:
        widget = cls(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(f"embed section: could not construct {path!r}: {exc}")
        return None

    if options.get("expanding", True):
        widget._autoform_expanding = True
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    return widget


__all__ = ["_embed_section_factory"]
