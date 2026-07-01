"""New-style GUI entrypoint for the PTU Header Editor tool.

:class:`TagsEditor` is a thin :class:`~qtpy.QtWidgets.QWidget` wrapping a single
:class:`~chisurf.gui.autoform.AutoForm` bound to the Qt-free
:class:`~..gui.view_model.HeaderEditorViewModel` and laid out from
``header.view.json``: a dock area with the editable tag table and a collapsible
read-only JSON view. Replaces the former hand-built ``wizard.py``. Mirrors the PSF
tool. ``TagsEditor`` keeps a compatible no-arg / optional-``json_data`` signature.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register the custom section)
from .view_model import HeaderEditorViewModel

logger = logging.getLogger(__name__)


class TagsEditor(QtWidgets.QWidget):
    """View and edit PicoQuant PTU header tags, then write a modified PTU."""

    def __init__(self, json_data: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PTU Header Editor")
        self.setMinimumSize(640, 480)

        self.model = HeaderEditorViewModel()
        if json_data:
            try:
                self.model.load_json(json_data)
            except Exception:
                logger.warning("PTU header: could not load initial json_data", exc_info=True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)

        self.model.add_observer(self._on_model_event)

    def _on_model_event(self, event: str) -> None:
        try:
            self.auto_form.sync_fields()
        except Exception:
            logger.warning("PTU header: field sync failed", exc_info=True)


__all__ = ["TagsEditor"]
