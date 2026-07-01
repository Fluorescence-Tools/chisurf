"""New-style GUI entrypoint for the ALEX Creator tool.

:class:`AlexPTUCreator` is a thin :class:`~qtpy.QtWidgets.QWidget` wrapping a
single :class:`~chisurf.gui.autoform.AutoForm` bound to the Qt-free
:class:`~..gui.view_model.AlexViewModel` and laid out from ``alex.view.json``:
a dock area with a collapsible Controls panel and the live micro-time histogram.
Replaces the former hand-built ``wizard.py`` layout. Mirrors the PSF tool.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register the custom sections)
from .view_model import AlexViewModel

logger = logging.getLogger(__name__)


class AlexPTUCreator(QtWidgets.QWidget):
    """Map ALEX macro-time alternation into micro-time and save as PTU/other."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ALEX Creator")
        self.setMinimumSize(700, 480)

        self.model = AlexViewModel()

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
            logger.warning("ALEX: field sync failed", exc_info=True)
        try:
            self.auto_form.refresh_plots()
        except Exception:
            logger.warning("ALEX: plot refresh failed", exc_info=True)


__all__ = ["AlexPTUCreator"]
