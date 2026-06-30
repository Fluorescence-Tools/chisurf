"""New-style GUI entrypoint for the TTTR Split / Convert tool.

:class:`PTUSplitter` is a thin :class:`~qtpy.QtWidgets.QWidget` wrapping a single
:class:`~chisurf.gui.autoform.AutoForm` bound to the Qt-free
:class:`~..gui.view_model.SplitterViewModel` and laid out from
``splitter.view.json``: foldable Input/Output, Split-options and Batch panels.
Replaces the former ``wizard.ui`` / hand-built layout. Mirrors the PSF
Determination tool.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register the custom sections)
from .view_model import SplitterViewModel

logger = logging.getLogger(__name__)


class PTUSplitter(QtWidgets.QWidget):
    """TTTR file splitter / container converter with an inline batch panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTTR Split / Convert")
        self.setMinimumWidth(420)

        self.model = SplitterViewModel()

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
            logger.warning("Splitter: field sync failed", exc_info=True)


__all__ = ["PTUSplitter"]
