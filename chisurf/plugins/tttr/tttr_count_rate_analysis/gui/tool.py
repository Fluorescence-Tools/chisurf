"""New-style GUI entrypoint for the Count Rate Analysis tool.

:class:`CountRateAnalyzer` is a thin :class:`~qtpy.QtWidgets.QWidget` wrapping a
single :class:`~chisurf.gui.autoform.AutoForm` bound to the Qt-free
:class:`~..gui.view_model.CountRateViewModel` and laid out from
``count_rate.view.json``: a persistent dock area with the channel-definition
page, the file list, the count-rate plot and the results table. Replaces the
former hand-built layout. Mirrors the PSF / ALEX tools.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register the custom sections)
from .view_model import CountRateViewModel

logger = logging.getLogger(__name__)


class CountRateAnalyzer(QtWidgets.QWidget):
    """Count rates per detector channel across many TTTR files (mean/std + plot)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Count Rate Analysis")
        self.setMinimumSize(700, 500)

        self.model = CountRateViewModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)

        self.model.add_observer(self._on_model_event)

    def _on_model_event(self, event: str) -> None:
        try:
            self.auto_form.refresh_plots()
        except Exception:
            logger.warning("Count rate: plot refresh failed", exc_info=True)


__all__ = ["CountRateAnalyzer"]
