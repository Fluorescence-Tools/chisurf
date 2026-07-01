"""New-style GUI entrypoint for the TTTR Audifier tool.

:class:`TTTRAudifierWidget` is a thin :class:`~qtpy.QtWidgets.QWidget` wrapping a
single :class:`~chisurf.gui.autoform.AutoForm` bound to the Qt-free
:class:`~..gui.view_model.AudifierViewModel` and laid out from
``audifier.view.json``: a persistent dock area with the setup page, the
detector/channel mixer, declarative audio + waterfall parameter panels, and the
reusable general ``waterfall`` image + audio transport. Replaces the former
hand-built ``gui.py`` layout.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register the custom sections)
from .view_model import AudifierViewModel

logger = logging.getLogger(__name__)


class TTTRAudifierWidget(QtWidgets.QWidget):
    """Convert TTTR photon streams to audio, with a live waterfall preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTTR Audifier")
        self.setMinimumSize(760, 560)

        self.model = AudifierViewModel()

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
            logger.warning("Audifier: plot refresh failed", exc_info=True)


__all__ = ["TTTRAudifierWidget"]
