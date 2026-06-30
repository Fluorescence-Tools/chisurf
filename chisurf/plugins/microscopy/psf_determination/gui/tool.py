"""New-style GUI entrypoint for the PSF Determination plugin.

:class:`PsfDeterminationTool` is a thin :class:`~qtpy.QtWidgets.QWidget` that
wraps a single :class:`~chisurf.gui.autoform.AutoForm` bound to the Qt-free
:class:`~..gui.view_model.PsfViewModel` and laid out from ``psf.view.json``:
the parameters/actions are compact dock panels, the bead stack uses AutoForm's
reusable 3D ``image`` section (z-slider + click-to-pick + bead markers + fit-ROI
circle), and the x/y/z profiles are declarative ``plot`` panels.

Mirrors :class:`chisurf.plugins.microscopy.clsm.gui.tool.CLSMPixelSelect`. The
class keeps the plots refreshed and the setting fields synced when the model
changes; all compute lives in :mod:`..api` via the view-model.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register the psf_controls section)
from .view_model import PsfViewModel

logger = logging.getLogger(__name__)


class PsfDeterminationTool(QtWidgets.QWidget):
    """Interactive 3D PSF determination: stack browsing, bead detection and fitting."""

    def __init__(self, parent=None, embedded: bool = False):
        super().__init__(parent)
        self._embedded = bool(embedded)
        self.setWindowTitle("PSF Determination")
        self.setMinimumSize(700, 520)

        self.model = PsfViewModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)

        self.model.add_observer(self._on_model_event)

    # ── model wiring ────────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        try:
            self.auto_form.sync_fields()
        except Exception:
            logger.warning("PSF: field sync failed", exc_info=True)
        try:
            self.auto_form.refresh_plots()
        except Exception:
            logger.warning("PSF: plot refresh failed", exc_info=True)


__all__ = ["PsfDeterminationTool"]
