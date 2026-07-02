"""Interactive CLSM pixel-select tool, assembled by AutoForm.

The whole tool is a single :class:`AutoForm` bound to a :class:`ClsmViewModel`
and laid out from ``clsm.view.json`` as one dock area: the File/Acquisition/
Brush&Decay settings are compact foldable dock panels, the image is AutoForm's
reusable brush+colormap ``image`` section, and the decay/FRC are declarative
``plot`` panels. All panels are draggable docks (no splitter). This class keeps
the plots refreshed and the setting fields synced when the model changes, and
exposes ``apply_setup_settings`` for the Imaging-Tools aggregator.
"""

from __future__ import annotations

from qtpy import QtWidgets

import chisurf as cs
from chisurf.gui.autoform import AutoForm

from . import sections  # noqa: F401  (side effect: register custom sections)
from .view_model import ClsmViewModel

log = cs.logging


class CLSMPixelSelect(QtWidgets.QWidget):
    """CLSM image representation, pixel selection and decay export."""

    def __init__(self, parent=None, embedded: bool = False):
        super().__init__(parent)
        self._embedded = bool(embedded)
        self.setWindowTitle("CLSM Pixel Select")

        self.model = ClsmViewModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)

        self.model.add_observer(self._on_model_event)

    # ── model wiring ───────────────────────────────────────────────────
    def _on_model_event(self, event: str) -> None:
        if event == "setup":
            # Marker auto-detection changed model.setup; re-read the form fields.
            try:
                self.auto_form.sync_fields()
            except Exception:
                log.warning("CLSM: field sync failed", exc_info=True)
        if event in ("decay", "image", "selection", "setup"):
            try:
                self.auto_form.refresh_plots()
            except Exception:
                log.warning("CLSM: plot refresh failed", exc_info=True)

    # ── Imaging-Tools shared-setup adapter ─────────────────────────────
    def apply_setup_settings(self, payload: dict) -> None:
        """Map a shared detector definition onto the setup channels."""
        if not payload:
            return
        detectors = payload.get("detectors") or {}
        channels: list[int] = []
        for det in detectors.values():
            if not isinstance(det, dict):
                continue
            for ch in det.get("chs", []) or []:
                if ch not in channels:
                    channels.append(ch)
        if channels:
            self.model.setup.channels_text = ",".join(str(c) for c in channels)
            self.model.notify("setup")


__all__ = ["CLSMPixelSelect"]
