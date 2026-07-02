"""New-style GUI entrypoint for the Pixel-wise MLE plugin.

``ImgPixelMleTool`` subclasses the existing
:class:`~chisurf.plugins.microscopy.img_pixel_mle.imgmle.LifetimeMleAnalysisWizard`
to add two embedding hooks used by the Imaging Tools aggregator:

* ``embedded=True`` hides the tool's own "Detector Definition" tab (the shared
  Setup panel provides it instead) — standalone use (``embedded=False``, the
  default) keeps the tab and is completely unchanged.
* ``apply_setup_settings(payload)`` loads a detector definition (the dict shape
  produced by ``DetectorWizardPage.get_settings()``) into the tool's channel
  definer, so the shared Setup panel can feed it over RPC.
"""

from __future__ import annotations

import logging

from chisurf.plugins.microscopy.img_pixel_mle.imgmle import LifetimeMleAnalysisWizard

logger = logging.getLogger(__name__)

_DETECTOR_TAB_NAME = "Detector Definition"


class ImgPixelMleTool(LifetimeMleAnalysisWizard):
    """Pixel-wise MLE wizard with shared-setup embedding support."""

    #: File-input widgets fed instead by the pipeline / IRF & BG step.
    _CARRIED_INPUT_WIDGETS = (
        "tttr_list", "browse_tttr_button", "clear_tttr_button",
        "irf_list", "browse_irf_button", "clear_irf_button",
        "bg_list", "browse_bg_button", "clear_bg_button",
    )

    def __init__(self, parent=None, embedded: bool = False, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self._embedded = bool(embedded)
        self._pipeline_calibration: dict = {}
        if self._embedded:
            self._hide_detector_tab()
            self._hide_carried_inputs()

    def _hide_detector_tab(self) -> None:
        """Remove the embedded detector tab when a shared Setup panel is present."""
        tab_widget = getattr(self, "tab_widget", None)
        if tab_widget is None:
            return
        for index in range(tab_widget.count()):
            if tab_widget.tabText(index) == _DETECTOR_TAB_NAME:
                tab_widget.removeTab(index)
                break

    def _hide_carried_inputs(self) -> None:
        """Hide the TTTR / IRF / BG file inputs — carried from upstream steps."""
        for name in self._CARRIED_INPUT_WIDGETS:
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.setVisible(False)
                    box = widget.parentWidget()
                    if box is not None and box.__class__.__name__ == "QGroupBox":
                        box.setVisible(False)
                except Exception:
                    logger.debug("could not hide %s", name, exc_info=True)

    def apply_pipeline_context(self, payload: dict) -> None:
        """Load the pipeline's source TTTR (no separate TTTR-data input needed)."""
        source = (payload or {}).get("source")
        if not source:
            return
        try:
            tttr_list = getattr(self, "tttr_list", None)
            if tttr_list is not None and str(source) not in tttr_list.get_selected_files():
                tttr_list.add_file(str(source))
                if callable(getattr(self, "update_tttr_files", None)):
                    self.update_tttr_files()
        except Exception:
            logger.debug("apply_pipeline_context (MLE) failed", exc_info=True)

    def apply_calibration(self, calibration: dict) -> None:
        """Adopt the per-detector IRF file + background from the IRF & BG step.

        The IRF file for the currently-selected detector is loaded into the MLE's
        IRF list, and its background feeds the fixed-background field — so the
        IRF/BG are carried on rather than re-entered here.
        """
        self._pipeline_calibration = dict(calibration or {})
        try:
            detector = None
            combo = getattr(self, "comboBox_detector_select", None)
            if combo is not None:
                detector = combo.currentText()
            entry = self._pipeline_calibration.get(detector) if detector else None
            if not entry:
                return
            irf_files = entry.get("irf") or []
            if isinstance(irf_files, str):
                irf_files = [irf_files]
            irf_list = getattr(self, "irf_list", None)
            if irf_list is not None and irf_files:
                existing = set(irf_list.get_selected_files())
                for irf in irf_files:
                    if irf not in existing:
                        irf_list.add_file(str(irf))
                if callable(getattr(self, "update_irf_files", None)):
                    self.update_irf_files()
            bg = float(entry.get("bg_vv") or 0.0) + float(entry.get("bg_vh") or 0.0)
            bg_spin = getattr(self, "bg_fixed_spinbox", None) or getattr(self, "doubleSpinBox_bg", None)
            if bg and bg_spin is not None:
                bg_spin.setValue(bg)
            # Carry the convolution/fit window (separate from the IRF range).
            start, stop = int(entry.get("conv_start") or 0), int(entry.get("conv_stop") or 0)
            if stop > start:
                for name, val in (("micro_time_start_spinbox", start), ("micro_time_stop_spinbox", stop)):
                    spin = getattr(self, name, None)
                    if spin is not None:
                        spin.setValue(val)
        except Exception:
            logger.debug("apply_calibration (MLE) failed", exc_info=True)

    def apply_setup_settings(self, payload: dict) -> None:
        """Apply a shared detector definition to this tool's channel definer."""
        definer = getattr(self, "channel_definer", None)
        if definer is None or not payload:
            return
        try:
            definer.load_data_into_tables(payload)
        except Exception:  # pragma: no cover - best-effort GUI sync
            logger.debug("load_data_into_tables failed", exc_info=True)
            return
        for hook in ("_on_detectors_updated", "_apply_setup_from_wizard"):
            fn = getattr(self, hook, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # pragma: no cover - best-effort GUI sync
                    logger.debug("%s failed during setup apply", hook, exc_info=True)


__all__ = ["ImgPixelMleTool"]
