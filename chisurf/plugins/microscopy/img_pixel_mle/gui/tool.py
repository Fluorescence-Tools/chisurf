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

    def __init__(self, parent=None, embedded: bool = False, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self._embedded = bool(embedded)
        if self._embedded:
            self._hide_detector_tab()

    def _hide_detector_tab(self) -> None:
        """Remove the embedded detector tab when a shared Setup panel is present."""
        tab_widget = getattr(self, "tab_widget", None)
        if tab_widget is None:
            return
        for index in range(tab_widget.count()):
            if tab_widget.tabText(index) == _DETECTOR_TAB_NAME:
                tab_widget.removeTab(index)
                break

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
