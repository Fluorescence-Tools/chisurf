"""New-style GUI entrypoint for the Molecule-wise MLE plugin.

``SmImageMleTool`` subclasses the existing
:class:`~chisurf.plugins.microscopy.sm_image_mle.MainWindow` to add the shared
setup adapter used by the Imaging Tools aggregator.  The tool has no embedded
detector wizard (it uses simple channel / micro-time controls), so ``embedded``
is accepted for API uniformity but changes nothing on its own; standalone use is
unaffected.

``apply_setup_settings(payload)`` maps a detector definition (the dict shape
produced by ``DetectorWizardPage.get_settings()``) onto the tool's detector
channel line edit and micro-time range spin boxes.

Importing this module requires Qt; it is only loaded lazily via the
``__getattr__`` hook in the package ``__init__.py``.
"""

from __future__ import annotations

import logging

# ``MainWindow`` is defined at module level in the package ``__init__`` (not
# behind the lazy ``__getattr__`` gate), so this import is safe.
from chisurf.plugins.microscopy.sm_image_mle import MainWindow

logger = logging.getLogger(__name__)


class SmImageMleTool(MainWindow):
    """Molecule-wise MLE tool with shared-setup support."""

    def __init__(self, parent=None, embedded: bool = False):
        super().__init__(parent)
        self._embedded = bool(embedded)

    def apply_setup_settings(self, payload: dict) -> None:
        """Map a shared detector definition onto this tool's controls."""
        if not payload:
            return
        detectors = payload.get("detectors") or {}
        channels: list[int] = []
        micro_range = None
        for det in detectors.values():
            if not isinstance(det, dict):
                continue
            for ch in det.get("chs", []) or []:
                if ch not in channels:
                    channels.append(ch)
            if micro_range is None:
                ranges = det.get("mtr") or det.get("microtime_ranges")
                if ranges:
                    micro_range = ranges[0]

        line_edit = getattr(self, "detector_lineedit", None)
        if line_edit is not None and channels:
            try:
                line_edit.setText(" ".join(str(ch) for ch in channels))
            except Exception:  # pragma: no cover - best-effort GUI sync
                logger.debug("Could not set detector channels", exc_info=True)

        if micro_range and len(micro_range) == 2:
            try:
                self.mtr_start_spin.setValue(int(micro_range[0]))
                self.mtr_stop_spin.setValue(int(micro_range[1]))
            except Exception:  # pragma: no cover - best-effort GUI sync
                logger.debug("Could not set micro-time range", exc_info=True)


__all__ = ["SmImageMleTool"]
