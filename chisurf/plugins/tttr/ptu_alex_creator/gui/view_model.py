"""Qt-free view-model backing the ALEX Creator tool.

:class:`AlexViewModel` holds the ALEX-to-micro-time conversion settings that
AutoForm binds its controls to, performs the conversion through :mod:`tttrlib`,
and exposes the resulting micro-time histogram for the declarative ``plot``
section. Free of Qt so it is unit-testable headlessly; the GUI
(``gui.tool`` + ``gui.sections``) owns Qt concerns and drives this model.

Mirrors :class:`chisurf.plugins.tttr.tttr_splitter.gui.view_model.SplitterViewModel`.
"""

from __future__ import annotations

import logging
import os
import pathlib
from collections.abc import Callable

import numpy as np
import tttrlib

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "alex.view.json"

# container name → (extension stem, record-type id, container id)
_CONTAINER_INFO = {
    "PTU": ("ptu", 4, 0),
    "HT3": ("ht3", 4, 1),
    "SPC-130": ("spc", 7, 2),
    "SPC-600_256": ("spc", 8, 3),
    "SPC-600_4096": ("spc", 9, 4),
    "PHOTON-HDF5": ("hdf", 4, 5),
    "CZ-RAW": ("raw", 10, 6),
    "SM": ("sm", 11, 7),
}


class AlexViewModel:
    """State + logic for the ALEX Creator tool (no Qt)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``alex.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        # ── AutoForm-bound settings ─────────────────────────────────────
        self.input_format = "Auto"
        self.output_format = "PTU"
        self.alex_period = 8000
        self.period_shift = 0

        # ── runtime state ──────────────────────────────────────────────
        self.input_file = ""
        self._tttr = None
        self._processed = None
        self._observers: list[Callable[[str], None]] = []

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("alex observer failed", exc_info=True)

    def update(self) -> None:
        """Recompute the preview after a bound field changes (AutoForm hook)."""
        self.notify("plot")

    # ── AutoForm options sources ───────────────────────────────────────
    def input_format_options(self) -> list[str]:
        """Input container choices: ``Auto`` plus every tttrlib container name."""
        return ["Auto", *tttrlib.TTTR.get_supported_container_names()]

    def output_format_options(self) -> list[str]:
        """Output container choices (every supported tttrlib container name)."""
        return list(tttrlib.TTTR.get_supported_container_names())

    @property
    def tttr_filetype(self) -> str | None:
        """The forced input container, ``None`` when ``Auto`` (let tttrlib infer)."""
        if self.input_format != "Auto":
            return self.input_format
        if self.input_file and os.path.exists(self.input_file):
            file_type_int = tttrlib.inferTTTRFileType(self.input_file)
            names = tttrlib.TTTR.get_supported_container_names()
            if file_type_int is not None and 0 <= file_type_int < len(names):
                return names[file_type_int]
        return None

    @property
    def has_data(self) -> bool:
        """Whether a TTTR file is loaded and ready to convert/save."""
        return self._tttr is not None

    # ── loading ────────────────────────────────────────────────────────
    def load(self, path: str) -> None:
        """Load a TTTR file from *path* and refresh the preview."""
        self.input_file = path
        self._tttr = tttrlib.TTTR(path, self.tttr_filetype)
        self._processed = None
        self.notify("loaded")

    def set_tttr(self, tttr, path: str) -> None:
        """Store an already-loaded TTTR object and its source path."""
        self.input_file = str(path)
        self._tttr = tttr
        self._processed = None
        self.notify("loaded")

    # ── conversion / preview ───────────────────────────────────────────
    def _compute_processed(self):
        """Apply the ALEX→micro-time conversion to a fresh copy of the file."""
        if self._tttr is None or not self.input_file:
            self._processed = None
            return None
        tt = tttrlib.TTTR(self.input_file, self.tttr_filetype)
        tt.alex_to_microtime(int(self.alex_period), int(self.period_shift))
        self._processed = tt
        return tt

    def histogram_series(self) -> list[dict]:
        """Return the ALEX micro-time histogram for the ``plot`` section."""
        tt = self._compute_processed()
        if tt is None:
            return []
        period = int(self.alex_period)
        counts = np.bincount(tt.micro_times, minlength=period)[:period]
        bins = np.arange(period)
        return [{"x": bins, "y": counts, "name": "ALEX µ-time", "color": "#4488ff", "width": 1}]

    # ── save ───────────────────────────────────────────────────────────
    def can_save(self) -> str | None:
        """Return ``None`` when a save can run, else a human-readable reason."""
        if self._tttr is None:
            return "Please load a TTTR file first."
        return None

    def default_save_name(self) -> str:
        """Suggested output filename for the current output container."""
        ext = _CONTAINER_INFO.get(self.output_format, ("ptu", 0, 0))[0]
        base = pathlib.Path(self.input_file).stem if self.input_file else "alex"
        return f"{base}_alex.{ext}"

    def save(self, path: str) -> None:
        """Write the ALEX-converted file to *path* in the chosen container."""
        if self._processed is None:
            self._compute_processed()
        if self._processed is None:
            raise ValueError("Failed to process TTTR data")
        tt = self._processed
        out_name = self.output_format

        if out_name != self.input_format and out_name != "Auto":
            ext, rec, cont = _CONTAINER_INFO.get(out_name, ("ptu", 4, 0))
            header = tt.header
            header.tttr_container_type = cont
            header.tttr_record_type = rec
            if out_name == "PTU":
                # PTU via HydraHarp wants the special tag group 0x00010304.
                header.set_tag("TTResultFormat_TTTRRecType", 0x00010304, 268435464)
                header.set_tag("TTResultFormat_BitsPerRecord", 32, 268435464)
                header.set_tag("MeasDesc_RecordType", rec, 268435464)
            else:
                # 268435464 == 0x10000008 (Int8) — the tag value type tttrlib expects.
                header.set_tag("TTResultFormat_TTTRRecType", rec, 268435464)
                header.set_tag("MeasDesc_RecordType", rec, 268435464)
            tt.write(path, header)
        else:
            tt.write(path)


__all__ = ["AlexViewModel"]
