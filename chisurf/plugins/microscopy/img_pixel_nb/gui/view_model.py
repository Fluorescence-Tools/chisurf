"""Qt-free view-model backing the Number & Brightness imaging tool.

Computes N / B / epsilon per pixel for **every** detector window defined in the
step-0 setup, and adds them to the standard imaging HDF5.
"""

from __future__ import annotations

import pathlib

from chisurf.plugins.microscopy.imaging_common.base import ImagingMapViewModel

_VIEW_JSON = pathlib.Path(__file__).parent / "nb.view.json"


class NBViewModel(ImagingMapViewModel):
    """State + logic for the interactive N&B imaging tool (no Qt)."""

    HDF5_ACTION_LABEL = "➕ Add N&B to HDF5"
    WINDOW_KIND = "nb"

    def __init__(self) -> None:
        super().__init__(_VIEW_JSON)

    # ── image accessors (view.json `image` sections; displayed window) ──
    def n_map(self):
        """Return the apparent-number (N) map of the displayed window."""
        return self._disp("N")

    def b_map(self):
        """Return the apparent-brightness (B) map of the displayed window."""
        return self._disp("B")

    def epsilon_map(self):
        """Return the molecular-brightness (epsilon) map of the displayed window."""
        return self._disp("epsilon")
