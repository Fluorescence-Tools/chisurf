"""Qt-free view-model backing the ALEX Creator tool.

Thin state holder that binds AutoForm controls and delegates every computation to
the plugin's Qt-free :mod:`..core` / :mod:`..api` layers (shared with the CLI and
RPC backend). Owns no conversion logic itself.

Mirrors :class:`chisurf.plugins.tttr.tttr_splitter.gui.view_model.SplitterViewModel`.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

from .. import core
from ..api import AlexRequest, run

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "alex.view.json"


class AlexViewModel:
    """State + view wiring for the ALEX Creator tool (logic lives in ``core``)."""

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

        # ── single-file preview state ───────────────────────────────────
        self.input_file = ""
        self._tttr = None
        self._observers: list[Callable[[str], None]] = []

        # ── batch state ────────────────────────────────────────────────
        #: files to batch-process (typically ``.sm`` ALEX measurements).
        self.batch_files: list[str] = []
        #: ``"convert"`` = one ALEX file out per input; ``"merge"`` = one combined file.
        self.batch_mode = "convert"
        self.batch_output_folder = ""

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
        return core.input_format_options()

    def output_format_options(self) -> list[str]:
        """Output container choices (every supported tttrlib container name)."""
        return core.supported_containers()

    @property
    def has_data(self) -> bool:
        """Whether a TTTR file is loaded and ready to convert/save."""
        return self._tttr is not None

    # ── loading (preview) ───────────────────────────────────────────────
    def load(self, path: str) -> None:
        """Load a TTTR file from *path* and refresh the preview."""
        self.input_file = path
        self._tttr = core.load(path, core.resolve_filetype(self.input_format, path))
        self.notify("loaded")

    def set_tttr(self, tttr, path: str) -> None:
        """Store an already-loaded TTTR object and its source path."""
        self.input_file = str(path)
        self._tttr = tttr
        self.notify("loaded")

    def histogram_series(self) -> list[dict]:
        """Return the ALEX micro-time histogram for the ``plot`` section."""
        if self._tttr is None or not self.input_file:
            return []
        filetype = core.resolve_filetype(self.input_format, self.input_file)
        counts = core.alex_histogram(self.input_file, self.alex_period, self.period_shift, filetype)
        bins = list(range(len(counts)))
        return [{"x": bins, "y": counts, "name": "ALEX µ-time", "color": "#4488ff", "width": 1}]

    # ── single-file save ────────────────────────────────────────────────
    def can_save(self) -> str | None:
        """Return ``None`` when a save can run, else a human-readable reason."""
        if self._tttr is None:
            return "Please load a TTTR file first."
        return None

    def default_save_name(self) -> str:
        """Suggested output filename for the current output container."""
        return core.default_output_name(self.input_file, self.output_format)

    def save(self, path: str) -> None:
        """Write the ALEX-converted single file to *path*."""
        if self._tttr is None:
            raise ValueError("Please load a TTTR file first.")
        core.convert_file(
            self.input_file,
            path,
            self.alex_period,
            self.period_shift,
            self.output_format,
            self.input_format,
        )

    # ── batch ───────────────────────────────────────────────────────────
    def add_batch_files(self, paths: list[str]) -> None:
        """Add unique files to the batch list."""
        added = False
        for p in paths:
            if p and p not in self.batch_files:
                self.batch_files.append(p)
                added = True
        if added:
            self.notify("batch")

    def clear_batch(self) -> None:
        """Empty the batch file list."""
        self.batch_files = []
        self.notify("batch")

    def can_run_batch(self) -> str | None:
        """Return ``None`` when a batch run can proceed, else a reason string."""
        if not self.batch_files:
            return "Add .sm (or other TTTR) files to the batch list first."
        if self.batch_mode == "convert" and not self.batch_output_folder.strip():
            return "Choose an output folder for the converted files."
        return None

    def run_batch(self) -> list[str]:
        """Convert each batch file, or merge them all, per :attr:`batch_mode`.

        Returns the list of written output paths.
        """
        reason = self.can_run_batch()
        if reason is not None:
            raise ValueError(reason)
        request = AlexRequest(
            files=list(self.batch_files),
            alex_period=int(self.alex_period),
            period_shift=int(self.period_shift),
            output_format=self.output_format,
            input_format=self.input_format,
            mode=self.batch_mode,
            output_dir=self.batch_output_folder.strip(),
        )
        return run(request).output_paths


__all__ = ["AlexViewModel"]
