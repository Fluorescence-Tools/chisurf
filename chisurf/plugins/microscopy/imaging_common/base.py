"""Shared base classes for the per-pixel imaging map tools.

Both an AutoForm-friendly Qt-free view-model base (:class:`ImagingMapViewModel`)
and a host widget (:class:`ImagingMapTool`) with the action **toolbar** and
**file-drop** support. The Intensity / N&B / Phasor tools are thin subclasses:
the view-model implements ``_compute`` and the image accessors, the host is a
one-line subclass. The per-pixel results use the standard imaging-HDF5 (the same
flat, column-extensible ``results`` table used for smFRET burst analysis and
readable by ndxplorer): a new analysis just adds columns.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

#: Worker-result keys that are display-only (movie stacks / raw intensity) and
#: must never become per-pixel HDF5 columns.
_DISPLAY_ONLY_KEYS = frozenset({"intensity", "frames", "g_frames", "s_frames", "mt_frames"})


class ImagingMapViewModel:
    """Qt-free base view-model for per-pixel imaging map tools.

    Subclasses set :attr:`RESULT_KEYS` (map names written to the HDF5) and
    ``_view_json``, implement :meth:`_compute` returning
    ``{"maps": dict, "shape": (ny, nx)}``, and add the image-accessor methods the
    ``image`` view.json sections read.
    """

    RESULT_KEYS: tuple[str, ...] = ()
    #: Label of the HDF5 toolbar action (Intensity overrides to "Create").
    HDF5_ACTION_LABEL: str = "➕ Add to HDF5"

    def __init__(self, view_json: pathlib.Path) -> None:
        self._view_json = pathlib.Path(view_json)
        # ── AutoForm-bound settings shared by all tools ──
        self.filename: str = ""
        self.colormap: str = "magma"
        #: Window shown in the map/plot sections (computation covers all windows).
        self.display_window: str = ""
        # ── detector windows (from the step-0 setup); {name: {chs, micro_time_ranges}} ──
        self.detectors: dict[str, dict] = {}
        # ── runtime state ──
        #: Per-window base maps: {window: {base_key: 2-D array}} (for display).
        self._by_window: dict[str, dict[str, np.ndarray]] = {}
        #: Flat, HDF5-ready columns (MFD-named intensity + '<key> (win)').
        self._columns: dict[str, np.ndarray] = {}
        self.results_text: str = "Load a TTTR imaging file (or drop one) and press Run."
        self._observers: list[Callable[[str], None]] = []
        # ── shared pipeline context (set by the imaging_tools coordinator) ──
        #: Remembered imaging-HDF5 for the current image (enrichment target).
        self.pipeline_hdf5: str = ""
        #: Callback(source, hdf5) to remember a written file pipeline-wide.
        self.pipeline_sink: Callable[..., None] | None = None
        #: Inputs signature of the last successful compute (recompute dedup).
        self._last_signature: tuple | None = None
        #: Memoized per-frame movie stacks: {name: (signature, array)}.
        self._movie_cache: dict[str, tuple] = {}

    def apply_pipeline_context(self, payload: dict) -> None:
        """Adopt the shared pipeline context (current source + imaging HDF5).

        Lets any step (freely navigable) reuse the current image: the source
        TTTR is loaded if none is set here, and the remembered HDF5 becomes the
        default enrichment target for :meth:`add_to_hdf5`.
        """
        try:
            source = (payload or {}).get("source")
            hdf5 = (payload or {}).get("hdf5")
            if hdf5:
                self.pipeline_hdf5 = str(hdf5)
            if source and str(source) != self.filename:
                self.filename = str(source)
                self.notify("changed")
        except Exception:
            logger.debug("apply_pipeline_context failed", exc_info=True)

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored view.json."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(self._view_json)

    # ── observer hook ──
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("imaging observer failed", exc_info=True)

    def apply_setup_settings(self, payload: dict) -> None:
        """Adopt the step-0 detector windows (channels + micro-time ranges)."""
        try:
            from chisurf.core.fluorescence.imaging import windows_from_payload

            dets = windows_from_payload(payload)
            if dets:
                self.detectors = dets
                if self.display_window not in dets:
                    self.display_window = next(iter(dets))
                self.notify("setup")
        except Exception:
            logger.debug("apply_setup_settings failed", exc_info=True)

    def window_names(self) -> list[str]:
        """Return the detector-window names (fallback: a single channel-0 window)."""
        return list(self.detectors.keys()) or ["ch0"]

    def _windows(self) -> dict[str, dict]:
        """Return the windows to compute (setup detectors, or a ch-0 fallback)."""
        return self.detectors or {"ch0": {"chs": [0], "micro_time_ranges": []}}

    def refresh_display(self, value=None) -> None:
        """Re-render maps/plots for the selected display window.

        Accepts the committed value because the AutoForm ``choice`` section calls
        bound methods as ``fn(value)``; the value itself is already written to
        :attr:`display_window` by the section before this runs.
        """
        self.notify("run")

    def _default_hdf5_path(self) -> str:
        """Return the auto-HDF5 path derived from the source file (``*.imaging.h5``)."""
        if not self.filename:
            return ""
        return str(pathlib.Path(self.filename).with_suffix(".imaging.h5"))

    def flush_to_hdf5(self) -> None:
        """Persist this tool's columns to the imaging HDF5 (called on tool close).

        Writes to the pipeline-remembered file if set, else to a default derived
        from the source file, and remembers it pipeline-wide so later steps append
        to the same file. Deferred to close so the interactive session stays in
        memory (ndxplorer reads the shared DataFrame live).
        """
        if not self._columns:
            return
        path = self.pipeline_hdf5 or self._default_hdf5_path()
        if not path:
            return
        try:
            self._write_hdf5(path)
            self.pipeline_hdf5 = path
            if callable(self.pipeline_sink):
                self.pipeline_sink(source=self.filename or None, hdf5=path)
        except Exception:
            logger.debug("auto-persist to imaging HDF5 failed", exc_info=True)

    # ── AutoForm accessors ──
    def results_html(self) -> str:
        """Return the status/summary text for the info panel."""
        return f"<pre>{self.results_text}</pre>"

    def _disp(self, key: str):
        """Return the *key* map for the currently displayed window (or None)."""
        return (self._by_window.get(self.display_window) or {}).get(key)

    def intensity_map(self):
        """Return the summed-intensity 2-D map of the displayed window (or None)."""
        return self._disp("intensity")

    def frame_stack(self):
        """Return the raw per-frame intensity stack ``(n_frames, y, x)`` of the window.

        Display-only source for the image ``movie`` control. The stack is produced
        by the compute **worker process** (stashed under ``"frames"``) so the movie
        never rebuilds a CLSM on the UI thread; when it is missing (standalone use
        with no compute yet) it falls back to a lazily-built, memoized CLSM fill.
        """
        stashed = self._disp("frames")
        if stashed is not None:
            return stashed
        if not self.filename:
            return None
        win = self._windows().get(self.display_window)
        if win is None:
            return None
        chs = list(win.get("chs", [0]) or [0])
        mtr = list(win.get("micro_time_ranges") or [])
        sig = (self.filename, self.display_window, tuple(chs), tuple(tuple(r) for r in mtr))

        def build():
            from chisurf.core.fluorescence.imaging import cached_clsm

            clsm = cached_clsm(self.filename, chs, mtr)
            return np.asarray(clsm.get_intensity(), dtype=float)

        return self._cached_stack("frame_stack", sig, build)

    # ── compute (subclass) ──
    #: Window analysis kind dispatched to the core worker ("nb"/"phasor").
    WINDOW_KIND: str | None = None

    def _window_params(self) -> dict:
        """Extra worker params (phasor overrides with frequency/irf/n_ph_min)."""
        return {}

    def _summary(self, ny: int, nx: int) -> str:
        wins = ", ".join(self.window_names())
        return (
            f"{pathlib.Path(self.filename).name}: {nx}x{ny} px.\n"
            f"Windows: {wins}."
        )

    # ── recompute dedup ──
    def _extra_signature(self) -> tuple:
        """Tool-specific inputs that affect the result (subclasses extend)."""
        return ()

    def apply_calibration(self, calibration: dict) -> None:
        """Merge per-detector IRF/BG calibration into the detector windows.

        *calibration* maps detector name → ``{"irf": path, "bg": kHz}`` (from the
        skippable IRF & BG step). Flows into the compute (phasor IRF reference,
        intensity background subtraction).
        """
        if not calibration:
            return
        try:
            for name, vals in calibration.items():
                if name in self.detectors:
                    det = self.detectors[name]
                    irf = vals.get("irf")
                    det["irf"] = list(irf) if isinstance(irf, list) else ([irf] if irf else [])
                    det["bg_vv"] = float(vals.get("bg_vv") or 0.0)
                    det["bg_vh"] = float(vals.get("bg_vh") or 0.0)
                    # Total background rate for intensity count-rate subtraction.
                    det["bg"] = det["bg_vv"] + det["bg_vh"]
                    det["shift_vv"] = float(vals.get("shift_vv") or 0.0)
                    det["shift_vh"] = float(vals.get("shift_vh") or 0.0)
                    det["conv_start"] = int(vals.get("conv_start") or 0)
                    det["conv_stop"] = int(vals.get("conv_stop") or 0)
                    det["irf_start"] = int(vals.get("irf_start") or 0)
                    det["irf_stop"] = int(vals.get("irf_stop") or 0)
            self.notify("setup")
        except Exception:
            logger.debug("apply_calibration failed", exc_info=True)

    def _signature(self) -> tuple:
        """Hashable signature of all inputs that affect the computed maps."""
        dets = tuple(
            (
                name,
                tuple(d.get("chs", []) or []),
                tuple(d.get("ch_p", []) or []),
                tuple(d.get("ch_s", []) or []),
                tuple(tuple(r) for r in (d.get("micro_time_ranges") or [])),
                tuple(d.get("irf") or []),
                float(d.get("bg_vv") or 0.0),
                float(d.get("bg_vh") or 0.0),
                float(d.get("shift_vv") or 0.0),
                float(d.get("shift_vh") or 0.0),
                int(d.get("conv_start") or 0),
                int(d.get("conv_stop") or 0),
                int(d.get("irf_start") or 0),
                int(d.get("irf_stop") or 0),
            )
            for name, d in sorted(self.detectors.items())
        )
        return (self.filename, dets, *self._extra_signature())

    def needs_recompute(self) -> bool:
        """Return True when nothing is computed yet, or the inputs changed."""
        return not self._columns or self._signature() != self._last_signature

    # ── actions (toolbar) ──
    def run(self, progress=None) -> None:
        """Compute (if inputs changed) then notify. Recomputes only when needed.

        *progress* is an optional ``callback(fraction, text)``. This runs the
        compute inline; the GUI normally drives :meth:`compute` on a background
        thread and calls :meth:`notify` on the UI thread instead.
        """
        if not self.filename:
            self.results_text = "No file selected."
            self.notify("run")
            return
        if self.needs_recompute():
            self.compute(progress)
        self.notify("run")

    def compute(self, progress=None) -> bool:
        """Compute per-pixel maps for all windows. **Qt-free** (safe off the UI thread).

        Sets the maps/columns/signature but does NOT notify — the caller notifies
        on the UI thread. Returns True on success.
        """
        try:
            from chisurf.core.fluorescence.imaging import compute_windows

            results = compute_windows(
                self.filename, self._windows(), self.WINDOW_KIND,
                self._window_params(), progress=progress,
            )
            by_window: dict[str, dict[str, np.ndarray]] = {}
            columns: dict[str, np.ndarray] = {}
            shape = None
            for win, wm in results.items():
                by_window[win] = wm
                if "intensity" in wm:
                    shape = np.asarray(wm["intensity"]).shape
                for key, arr in wm.items():
                    # Display-only entries: 'intensity' (the Intensity tool owns
                    # the per-window count columns) and the per-frame movie stacks
                    # ('frames'/'g_frames'/'s_frames') — never written to the HDF5.
                    if key in _DISPLAY_ONLY_KEYS:
                        continue
                    columns[f"{key} ({win})"] = arr
        except Exception as exc:
            self.results_text = f"Computation failed: {exc}"
            return False
        self._last_signature = self._signature()
        self._by_window = by_window
        self._columns = columns
        if self.display_window not in by_window and by_window:
            self.display_window = next(iter(by_window))
        if shape is not None:
            self.results_text = self._summary(*shape)
        # Warm the (expensive) per-frame movie stacks HERE, on the background
        # compute thread, so the image docks' refresh() on the UI thread is a
        # cheap cache read instead of a synchronous per-frame recompute (stall).
        self._warm_movie_cache()
        return True

    # ── per-frame movie stacks (cached; built on the bg compute thread) ──
    def _cached_stack(self, name: str, sig: tuple, builder):
        """Return a memoized movie stack, rebuilding only when *sig* changes.

        Keeps refresh() (UI thread) O(1): the heavy build runs once (warmed by
        :meth:`_warm_movie_cache` on the background compute thread) and is reused
        until an input in *sig* changes.
        """
        ent = self._movie_cache.get(name)
        if ent is not None and ent[0] == sig:
            return ent[1]
        # Don't build on a cache miss until the maps exist: a pre-compute refresh
        # would otherwise trigger a synchronous CLSM fill on the UI thread. The
        # background compute warms these caches (it runs after ``_by_window`` is
        # set), so the miss-returns-None only bites the transient pre-Run refresh.
        if not self._by_window:
            return None
        try:
            arr = builder()
        except Exception:
            logger.debug("movie stack %r build failed", name, exc_info=True)
            arr = None
        self._movie_cache[name] = (sig, arr)
        return arr

    def _warm_movie_cache(self) -> None:
        """Pre-build this tool's movie stacks for the displayed window (bg thread).

        Base warms the raw-frame stack; subclasses extend for their own movies.
        Called from :meth:`compute` (off the UI thread) and best-effort.
        """
        try:
            self.frame_stack()
        except Exception:
            logger.debug("warm frame_stack failed", exc_info=True)

    def load_file(self, path: str) -> None:
        """Load *path* (from a file drop or picker) and run."""
        self.filename = str(path)
        self.notify("changed")
        self.run()

    def add_to_hdf5(self) -> None:
        """Add this tool's fields to the imaging HDF5.

        Defaults to the pipeline-remembered file when one exists (so successive
        steps enrich the same file without re-picking); otherwise asks for a path.
        The written path is remembered pipeline-wide via :attr:`pipeline_sink`.
        """
        if not self._columns:
            self.results_text = "Nothing to add — press Run first."
            self.notify("run")
            return
        path = self.pipeline_hdf5 or self._ask_hdf5_path()
        if not path:
            return
        try:
            added = self._write_hdf5(path)
            self.pipeline_hdf5 = path
            if callable(self.pipeline_sink):
                self.pipeline_sink(source=self.filename or None, hdf5=path)
            self.results_text = f"added {len(added)} column(s) → {pathlib.Path(path).name}"
        except Exception as exc:
            self.results_text = f"HDF5 write failed: {exc}"
        self.notify("run")

    def _write_hdf5(self, path: str) -> list[str]:
        """Merge this tool's per-window columns into the imaging HDF5 at *path*."""
        from chisurf.core.fluorescence.imaging import add_maps_to_hdf5

        return add_maps_to_hdf5(path, self._columns)

    def to_dataframe(self):
        """Return the per-pixel table as a DataFrame (shared with ndxplorer).

        This is the same in-memory table written to the HDF5; the host hands it
        straight to an ndxplorer ``DataSource`` (no file round-trip) so the tool
        and ndxplorer share one dataset and updates transfer dynamically.
        """
        if not self._columns:
            return None
        from chisurf.core.fluorescence.imaging import maps_to_dataframe

        return maps_to_dataframe(self._columns)

    def _ask_hdf5_path(self) -> str:
        try:
            from qtpy import QtWidgets

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                None, "Imaging HDF5", "", "HDF5 (*.h5 *.hdf5)"
            )
            return path
        except Exception:
            return ""


def build_ndx_data_source(df):
    """Build an ndxplorer ``DataSource`` from an in-memory per-pixel DataFrame."""
    from ndxplorer.core.data_source import DataSource

    return DataSource(parameter_names=list(df.columns), data=df)
