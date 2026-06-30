"""Qt-free view-model backing the PSF Determination tool.

:class:`PsfViewModel` holds all interactive state (the loaded bead stack, the
detected beads, the current selection and the last fit) and performs every
computation through the plugin's Qt-free :mod:`..api` layer. It carries the
attribute fields that AutoForm binds its setting controls to, the ``source``
methods the declarative plot/image sections read, and a small observer hook so
the GUI refreshes when state changes.

Deliberately free of Qt and pyqtgraph imports so it can be unit-tested headlessly
and so the architecture's GUI/logic boundary is respected (mirrors
:class:`chisurf.plugins.microscopy.clsm.gui.view_model.ClsmViewModel`).
"""

from __future__ import annotations

import csv
import logging
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np

from ..api import models as _models
from ..api import psf as _psf

_VIEW_JSON = pathlib.Path(__file__).parent / "psf.view.json"
logger = logging.getLogger(__name__)


class PsfViewModel:
    """State + logic for the interactive PSF Determination tool (no Qt)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``psf.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        defaults = _models.PsfSettings()
        # ── AutoForm-bound settings (value sections bind to these attrs) ──
        self.pixel_size_nm = defaults.pixel_size_nm
        self.z_step_nm = defaults.z_step_nm
        self.roi_xy = defaults.roi_xy
        self.roi_z = defaults.roi_z
        self.pixels_per_frame = defaults.pixels_per_frame
        self.min_distance = defaults.min_distance
        self.colormap = "magma"

        # ── runtime state ──────────────────────────────────────────────
        self.filename: str = ""
        self.stack: np.ndarray | None = None
        self.detected_beads: list[tuple[int, int, int]] = []
        #: ``(z, y, x)`` written by the image widget on click / by the bead spinner.
        self.selected_bead: tuple[int, int, int] | None = None
        self.results_text: str = "Load a stack, then detect or click a bead."

        # last single-bead fit (for the profile plots and the ROI overlay)
        self._fit_roi: np.ndarray | None = None
        self._fit_params: np.ndarray | None = None
        self._fit_circle: dict[str, float] | None = None

        self._observers: list[Callable[[str], None]] = []

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed (``"stack"``/``"beads"``/``"fit"``)."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                pass

    # ── AutoForm data accessors ────────────────────────────────────────
    def stack_image(self) -> np.ndarray | None:
        """Return the 3-D bead stack ``(z, y, x)`` for the image section."""
        return self.stack

    def detected_beads_zyx(self) -> list[tuple[int, int, int]]:
        """Return detected bead positions ``(z, y, x)`` for the image markers."""
        return self.detected_beads

    def fit_circle(self) -> dict[str, float] | None:
        """Return the fitted lateral-FWHM circle overlay descriptor, or ``None``."""
        return self._fit_circle

    def x_profile_series(self) -> list[dict[str, Any]]:
        """Return the x-profile data + fit overlay for the profile plot."""
        return self._profile_series(axis=2)

    def y_profile_series(self) -> list[dict[str, Any]]:
        """Return the y-profile data + fit overlay for the profile plot."""
        return self._profile_series(axis=1)

    def z_profile_series(self) -> list[dict[str, Any]]:
        """Return the z-profile data + fit overlay for the profile plot."""
        return self._profile_series(axis=0)

    def _profile_series(self, axis: int) -> list[dict[str, Any]]:
        """Build a data+fit series along *axis* (0=z, 1=y, 2=x) through the fit centre."""
        roi, params = self._fit_roi, self._fit_params
        if roi is None or params is None:
            return []
        z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset = params
        centres = (z_c, y_c, x_c)
        sigmas = (sigma_z, sigma_y, sigma_x)
        idx = [int(np.clip(round(c), 0, roi.shape[a] - 1)) for a, c in enumerate(centres)]
        n = roi.shape[axis]
        coord = np.arange(n)
        if axis == 0:
            data = roi[:, idx[1], idx[2]]
        elif axis == 1:
            data = roi[idx[0], :, idx[2]]
        else:
            data = roi[idx[0], idx[1], :]
        model = amplitude * np.exp(-0.5 * ((coord - centres[axis]) / sigmas[axis]) ** 2) + offset
        return [
            {
                "x": coord,
                "y": data,
                "name": "data",
                "color": "w",
                "symbol": "o",
                "symbol_size": 5,
                "no_line": True,
            },
            {"x": coord, "y": model, "name": "fit", "color": "r", "width": 2},
        ]

    # ── stack loading ──────────────────────────────────────────────────
    def load_stack(self, path: str) -> None:
        """Load a 3-D image stack from *path* and reset detection/fit state."""
        arr = _psf.load_stack(path)

        self.filename = path
        self.stack = arr
        self.detected_beads = []
        self.selected_bead = None
        self._fit_roi = self._fit_params = self._fit_circle = None
        nz, ny, nx = arr.shape
        self.results_text = (
            f"Loaded {pathlib.Path(path).name}: {nx}×{ny}×{nz} (x×y×z). "
            f"Detect beads or click one, then fit."
        )
        self.notify("stack")

    def set_stack(self, arr: np.ndarray) -> None:
        """Set the stack directly (for tests/scripts)."""
        self.stack = np.asarray(arr, dtype=np.float32)
        self.detected_beads = []
        self.selected_bead = None
        self._fit_roi = self._fit_params = self._fit_circle = None
        self.notify("stack")

    # ── detection / fitting ────────────────────────────────────────────
    def detect_beads(self) -> int:
        """Detect beads in the loaded stack; auto-select the first one."""
        if self.stack is None:
            return 0
        self.detected_beads = _psf.detect_beads(
            self.stack,
            roi_xy=int(self.roi_xy),
            roi_z=int(self.roi_z),
            pixels_per_frame=int(self.pixels_per_frame),
            min_distance=float(self.min_distance),
        )
        n = len(self.detected_beads)
        if n:
            self.selected_bead = self.detected_beads[0]
            self.results_text = f"Detected {n} bead(s). Select a bead index or click to fit."
        else:
            self.selected_bead = None
            self.results_text = "No beads detected. Adjust pixels/frame or min distance."
        self.notify("beads")
        return n

    def select_bead_index(self, idx: int) -> None:
        """Select the detected bead at *idx* and fit it."""
        if not (0 <= idx < len(self.detected_beads)):
            return
        self.selected_bead = self.detected_beads[idx]
        self.fit_selected()

    def on_pick(self) -> None:
        """Fit the bead the image widget just selected (the ``on_pick`` hook)."""
        self.fit_selected()

    def fit_selected(self) -> dict | None:
        """Fit a 3-D Gaussian to the currently selected bead and update results."""
        if self.stack is None or self.selected_bead is None:
            return None
        z0, y0, x0 = self.selected_bead
        roi, bounds = _psf.extract_roi(self.stack, z0, y0, x0, int(self.roi_xy), int(self.roi_z))
        if roi is None:
            self.results_text = (
                "Selected bead is too close to the stack boundary for the requested ROI size."
            )
            self._fit_roi = self._fit_params = self._fit_circle = None
            self.notify("fit")
            return None
        try:
            fit = _psf.fit_3d_gaussian(roi)
        except Exception as exc:  # pragma: no cover - solver-dependent
            self.results_text = f"3D Gaussian fit failed: {exc}"
            self._fit_roi = self._fit_params = self._fit_circle = None
            self.notify("fit")
            return None

        params = np.asarray(fit["params"], dtype=float)
        self._fit_roi = roi
        self._fit_params = params
        z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset = params

        pixel_nm = float(self.pixel_size_nm)
        z_step_nm = float(self.z_step_nm)
        fwhm_x_nm = 2.355 * sigma_x * pixel_nm
        fwhm_y_nm = 2.355 * sigma_y * pixel_nm
        fwhm_z_nm = 2.355 * sigma_z * z_step_nm
        sigma_xy = (sigma_x + sigma_y) / 2.0
        sigma_xy_nm = sigma_xy * pixel_nm
        sigma_z_nm = sigma_z * z_step_nm
        axial_ratio = sigma_z_nm / sigma_xy_nm if sigma_xy_nm > 0 else float("nan")

        _, _, y_min, _, x_min, _ = bounds
        abs_y, abs_x = y_min + y_c, x_min + x_c
        # Anchor the lateral-FWHM circle to the clicked bead's slice so it
        # co-locates with the red pick marker the user is looking at.
        self._fit_circle = {
            "x": abs_x,
            "y": abs_y,
            "r": 2.355 * sigma_xy / 2.0,
            "z": int(z0),
        }

        self.results_text = (
            "=== PSF Fit Results ===\n\n"
            f"Bead position: x={x0}, y={y0}, z={z0} (pixels)\n"
            f"ROI size: {int(self.roi_xy)}×{int(self.roi_xy)}×{int(self.roi_z)} (xy×z)\n\n"
            "--- Fitted (pixels) ---\n"
            f"Center: x={x_c:.2f}, y={y_c:.2f}, z={z_c:.2f}\n"
            f"Sigma:  σx={sigma_x:.2f}, σy={sigma_y:.2f}, σz={sigma_z:.2f}\n"
            f"Amplitude: {amplitude:.1f}   Offset: {offset:.1f}\n\n"
            "--- Physical (nm) ---\n"
            f"FWHMx: {fwhm_x_nm:.1f}   FWHMy: {fwhm_y_nm:.1f}   FWHMz: {fwhm_z_nm:.1f}\n"
            f"FWHMxy (avg): {(fwhm_x_nm + fwhm_y_nm) / 2:.1f}\n"
            f"σxy: {sigma_xy_nm:.1f}   σz: {sigma_z_nm:.1f}\n"
            f"Axial ratio (σz/σxy): {axial_ratio:.2f}\n\n"
            "--- Fit quality ---\n"
            f"Residual norm: {fit['cost']:.2e}   Success: {fit['success']}"
        )
        self.notify("fit")
        return fit

    def fit_all(self) -> list[dict]:
        """Fit every detected bead and summarize the results as text."""
        if self.stack is None or not self.detected_beads:
            self.results_text = "Detect beads first."
            self.notify("fit")
            return []
        results = _psf.fit_all_beads(
            self.stack,
            self.detected_beads,
            int(self.roi_xy),
            int(self.roi_z),
            float(self.pixel_size_nm),
            float(self.z_step_nm),
        )
        lines = ["=== Batch PSF fits ===", f"Detected beads: {len(results)}", ""]
        for f in results:
            if f.get("error"):
                lines.append(
                    f"[{f['index']:03d}] x={f['x_px']}, y={f['y_px']}, z={f['z_slice']}: {f['error']}"
                )
            else:
                lines.append(
                    f"[{f['index']:03d}] x={f['x_px']}, y={f['y_px']}, z={f['z_slice']} | "
                    f"σx={f['sigma_x_px']:.2f}, σy={f['sigma_y_px']:.2f}, σz={f['sigma_z_px']:.2f} px | "
                    f"FWHMxy≈{f['fwhm_xy_nm']:.1f} nm, FWHMz={f['fwhm_z_nm']:.1f} nm | "
                    f"axial={f['axial_ratio']:.2f} | ok={f['success']}"
                )
        self.results_text = "\n".join(lines)
        self.notify("fit")
        return results

    def export_csv(self, path: str) -> None:
        """Fit every detected bead and write the summary table to *path*."""
        results = (
            _psf.fit_all_beads(
                self.stack,
                self.detected_beads,
                int(self.roi_xy),
                int(self.roi_z),
                float(self.pixel_size_nm),
                float(self.z_step_nm),
            )
            if (self.stack is not None and self.detected_beads)
            else []
        )
        if not results:
            return
        keys = list(results[0].keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
