"""Qt-free view-model backing the CLSM tool.

:class:`ClsmViewModel` holds all interactive state (TTTR data, CLSM images,
representations, the pixel selection, ROIs and saved decays) and performs every
computation through the plugin ``core`` layer. It carries the attribute groups
that AutoForm binds its setting fields to, the ``source`` methods the declarative
plot sections read, and a tiny observer hook so the custom widgets refresh when
state changes.

Deliberately free of Qt and pyqtgraph imports so it can be unit-tested headlessly
and so the architecture's GUI/logic boundary is respected.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np

from ..api.models import ClsmSetup
from ..core import frc as frc_mod
from ..core import imaging, setups

_VIEW_JSON = pathlib.Path(__file__).parent / "clsm.view.json"


class _Group:
    """A plain attribute bag used as an AutoForm binding target."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _parse_int_list(text: str) -> list[int]:
    return [int(p.strip()) for p in str(text).split(",") if p.strip()]


class ClsmViewModel:
    """State + logic for the interactive CLSM tool (no Qt)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``clsm.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        self._presets = setups.builtin_setups()
        first = next(iter(self._presets))

        # ── AutoForm-bound setting groups ──────────────────────────────
        preset = self._presets[first]
        self.setup = _Group(
            setup_name=first,
            tttr_type=preset.get("tttr_type", "PTU"),
            routine=preset.get("routine", "default"),
            frame_marker_text=",".join(str(m) for m in preset.get("frame_marker", [])),
            line_start_marker=int(preset.get("line_start_marker", 1)),
            line_stop_marker=int(preset.get("line_stop_marker", 2)),
            event_type_marker=int(preset.get("event_type_marker", 1)),
            pixel_per_line=int(preset.get("pixel_per_line", 0) or 0),
            channels_text="0",
        )
        self.brush = _Group(size=7, width=3.0, mode="select", live_update=True)
        self.colormap = "magma"
        self.decay = _Group(
            image_type="Intensity",
            n_ph_min=1,
            tac_coarsening=1,
            frame_mode="sum",
            frame_idx=0,
        )

        # ── runtime state ──────────────────────────────────────────────
        self.filename: str = ""
        self.tttr_data: Any = None
        self.clsm_images: dict[str, Any] = {}
        self.representations: dict[str, np.ndarray] = {}
        self.current_clsm_name: str = ""
        self.current_representation_name: str = ""
        self.current_image: np.ndarray | None = None
        self._subset_1: np.ndarray | None = None
        self._subset_2: np.ndarray | None = None
        self.selection_mask: np.ndarray | None = None
        self.rois: dict[str, np.ndarray] = {}
        self.curves: list[dict[str, Any]] = []
        self.current_decay: dict[str, Any] | None = None

        self._observers: list[Callable[[str], None]] = []

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed (e.g. ``"image"``/``"decay"``)."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                pass

    # ── setup helpers ──────────────────────────────────────────────────
    @property
    def setup_names(self) -> list[str]:
        """Names of the available setup presets."""
        return list(self._presets)

    def clsm_image_names(self) -> list[str]:
        """Names of the built CLSM images (drives the CLSM selector combo)."""
        return list(self.clsm_images)

    def representation_names(self) -> list[str]:
        """Names of the image representations (drives the Image selector combo)."""
        return list(self.representations)

    def apply_preset(self, name: str) -> None:
        """Copy a named preset into the bound ``setup`` group."""
        preset = self._presets.get(name)
        if not preset:
            return
        self.setup.setup_name = name
        self.setup.tttr_type = preset.get("tttr_type", "PTU")
        self.setup.routine = preset.get("routine", "default")
        self.setup.frame_marker_text = ",".join(str(m) for m in preset.get("frame_marker", []))
        self.setup.line_start_marker = int(preset.get("line_start_marker", 1))
        self.setup.line_stop_marker = int(preset.get("line_stop_marker", 2))
        self.setup.event_type_marker = int(preset.get("event_type_marker", 1))
        self.setup.pixel_per_line = int(preset.get("pixel_per_line", 0) or 0)
        self.notify("setup")

    def build_setup(self) -> ClsmSetup:
        """Build a :class:`ClsmSetup` from the current bound setting groups."""
        s = self.setup
        return ClsmSetup(
            tttr_type=s.tttr_type,
            routine=s.routine,
            frame_marker=_parse_int_list(s.frame_marker_text),
            line_start_marker=int(s.line_start_marker),
            line_stop_marker=int(s.line_stop_marker),
            event_type_marker=int(s.event_type_marker),
            pixel_per_line=int(s.pixel_per_line),
            channels=_parse_int_list(s.channels_text),
        )

    # ── data operations (delegate to core) ─────────────────────────────
    def load_file(self, filename: str) -> None:
        """Load a TTTR file, or an imaging HDF5 resolved to its source photon data.

        An imaging HDF5 (``.h5`` / ``.hdf5``) written by the imaging tools carries
        a back-reference to the original TTTR file; CLSM Draw follows it so pixel
        selections can still extract decays from the raw photons.
        """
        p = str(filename)
        if p.lower().endswith((".h5", ".hdf5")):
            from chisurf.core.fluorescence.imaging import read_imaging_source

            source = read_imaging_source(p)
            if not source:
                raise RuntimeError(
                    f"Imaging HDF5 has no source-TTTR back-reference: {p}"
                )
            self.load_tttr(source)
        else:
            self.load_tttr(p)

    def load_tttr(self, filename: str) -> None:
        """Load a TTTR file and auto-fill markers detected from its header."""
        import tttrlib

        self.filename = filename
        self.tttr_data = tttrlib.TTTR(filename, self.setup.tttr_type)
        detected = setups.read_clsm_markers(self.tttr_data)
        if detected:
            if detected.get("frame_marker"):
                self.setup.frame_marker_text = ",".join(str(m) for m in detected["frame_marker"])
            for key in (
                "line_start_marker",
                "line_stop_marker",
                "event_type_marker",
                "pixel_per_line",
            ):
                if detected.get(key) is not None:
                    setattr(self.setup, key, int(detected[key]))
        self.notify("setup")

    def add_clsm(self) -> str:
        """Build a CLSM image from the current setup/channels; returns its name."""
        if self.tttr_data is None:
            raise RuntimeError("Load a TTTR file first")
        setup = self.build_setup()
        clsm_image = imaging.build_clsm_image(self.tttr_data, setup)
        stem = pathlib.Path(self.filename).stem or "clsm"
        name = f"{stem}_ch({self.setup.channels_text})"
        self.clsm_images[name] = clsm_image
        self.current_clsm_name = name
        self.notify("clsm")
        return name

    def remove_clsm(self, name: str) -> None:
        """Remove a CLSM image by name."""
        self.clsm_images.pop(name, None)
        if self.current_clsm_name == name:
            self.current_clsm_name = next(iter(self.clsm_images), "")
        self.notify("clsm")

    def add_representation(self) -> str:
        """Compute the current image representation; returns its name."""
        clsm_image = self.clsm_images.get(self.current_clsm_name)
        if clsm_image is None:
            raise RuntimeError("Create a CLSM image first")
        image_type = self.decay.image_type
        data = imaging.representation(
            clsm_image, self.tttr_data, image_type, int(self.decay.n_ph_min)
        )
        name = f"{self.current_clsm_name}_{image_type}"
        self.representations[name] = data
        self.select_representation(name)
        return name

    def remove_representation(self, name: str) -> None:
        """Remove an image representation by name; select another if available."""
        self.representations.pop(name, None)
        if self.current_representation_name == name:
            remaining = next(iter(self.representations), "")
            if remaining:
                self.select_representation(remaining)
            else:
                self.current_representation_name = ""
                self.current_image = None
                self.notify("image")

    @property
    def n_frames(self) -> int:
        """Number of frames in the current representation (0 if none)."""
        img = self.representations.get(self.current_representation_name)
        return int(img.shape[0]) if img is not None else 0

    def select_representation(self, name: str) -> None:
        """Make *name* the displayed representation and reduce its frames."""
        image = self.representations.get(name)
        if image is None:
            return
        self.current_representation_name = name
        current, s1, s2 = imaging.reduce_frames(
            image, self.decay.frame_mode, int(self.decay.frame_idx)
        )
        self.current_image = current
        self._subset_1, self._subset_2 = s1, s2
        self.selection_mask = np.zeros_like(current)
        self.notify("image")

    def refresh_current_image(self) -> None:
        """Re-reduce the active representation (after a frame-mode change)."""
        if self.current_representation_name:
            self.select_representation(self.current_representation_name)

    # ── brush ──────────────────────────────────────────────────────────
    @property
    def live_update(self) -> bool:
        """Whether the decay recomputes live while brushing."""
        return bool(self.brush.live_update)

    def brush_kernel(self) -> np.ndarray:
        """Return the current paint kernel for the image-canvas brush."""
        b = self.brush
        return imaging.brush_kernel(int(b.size), float(b.width), select=(b.mode == "select"))

    # ── selection / decay ──────────────────────────────────────────────
    def clear_selection(self) -> None:
        """Reset the pixel selection mask to empty."""
        if self.current_image is not None:
            self.selection_mask = np.zeros_like(self.current_image)
        self.notify("selection")

    def recompute_decay(self) -> dict[str, Any] | None:
        """Recompute the decay of the current pixel selection."""
        clsm_image = self.clsm_images.get(self.current_clsm_name)
        if clsm_image is None or self.selection_mask is None:
            return None
        stack = self.decay.frame_mode in ("sum", "mean")
        t, y, ey = imaging.decay_of_selection(
            clsm_image,
            self.tttr_data,
            self.selection_mask,
            tac_coarsening=int(self.decay.tac_coarsening),
            stack_frames=stack,
            frame_idx=int(self.decay.frame_idx),
        )
        self.current_decay = {"time_ns": t, "counts": y, "noise": ey}
        self.notify("decay")
        return self.current_decay

    def add_decay_curve(self, name: str | None = None) -> str:
        """Persist the current decay as a named curve; returns its name."""
        if self.current_decay is None:
            return ""
        roi = self.current_representation_name or "selection"
        curve = dict(self.current_decay)
        curve["name"] = name or f"{self.current_clsm_name}_ROI({roi})"
        self.curves.append(curve)
        self.notify("decay")
        return curve["name"]

    def to_chisurf_dataset(self) -> bool:
        """Export the current decay as a ChiSurf TCSPC dataset (best-effort)."""
        decay = self.recompute_decay()
        if not decay:
            return False
        name = self.add_decay_curve()
        try:
            import chisurf as cs
            from chisurf.core.data import DataCurve
            from chisurf.macros import core_data

            try:
                experiment = cs.experiment["TCSPC"]
            except Exception:
                experiment = None
            curve = DataCurve(
                x=np.asarray(decay["time_ns"]),
                y=np.asarray(decay["counts"]),
                ey=np.asarray(decay["noise"]),
                experiment=experiment,
                load_filename_on_init=False,
            )
            curve.name = name
            core_data.add_dataset(dataset=curve)
            return True
        except Exception:
            return False

    # ── ROI management ─────────────────────────────────────────────────
    def add_roi(self, name: str) -> None:
        """Save the current selection mask as a named ROI."""
        if self.selection_mask is not None:
            self.rois[name] = np.copy(self.selection_mask)
            self.notify("roi")

    def apply_roi(self, name: str) -> None:
        """Make a saved ROI the current selection and recompute the decay."""
        roi = self.rois.get(name)
        if roi is not None and self.current_image is not None:
            self.selection_mask = np.copy(roi)
            self.recompute_decay()
            self.notify("selection")

    def remove_roi(self, name: str) -> None:
        """Remove a saved ROI by name."""
        self.rois.pop(name, None)
        self.notify("roi")

    def save_roi(self, name: str, filename: str) -> None:
        """Write a saved ROI mask to an image file."""
        import skimage as ski

        roi = self.rois.get(name)
        if roi is None:
            return
        image = np.copy(roi)
        image[image > 0] = 255
        ski.io.imsave(filename, image.astype(np.uint8))

    def load_roi(self, filename: str, name: str | None = None) -> None:
        """Load an ROI mask from an image file."""
        import skimage as ski

        image = np.asarray(ski.io.imread(filename))
        if image.ndim == 3:
            image = image[0]
        roi_name = name or pathlib.Path(filename).stem
        self.rois[roi_name] = image
        self.notify("roi")

    # ── plot sources (read by declarative PlotSection widgets) ─────────
    def decay_series(self) -> list[dict[str, Any]]:
        """Series for the decay plot: saved curves plus the current selection."""
        series: list[dict[str, Any]] = []
        for i, c in enumerate(self.curves):
            series.append(
                {
                    "x": np.asarray(c["time_ns"]),
                    "y": np.asarray(c["counts"]),
                    "name": c.get("name", f"curve {i}"),
                }
            )
        if self.current_decay is not None:
            series.append(
                {
                    "x": np.asarray(self.current_decay["time_ns"]),
                    "y": np.asarray(self.current_decay["counts"]),
                    "name": "Current selection",
                    "width": 2,
                }
            )
        return series

    def frc_series(self) -> list[dict[str, Any]]:
        """Series for the FRC plot of the current representation."""
        if self._subset_1 is None or self._subset_2 is None:
            return []
        density, bins = frc_mod.compute_frc(self._subset_1, self._subset_2)
        return [{"x": bins, "y": np.nan_to_num(density), "name": "FRC"}]
