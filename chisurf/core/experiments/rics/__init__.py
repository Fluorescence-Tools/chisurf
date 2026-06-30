from __future__ import annotations

import pathlib

_VIEW_JSON = pathlib.Path(__file__).parent / "rics.view.json"

import pathlib

import numpy as np
import tttrlib

import chisurf.core.data
from chisurf.core.experiments.core.reader import ExperimentReader
from .data import RicsData, RicsSettings
from .ics_core import compute_rics_from_images
from .tttr_loader import load_clsm_from_tttr
from .masks import make_rect_mask, make_intensity_threshold_mask, combine_masks

try:
    import imageio.v2 as imageio  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    try:
        import imageio  # type: ignore[import]
    except Exception:  # pragma: no cover - optional dependency
        imageio = None


class RICSReader(ExperimentReader):

    name: str = "RICS (TTTR/ICS)"

    def __init__(
            self,
            name: str = "RICS (TTTR/ICS)",
            reading_routine: str | None = "PTU",
            channel: int = 0,
            pixel_duration: float | None = None,
            line_duration: float | None = None,
            x_range=None,
            y_range=None,
            subtract_average: str = "frame",
            frame_shift: int = 0,
            fftshift: bool = True,
            framewise_rics: bool = False,
            micro_time_ranges=None,
            *args,
            **kwargs
    ):
        """Initialize a RICS reader.

        Parameters
        ----------
        name : str
            Human-readable reader name.
        reading_routine : str or None
            tttrlib reading routine (e.g. ``'PTU'``).
        channel : int
            Default routing channel index.
        pixel_duration : float, optional
            Pixel dwell time in microseconds.
        line_duration : float, optional
            Line duration in milliseconds.
        x_range : tuple of int, optional
            ROI x-range ``(start, stop)``.
        y_range : tuple of int, optional
            ROI y-range ``(start, stop)``.
        subtract_average : str
            Background subtraction mode (``'frame'``, ``'stack'``, or ``''``).
        frame_shift : int
            Frame shift for cross-correlation.
        fftshift : bool
            Whether to center the zero-lag pixel in the ICS map.
        framewise_rics : bool
            Whether to compute RICS per frame.
        micro_time_ranges : list, optional
            Micro-time ranges for photon selection.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.reading_routine = reading_routine
        self.channel = int(channel)
        # Optional per-dataset imaging timing parameters. When loading PTU
        # data we attempt to infer these from the TTTR header; otherwise
        # they may be provided by the GUI/controller and are propagated
        # into the RICS metadata for model construction.
        self.pixel_duration = pixel_duration
        self.line_duration = line_duration
        # Optional list of TTTR routing channels defining this logical
        # tttr_channeldefinition. When present, the TTTR reader will use all of these
        # channels instead of the single "channel" index. This attribute
        # is typically populated from the GUI (e.g. tttr_channeldefinition setups).
        if not hasattr(self, "channel_numbers"):
            self.channel_numbers = None
        self.micro_time_ranges = micro_time_ranges
        self.x_range = x_range
        self.y_range = y_range
        self.subtract_average = subtract_average
        self.frame_shift = int(frame_shift)
        self.fftshift = bool(fftshift)
        self.framewise_rics = bool(framewise_rics)

        # Optional internal cache to reuse the most recently computed
        # intensity/ICS results for previews and data loading. The cache is
        # keyed by the full filename used in :meth:`read`.
        self._cache_filename: str | None = None
        self._cache_images = None
        self._cache_intensity_stack = None
        self._cache_intensity_mean = None
        self._cache_ics_stack = None
        # Channels used for the cached TTTR/ICS computation. This is a
        # normalized, sorted tuple of routing channel indices.
        self._cache_channels = None
        self._cache_micro_time_ranges = None

    def autofitrange(self, data, **kwargs):
        """Return the full data range as the default fit interval.

        Parameters
        ----------
        data : chisurf.core.base.Data
            The experimental RICS data.

        Returns
        -------
        tuple of int
            ``(0, len(y))`` on success, ``(0, 0)`` on failure.
        """
        try:
            y = data.y
            return 0, len(y)
        except Exception:
            return 0, 0

    @property
    def pixel_duration_us(self) -> float:
        """Pixel dwell time in µs; 0.0 means auto-detect from TTTR header."""
        v = self.pixel_duration
        return float(v) if v is not None else 0.0

    @pixel_duration_us.setter
    def pixel_duration_us(self, value: float) -> None:
        self.pixel_duration = float(value) if float(value) > 0.0 else None

    @property
    def line_duration_ms(self) -> float:
        """Line scan time in ms; 0.0 means auto-detect from TTTR header."""
        v = self.line_duration
        return float(v) if v is not None else 0.0

    @line_duration_ms.setter
    def line_duration_ms(self, value: float) -> None:
        self.line_duration = float(value) if float(value) > 0.0 else None

    def view_spec(self):
        """Return the declarative editor spec for RICS reader settings."""
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_VIEW_JSON)

    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.core.data.ExperimentDataCurveGroup:
        """Read a TTTR or TIFF file and return a RICS curve group.

        Parameters
        ----------
        filename : str, optional
            Path to the TTTR file or TIFF stack.

        Returns
        -------
        chisurf.core.data.ExperimentDataCurveGroup
            Group containing the ICS mean curve with spatial shift metadata.
        """
        group = chisurf.core.data.ExperimentDataCurveGroup([])
        if filename is None:
            return group
        if isinstance(filename, (list, tuple)):
            if not filename:
                return group
            filename = filename[0]
        fn = pathlib.Path(filename)
        if not fn.is_file():
            return group
        suffix = fn.suffix.lower()

        # Initial guess for pixel/line durations (µs / ms) from the current
        # setup. For TTTR files we may refine these using the TTTR header.
        pixel_duration_us = getattr(self, "pixel_duration", None)
        line_duration_ms = getattr(self, "line_duration", None)

        # Normalize current routing channel selection for cache key.
        chs = getattr(self, "channel_numbers", None)
        if chs is None:
            chs = [int(getattr(self, "channel", 0) or 0)]
        try:
            current_channels = tuple(sorted({int(c) for c in chs}))
        except Exception:
            current_channels = (int(getattr(self, "channel", 0) or 0),)

        mtr_value = getattr(self, "micro_time_ranges", None)
        mtr_norm = None
        if isinstance(mtr_value, (list, tuple)):
            tmp = []
            for r in mtr_value:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    try:
                        a = int(r[0])
                        b = int(r[1])
                    except Exception:
                        continue
                    tmp.append((a, b))
            if tmp:
                mtr_norm = tuple(tmp)

        # Attempt to reuse cached results when possible to avoid recomputing
        # ICS for the same file and selection repeatedly (e.g. during
        # preview + final load).
        use_cache = (
            self._cache_filename is not None
            and str(fn) == self._cache_filename
            and self._cache_ics_stack is not None
            and getattr(self, "_cache_channels", None) == current_channels
            and getattr(self, "_cache_micro_time_ranges", None) == mtr_norm
        )

        if use_cache:
            images = np.asarray(self._cache_images, dtype=float)
            intensity_stack = (
                None if self._cache_intensity_stack is None
                else np.asarray(self._cache_intensity_stack, dtype=float)
            )
            intensity_mean = (
                None if self._cache_intensity_mean is None
                else np.asarray(self._cache_intensity_mean, dtype=float)
            )
            ics_stack = np.asarray(self._cache_ics_stack, dtype=float)
            clsm = None
        else:
            images = None
            clsm = None

            # Branch on file type: TIFF stack vs TTTR container
            if suffix in (".tif", ".tiff") and imageio is not None:
                data = imageio.imread(fn.as_posix())
                arr = np.asarray(data)
                # Normalize to a stack of shape (n_frames, ny, nx)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    # Color image (ny, nx, channels)
                    ch = int(getattr(self, "channel", 0) or 0)
                    ch = max(0, min(ch, arr.shape[-1] - 1))
                    arr = arr[..., ch]
                    arr = arr[None, ...]
                elif arr.ndim == 3:
                    # Assume (n_frames, ny, nx)
                    pass
                elif arr.ndim == 4:
                    # Assume (n_frames, ny, nx, channels)
                    ch = int(getattr(self, "channel", 0) or 0)
                    ch = max(0, min(ch, arr.shape[-1] - 1))
                    arr = arr[..., ch]
                else:
                    return group
                images = np.asarray(arr, dtype=float)
            else:
                # Fall back to TTTR-based CLSM reconstruction. Respect
                # optional channel_numbers (routing channels) if present
                # so that logical detectors (e.g. parallel + perpendicular
                # in one color) can be represented by multiple TTTR
                # channels. We pass the full TTTR object together with a
                # channels list to CLSMImage rather than pre-filtering the
                # TTTR ourselves, which is the pattern used throughout the
                # tttrlib examples and avoids empty TTTR selections.
                if self.reading_routine:
                    tttr_all = self._open_tttr(fn.as_posix(), self.reading_routine)
                else:
                    tttr_all = self._open_tttr(fn.as_posix())

                # Attempt to estimate pixel and line durations from the
                # TTTR header. For PTU files this uses the $TimePerPixel
                # tag and the global macro-time resolution. Values are
                # converted to µs (pixel) and ms (line) for use by the
                # analytic RICS models.
                try:
                    hdr = getattr(tttr_all, "header", None)
                except Exception:
                    hdr = None
                if hdr is not None:
                    try:
                        macro_res = float(getattr(hdr, "macro_time_resolution", 0.0) or 0.0)
                    except Exception:
                        macro_res = 0.0
                    try:
                        pd_clk = hdr.get_pixel_duration()
                    except Exception:
                        pd_clk = None
                    try:
                        ld_clk = hdr.get_line_duration()
                    except Exception:
                        ld_clk = None
                    if macro_res > 0.0 and isinstance(pd_clk, (int, float)) and pd_clk > 0:
                        pixel_duration_us = float(pd_clk) * macro_res * 1.0e6
                    if macro_res > 0.0 and isinstance(ld_clk, (int, float)) and ld_clk > 0:
                        line_duration_ms = float(ld_clk) * macro_res * 1.0e3
                try:
                    if pixel_duration_us is not None:
                        self.pixel_duration = float(pixel_duration_us)
                    if line_duration_ms is not None:
                        self.line_duration = float(line_duration_ms)
                except Exception:
                    pass

                ch_list = list(current_channels) if current_channels else [int(getattr(self, "channel", 0) or 0)]

                mtr_arg = None
                if mtr_norm is not None:
                    try:
                        mtr_arg = [(int(a), int(b)) for (a, b) in mtr_norm]
                    except Exception:
                        mtr_arg = None

                if mtr_arg:
                    clsm = tttrlib.CLSMImage(
                        tttr_data=tttr_all,
                        channels=ch_list,
                        micro_time_ranges=mtr_arg,
                        fill=True
                    )
                else:
                    clsm = tttrlib.CLSMImage(
                        tttr_data=tttr_all,
                        channels=ch_list,
                        fill=True
                    )
                images = np.asarray(clsm.intensity, dtype=float)
                if images.ndim == 2:
                    images = images[None, ...]

        n_frames = images.shape[0]
        x_range = getattr(self, "x_range", None)
        if not isinstance(x_range, (list, tuple)) or len(x_range) < 2:
            x_range = [0, -1]
        else:
            x_range = [int(x_range[0]), int(x_range[1])]
        y_range = getattr(self, "y_range", None)
        if not isinstance(y_range, (list, tuple)) or len(y_range) < 2:
            y_range = [0, -1]
        else:
            y_range = [int(y_range[0]), int(y_range[1])]
        subtract_average = getattr(self, "subtract_average", "frame") or ""
        if subtract_average not in ("frame", "stack", ""):
            subtract_average = "frame"
        frame_shift = int(getattr(self, "frame_shift", 0) or 0)
        frames_index_pairs = None
        if n_frames > 0 and frame_shift != 0:
            frames = np.arange(n_frames, dtype=int)
            frames_shifted = np.roll(frames, frame_shift)
            frames_index_pairs = list(zip(frames.tolist(), frames_shifted.tolist()))
        if not use_cache:
            # Determine ROI extents in the full field of view for metadata,
            # but keep the stored intensity stack as the full images so that
            # the intensity preview always shows the entire frame regardless
            # of the RICS ROI.
            ny_full = int(images.shape[1]) if images.ndim >= 3 else 0
            nx_full = int(images.shape[2]) if images.ndim >= 3 else 0
            x0_roi, x1_roi = int(x_range[0]), int(x_range[1])
            y0_roi, y1_roi = int(y_range[0]), int(y_range[1])
            if x1_roi < 0 or x1_roi > nx_full:
                x1_roi = nx_full
            if y1_roi < 0 or y1_roi > ny_full:
                y1_roi = ny_full
            if x0_roi < 0:
                x0_roi = 0
            if y0_roi < 0:
                y0_roi = 0
            if x1_roi <= x0_roi or y1_roi <= y0_roi or nx_full == 0 or ny_full == 0:
                # Fallback to full field of view
                x0_roi, y0_roi = 0, 0
                x1_roi, y1_roi = nx_full, ny_full

            # Full-field intensity stack for preview (no cropping by ROI).
            try:
                intensity_stack = images
            except Exception:
                intensity_stack = None
            if intensity_stack is not None and intensity_stack.ndim == 3 and intensity_stack.size > 0:
                try:
                    intensity_mean = intensity_stack.mean(axis=0)
                except Exception:
                    intensity_mean = None
            else:
                intensity_mean = None

            ics_kwargs = dict(
                images=images,
                x_range=x_range,
                y_range=y_range,
                subtract_average=subtract_average,
            )
            if frames_index_pairs is not None:
                ics_kwargs["frames_index_pairs"] = frames_index_pairs
            ics = tttrlib.CLSMImage.compute_ics(**ics_kwargs)
            ics_stack = np.asarray(ics, dtype=float)

            # ------------------------------------------------------------------
            # Normalization
            # ------------------------------------------------------------------
            # In the tttrlib examples, a reference numpy implementation
            # computes the ICS as
            #
            #   ics = ifft2(fft(deltaI) * conj(fft(deltaI))).real
            #         / (mean(I)**2 * N)
            #
            # with N = n_pixels in the ROI. Some tttrlib builds already
            # apply this normalization internally, but others may return an
            # unnormalized correlation (values >> 1). To guard against
            # missing normalization, we heuristically detect "too large"
            # ICS magnitudes and rescale by <I>^2 * N on a per-frame basis.
            try:
                finite = np.isfinite(ics_stack)
                if np.any(finite):
                    median_abs = float(np.median(np.abs(ics_stack[finite])))
                else:
                    median_abs = 0.0
            except Exception:
                median_abs = 0.0

            if median_abs > 1.0:
                try:
                    # Use the ROI region for the mean intensity, falling
                    # back to the full frame if necessary.
                    roi_images = images[:, y0_roi:y1_roi, x0_roi:x1_roi]
                    if roi_images.ndim != 3 or roi_images.size == 0:
                        roi_images = images
                    n_frames_norm, ny_roi, nx_roi = roi_images.shape
                    if n_frames_norm > 0 and ny_roi > 0 and nx_roi > 0:
                        N_roi = float(ny_roi * nx_roi)
                        means = roi_images.reshape(n_frames_norm, -1).mean(axis=1)
                        norm = (means ** 2) * N_roi
                        # Avoid division by zero or negative factors
                        norm[norm <= 0] = 1.0
                        ics_stack = ics_stack / norm[:, None, None]
                except Exception:
                    pass

            # Update cache for subsequent calls with the same
            # filename/channel selection.
            self._cache_filename = str(fn)
            self._cache_images = images
            self._cache_intensity_stack = intensity_stack
            self._cache_intensity_mean = intensity_mean
            self._cache_ics_stack = ics_stack
            self._cache_channels = current_channels
            self._cache_micro_time_ranges = mtr_norm
        else:
            # When reusing cache, we already have intensity_stack/intensity_mean
            intensity_stack = self._cache_intensity_stack
            intensity_mean = self._cache_intensity_mean
        if ics_stack.ndim == 2:
            ics_stack = ics_stack[None, ...]
        n_ics = ics_stack.shape[0]

        # Optionally center zero lag by fftshifting the entire ICS stack.
        # Shifting the stack and then taking the mean is equivalent to
        # shifting the mean alone, but additionally ensures that any
        # framewise views (which use ics_stack) are also centered.
        use_fftshift = bool(getattr(self, "fftshift", True))
        if use_fftshift:
            try:
                ics_stack = np.fft.fftshift(ics_stack, axes=(-2, -1))
            except Exception:
                pass

        ics_mean = ics_stack.mean(axis=0)
        ics_std = ics_stack.std(axis=0)
        if n_ics > 0:
            ics_std = ics_std / np.sqrt(float(n_ics))
        ny, nx = ics_mean.shape
        line_shift, pixel_shift = np.indices((ny, nx))
        line_shift = line_shift - ny // 2
        pixel_shift = pixel_shift - nx // 2
        y = ics_mean.ravel()
        ey = ics_std.ravel()
        x = np.arange(y.size, dtype=float)
        if ey.size == x.size:
            ey = np.where(ey > 0, ey, 1.0)
        else:
            ey = np.ones_like(y)
        clsm_shape = tuple(getattr(clsm, "shape", images.shape))
        meta_rics = {
            "ics_mean": ics_mean,
            "ics_std": ics_std,
            "ics_stack": ics_stack,
            "line_shift": line_shift,
            "pixel_shift": pixel_shift,
            "filename": str(fn),
            "clsm_shape": clsm_shape,
            "n_frames": int(n_frames),
            "x_range": tuple(x_range),
            "y_range": tuple(y_range),
            "subtract_average": subtract_average,
            "frame_shift": int(frame_shift),
            "fftshift": bool(use_fftshift),
            "framewise_rics": bool(getattr(self, "framewise_rics", False)),
            "intensity_stack": intensity_stack,
            "intensity_mean": intensity_mean,
            "intensity_roi": (y0_roi, y1_roi, x0_roi, x1_roi),
            "micro_time_ranges": mtr_norm,
            "pixel_duration_us": float(pixel_duration_us) if pixel_duration_us is not None else None,
            "line_duration_ms": float(line_duration_ms) if line_duration_ms is not None else None,
        }

        # Generic grid metadata describing the dimensionality and memory
        # layout of the ICS maps used for fitting. This is intentionally
        # experiment-agnostic so that GUI components can reason about 2D
        # data without referring to RICS explicitly.
        grid_meta = {
            "ndim": 2,
            "shape": (int(ny), int(nx)),
            # NumPy-style order string: 'C' -> row-major, 'F' -> column-major
            "order": "C",
            # Total number of 1D points in the flattened representation
            "size": int(y.size),
        }

        meta_all = {
            "rics": meta_rics,
            "grid": grid_meta,
        }

        data = chisurf.core.data.DataCurve(
            name=fn.stem,
            x=x,
            y=y,
            ey=ey,
            filename=str(fn),
            data_reader=self,
            meta_data=meta_all,
            load_filename_on_init=False
        )
        group.append(data)
        group.data_reader = self
        return group
