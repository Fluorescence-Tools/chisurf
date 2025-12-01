from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import pathlib

import numpy as np
import tttrlib

import chisurf.data
from chisurf.experiments import reader


class PCHReader(reader.ExperimentReader):
    """Experiment reader for photon counting histograms (PCH).

    This reader turns TTTR containers (e.g. PTU/HT3) into a 1D P(k) dataset
    suitable for PCH fitting. The underlying TTTR data is binned in time to an
    intensity trace; the photon-count histogram of that trace is then stored as
    experimental P(k) together with rich metadata.
    """

    name: str = "PCH (TTTR)"

    def __init__(
        self,
        name: str = "PCH (TTTR)",
        reading_routine: str | None = "PTU",
        channels: Optional[Sequence[int]] = None,
        channel: int = 0,
        bin_time_us: float = 100.0,
        micro_time_range: Optional[Tuple[int, int]] = (0, 65535),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.reading_routine = reading_routine
        # Optional list of routing channels defining the logical detector. If
        # not provided, fall back to a single channel index.
        self.channel = int(channel)
        self.channel_numbers: Optional[Sequence[int]] = channels
        self.bin_time_us = float(bin_time_us)
        self.micro_time_range: Optional[Tuple[int, int]] = (
            int(micro_time_range[0]),
            int(micro_time_range[1]),
        ) if isinstance(micro_time_range, (tuple, list)) and len(micro_time_range) >= 2 else None

    def autofitrange(self, data, **kwargs) -> Tuple[int, int]:  # type: ignore[override]
        try:
            y = data.y
            return 0, len(y)
        except Exception:
            return 0, 0

    def _get_channels(self) -> Tuple[int, ...]:
        chs = self.channel_numbers
        if chs is None:
            chs = [self.channel]
        try:
            return tuple(sorted({int(c) for c in chs}))
        except Exception:
            return (int(self.channel),)

    def _get_micro_time_range(self) -> Optional[Tuple[int, int]]:
        mtr = getattr(self, "micro_time_range", None)
        if isinstance(mtr, (tuple, list)) and len(mtr) >= 2:
            try:
                a = int(mtr[0])
                b = int(mtr[1])
                return a, b
            except Exception:
                return None
        return None

    def _compute_trace_and_histogram(
        self,
        tttr: "tttrlib.TTTR",
        channels: Sequence[int],
        bin_time_us: float,
        micro_time_range: Optional[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (t_centers, trace_counts, k_vals, p_exp).

        The implementation mirrors the logic used in the standalone PCH plugin
        (chisurf.plugins.pch.PCHApp.compute_trace_pch) but without any GUI
        dependencies.
        """

        # Routing-channel selection
        try:
            rc = np.asarray(tttr.routing_channels)
        except Exception:
            rc = None
        mask = np.ones_like(tttr.macro_times, dtype=bool)
        if rc is not None:
            try:
                mask &= np.isin(rc, np.asarray(list(channels), dtype=int))
            except Exception:
                pass

        # Micro-time window
        if micro_time_range is not None:
            mt_min, mt_max = micro_time_range
            try:
                mt = np.asarray(tttr.micro_times)
                mask &= (mt >= int(mt_min)) & (mt <= int(mt_max))
            except Exception:
                pass

        try:
            macro_res = float(getattr(tttr, "header").macro_time_resolution)
        except Exception:
            macro_res = 1.0

        try:
            mtimes = np.asarray(tttr.macro_times, dtype=np.int64)
        except Exception:
            mtimes = np.asarray([], dtype=np.int64)

        if mtimes.size == 0:
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != mtimes.shape:
            mask = np.ones_like(mtimes, dtype=bool)

        sel_times = mtimes[mask].astype(np.float64) * float(macro_res)
        if sel_times.size == 0:
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        bin_t = float(bin_time_us) * 1.0e-6
        if not (bin_t > 0.0):
            bin_t = 100.0e-6

        t_max = float(sel_times.max())
        if not (t_max > 0.0):
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        n_bins = int(np.ceil(t_max / bin_t))
        if n_bins <= 0:
            n_bins = 1

        counts, edges = np.histogram(
            sel_times,
            bins=n_bins,
            range=(0.0, bin_t * n_bins),
        )
        t_centers = 0.5 * (edges[:-1] + edges[1:])

        if counts.size == 0:
            return (
                t_centers.astype(float),
                counts.astype(float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
            )

        max_c = int(counts.max())
        if max_c < 0:
            max_c = 0
        hist_counts = np.bincount(counts, minlength=max_c + 1)
        total_bins = int(counts.size)
        if total_bins <= 0:
            total_bins = 1
        p_exp = hist_counts.astype(float) / float(total_bins)
        k_vals = np.arange(hist_counts.size, dtype=float)

        return t_centers.astype(float), counts.astype(float), k_vals, p_exp

    def read(
        self,
        filename: str | Sequence[str] | None = None,
        *args,
        **kwargs,
    ) -> chisurf.data.ExperimentDataCurveGroup:  # type: ignore[override]
        group = chisurf.data.ExperimentDataCurveGroup([])
        if filename is None:
            return group
        if isinstance(filename, (list, tuple)):
            if not filename:
                return group
            filename = filename[0]
        fn = pathlib.Path(str(filename))
        if not fn.is_file():
            return group

        if self.reading_routine:
            tttr = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
        else:
            tttr = tttrlib.TTTR(fn.as_posix())

        chs = self._get_channels()
        mtr = self._get_micro_time_range()
        t_centers, trace_counts, k_vals, p_exp = self._compute_trace_and_histogram(
            tttr=tttr,
            channels=chs,
            bin_time_us=float(self.bin_time_us),
            micro_time_range=mtr,
        )

        if k_vals.size == 0 or p_exp.size == 0:
            return group

        # Error estimate assuming counting statistics on the histogram counts.
        # Var[p(k)] ~ hist_counts / total_bins^2.
        try:
            counts_int = np.asarray(trace_counts, dtype=int)
            if counts_int.size == 0:
                hist_counts = np.zeros_like(k_vals, dtype=float)
            else:
                max_k = int(max(k_vals.max(), 0.0))
                hist_counts = np.bincount(counts_int, minlength=max_k + 1).astype(float)
        except Exception:
            hist_counts = np.zeros_like(k_vals, dtype=float)

        total_bins = float(trace_counts.size) if trace_counts.size > 0 else 1.0
        if total_bins <= 0.0:
            total_bins = 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            ey = np.where(
                hist_counts > 0.0,
                np.sqrt(hist_counts) / total_bins,
                1.0 / total_bins,
            )
        x = k_vals.astype(float)
        y = p_exp.astype(float)

        meta_pch = {
            "k_vals": k_vals.astype(float),
            "p_exp": p_exp.astype(float),
            "bin_time_us": float(self.bin_time_us),
            "channels": tuple(int(c) for c in chs),
            "micro_time_range": tuple(mtr) if mtr is not None else None,
            "filename": str(fn),
            "trace_time_centers": t_centers.astype(float),
            "trace_counts": trace_counts.astype(float),
            "hist_counts": hist_counts.astype(float),
            "total_bins": float(total_bins),
        }

        data = chisurf.data.DataCurve(
            name=fn.stem,
            x=x,
            y=y,
            ey=ey,
            filename=str(fn),
            data_reader=self,
            experiment=self.experiment,
            meta_data={"pch": meta_pch},
            load_filename_on_init=False,
        )
        group.append(data)
        group.data_reader = self
        return group
