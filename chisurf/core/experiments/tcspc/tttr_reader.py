from __future__ import annotations

import os.path
import pathlib

import numpy as np
import tttrlib

import chisurf.core.data
import chisurf.core.fluorescence.tcspc

from chisurf import typing

from .reader import TCSPCReader

_VIEW_JSON = pathlib.Path(__file__).parent / "tcspc_tttr.view.json"


class TCSPCTTTRReader(TCSPCReader):
    operation_type = "tcspc_histogram_computation"
    artifact_kind_source = "raw_data"
    artifact_kind_derived = "tcspc_decay"
    derived_data_format = "json"
    derived_mime_type = "application/json"

    def __init__(
            self,
            *args,
            channel_numbers=None,
            channel: int = 0,
            micro_time_coarsening: int = 1,
            micro_time_shift: int = 0,
            reading_routine: str | None = None,
            **kwargs
    ):
        """Initialize a TCSPC TTTR reader.

        Parameters
        ----------
        channel_numbers : list of int, optional
            Routing channel numbers to select.
        channel : int
            Fallback single routing channel index.
        micro_time_coarsening : int
            Coarsening factor for the micro-time histogram.
        micro_time_shift : int
            Shift applied to the micro-time histogram (in bins).
        reading_routine : str or None
            tttrlib reading routine (e.g. ``'PTU'``).
        """
        super().__init__(*args, **kwargs)
        if reading_routine is not None:
            self.reading_routine = reading_routine
        if not hasattr(self, "channel_numbers"):
            self.channel_numbers = None
        if channel_numbers is not None:
            try:
                self.channel_numbers = list(channel_numbers)
            except TypeError:
                self.channel_numbers = [channel_numbers]
        self.channel = int(channel)
        try:
            self.micro_time_coarsening = int(micro_time_coarsening)
        except Exception:
            self.micro_time_coarsening = 1
        try:
            self.micro_time_shift = int(micro_time_shift)
        except Exception:
            self.micro_time_shift = 0

    # -- declarative editor adapters ---------------------------------------
    @property
    def channel_numbers_str(self) -> str:
        """Comma-separated routing channels; mirrors ``channel_numbers``."""
        chs = getattr(self, "channel_numbers", None)
        if chs is None:
            chs = [getattr(self, "channel", 0)]
        try:
            return ", ".join(str(int(c)) for c in chs)
        except Exception:
            return ""

    @channel_numbers_str.setter
    def channel_numbers_str(self, value: str) -> None:
        chs = []
        for part in str(value).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                chs.append(int(part))
            except Exception:
                continue
        if not chs:
            chs = [0]
        self.channel_numbers = np.array(chs, dtype=np.int8)
        self.channel = int(chs[0])

    def view_spec(self):
        """Return the declarative editor spec for the TCSPC-TTTR reader."""
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_VIEW_JSON)

    def _get_channels(self) -> typing.Tuple[int, ...]:
        """Return the sorted tuple of routing channel numbers.

        Returns
        -------
        tuple of int
            Sorted unique channel indices.
        """
        chs = getattr(self, "channel_numbers", None)
        if chs is None:
            chs = [getattr(self, "channel", 0)]
        try:
            return tuple(sorted({int(c) for c in chs}))
        except Exception:
            return (int(getattr(self, "channel", 0) or 0),)

    def _get_micro_time_coarsening(self) -> int:
        """Return the micro-time histogram coarsening factor.

        Returns
        -------
        int
            Coarsening factor (minimum 1).
        """
        try:
            mtc = int(getattr(self, "micro_time_coarsening", 1) or 1)
        except Exception:
            mtc = 1
        if mtc <= 0:
            mtc = 1
        return mtc

    def _get_micro_time_shift(self) -> int:
        """Return the micro-time histogram shift in bins.

        Returns
        -------
        int
            Shift value (may be negative).
        """
        try:
            s = int(getattr(self, "micro_time_shift", 0) or 0)
        except Exception:
            s = 0
        return s

    def _apply_shift(self, y: np.ndarray, shift: int) -> np.ndarray:
        """Apply a circular-like shift to a histogram array.

        Positive *shift* pads at the beginning (right-shift);
        negative *shift* pads at the end (left-shift).

        Parameters
        ----------
        y : np.ndarray
            The input histogram array.
        shift : int
            Number of bins to shift.

        Returns
        -------
        np.ndarray
            The shifted array (same length as input).
        """
        arr = np.asarray(y, dtype=float)
        if arr.size == 0 or shift == 0:
            return arr
        if shift > 0:
            return np.pad(arr, (shift, 0), mode="constant")[:-shift]
        step = abs(shift)
        return np.pad(arr, (0, step), mode="constant")[step:]

    def _compute_histogram(self, filename: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Compute a micro-time histogram from a TTTR file.

        Parameters
        ----------
        filename : str
            Path to the TTTR file.

        Returns
        -------
        tuple of np.ndarray
            ``(y, x)`` where *y* is the histogram counts and *x* the
            micro-time axis in nanoseconds.
        """
        routine = getattr(self, "reading_routine", None)
        if routine:
            tttr = tttrlib.TTTR(filename, routine)
        else:
            tttr = tttrlib.TTTR(filename)
        # Save header JSON for later metadata display (avoids re-reading the file)
        try:
            self._tttr_header_json = tttr.header.json
        except Exception:
            self._tttr_header_json = ""
        chs = self._get_channels()
        coarsening = self._get_micro_time_coarsening()
        shift = self._get_micro_time_shift()
        tttr_selected = tttr.get_tttr_by_channel(list(chs))
        y_raw, x_raw = tttr_selected.get_microtime_histogram(coarsening)
        y = np.asarray(y_raw, dtype=float)
        x = np.asarray(x_raw, dtype=float)
        if y.size == 0 or x.size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        if shift != 0:
            y = self._apply_shift(y, shift)
        x = x * 1.0e9
        n = int(min(y.size, x.size))
        if n <= 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        return y[:n].astype(float), x[:n].astype(float)

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.core.data.DataCurveGroup:
        """Read a TTTR file and return a TCSPC decay curve.

        Parameters
        ----------
        filename : str, optional
            Path to the TTTR file.

        Returns
        -------
        chisurf.core.data.DataCurveGroup
            Group containing the TCSPC decay.
        """
        if filename is None:
            return chisurf.core.data.DataGroup([])
        if isinstance(filename, (list, tuple)):
            if not filename:
                return chisurf.core.data.DataGroup([])
            filename = filename[0]
        if not os.path.isfile(filename):
            return chisurf.core.data.DataGroup([])
        try:
            y, t = self._compute_histogram(filename)
        except Exception:
            return chisurf.core.data.DataGroup([])
        if y.size == 0 or t.size == 0:
            return chisurf.core.data.DataGroup([])
        try:
            y_pos = np.where(y > 0)[0]
            if y_pos.size > 0:
                i_y_max = int(y_pos[-1]) + 1
                y = y[:i_y_max]
                t = t[:i_y_max]
        except Exception:
            pass
        fn, _ = os.path.splitext(filename)
        try:
            chs = self._get_channels()
            ch_text = ",".join(str(int(c)) for c in chs)
        except Exception:
            ch_text = ""
        name = f"{fn}_ch({ch_text})"
        meta_data = {}
        hdr = getattr(self, "_tttr_header_json", "")
        if hdr:
            meta_data["tttr_header_json"] = hdr
        data_set = chisurf.core.data.DataCurve(
            x=t,
            y=y,
            name=name,
            meta_data=meta_data,
            experiment=self.experiment,
            data_reader=self,
            ey=chisurf.core.fluorescence.tcspc.counting_noise(y)
        )
        data_set.filename = filename
        data_group = chisurf.core.data.DataGroup([data_set])
        data_group.data_reader = self
        return data_group
