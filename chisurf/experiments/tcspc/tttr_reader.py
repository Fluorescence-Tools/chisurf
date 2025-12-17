from __future__ import annotations

import os.path

import numpy as np
import tttrlib

import chisurf.data
import chisurf.fluorescence.tcspc

from chisurf import typing

from .reader import TCSPCReader


class TCSPCTTTRReader(TCSPCReader):

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

    def _get_channels(self) -> typing.Tuple[int, ...]:
        chs = getattr(self, "channel_numbers", None)
        if chs is None:
            chs = [getattr(self, "channel", 0)]
        try:
            return tuple(sorted({int(c) for c in chs}))
        except Exception:
            return (int(getattr(self, "channel", 0) or 0),)

    def _get_micro_time_coarsening(self) -> int:
        try:
            mtc = int(getattr(self, "micro_time_coarsening", 1) or 1)
        except Exception:
            mtc = 1
        if mtc <= 0:
            mtc = 1
        return mtc

    def _get_micro_time_shift(self) -> int:
        try:
            s = int(getattr(self, "micro_time_shift", 0) or 0)
        except Exception:
            s = 0
        return s

    def _apply_shift(self, y: np.ndarray, shift: int) -> np.ndarray:
        arr = np.asarray(y, dtype=float)
        if arr.size == 0 or shift == 0:
            return arr
        if shift > 0:
            return np.pad(arr, (shift, 0), mode="constant")[:-shift]
        step = abs(shift)
        return np.pad(arr, (0, step), mode="constant")[step:]

    def _compute_histogram(self, filename: str) -> typing.Tuple[np.ndarray, np.ndarray]:
        routine = getattr(self, "reading_routine", None)
        if routine:
            tttr = tttrlib.TTTR(filename, routine)
        else:
            tttr = tttrlib.TTTR(filename)
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

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.data.DataCurveGroup:
        if filename is None:
            return chisurf.data.DataGroup([])
        if isinstance(filename, (list, tuple)):
            if not filename:
                return chisurf.data.DataGroup([])
            filename = filename[0]
        if not os.path.isfile(filename):
            return chisurf.data.DataGroup([])
        try:
            y, t = self._compute_histogram(filename)
        except Exception:
            return chisurf.data.DataGroup([])
        if y.size == 0 or t.size == 0:
            return chisurf.data.DataGroup([])
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
        data_set = chisurf.data.DataCurve(
            x=t,
            y=y,
            name=name,
            experiment=self.experiment,
            data_reader=self,
            ey=chisurf.fluorescence.tcspc.counting_noise(y)
        )
        data_group = chisurf.data.DataGroup([data_set])
        data_group.data_reader = self
        return data_group
