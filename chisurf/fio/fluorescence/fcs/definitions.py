from __future__ import annotations

import numpy as np

from chisurf import typing


class FCSDataset(typing.TypedDict):
    filename: str
    measurement_id: str
    acquisition_time: float
    mean_count_rate: float
    correlation_times: np.ndarray
    correlation_amplitudes: np.ndarray
    correlation_amplitude_weights: np.ndarray
    intensity_trace_times: np.ndarray
    intensity_trace: np.ndarray
    intensity_trace_name: str
    meta_data: typing.Dict

