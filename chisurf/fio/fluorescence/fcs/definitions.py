from __future__ import annotations

import numpy as np

from chisurf import typing


class FCSDataset(typing.TypedDict):
    filename: str
    measurement_id: str
    acquisition_time: float
    mean_count_rates: typing.List[float]
    correlation_times: typing.List[np.ndarray]
    correlation_amplitudes: typing.List[np.ndarray]
    correlation_amplitude_weights: typing.List[np.ndarray]
    correlation_names: typing.List[str]
    intensity_trace_times: typing.List[np.ndarray]
    intensity_traces: typing.List[np.ndarray]
    intensity_trace_names: typing.List[str]
    meta_data: typing.Dict

