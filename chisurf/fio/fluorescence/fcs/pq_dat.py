from __future__ import annotations

import os
import numpy as np
from chisurf import typing


def read_dat(
        filename: str,
        verbose: bool = False
) -> typing.List[typing.Dict]:
    if verbose:
        print("Reading PicoQuant data file: %s" % filename)
    correlations = list()
    d = np.loadtxt(filename, skiprows=2, encoding="latin1").T
    n_corr = d.shape[0] // 3
    for i in range(n_corr):
        correlation_time = d[i * 3 + 0]
        correlation_amplitude = d[i * 3 + 1]
        correlation_weight = 1. / d[i * 3 + 2]
        corr = {
            'filename': filename,
            'measurement_id': "%s_%s" % (
                os.path.splitext(os.path.basename(filename))[1], i
            ),
            'correlation_time': correlation_time.tolist(),
            'correlation_amplitude': correlation_amplitude.tolist(),
            'correlation_amplitude_weights': correlation_weight.tolist(),
        }
        correlations.append(
            corr
        )
    return correlations
