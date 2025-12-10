"""
Scaling functions for lifetime fitting.

This module provides functions for scaling model decays to experimental decays.
"""

from __future__ import annotations
import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True)
def rescale_w_bg(
        model_decay: np.array,
        experimental_decay: np.array,
        experimental_weights: np.array,
        experimental_background: float,
        start: int,
        stop: int
) -> float:
    """Computes a scaling factor that scales a model decay to an
    experimental decay on a defined range.

    Parameters
    ----------
    model_decay : numpy.array
        Model decay for that a scaling factor is computed.
    experimental_decay : numpy.array
        Experimental decay to which the model decay provided by `model_decay`
        is scaled by the returned floating number
    experimental_weights : numpy.array
        Weights of the experimental decay that are used scale the model to
        the experiment.
    experimental_background : float
        Constant offset in the experimental data that is subtracted from the
        experimental decay
    start : int
        Start index that defines the range in which the model decay is scaled
        to the experimental decay
    stop : int
        Stop index that defines the range in which the model decay is scaled
        to the experimental decay.

    Returns
    -------
    float
        The scaling factor that was used to scale the model function to the
        experimental decay.
    """
    sum_nom = 0.0
    sum_denom = 0.0
    w = experimental_weights
    e = experimental_decay
    b = experimental_background
    m = model_decay
    for i in range(start, stop):
        if e[i] > 0.0:
            iwsq = 1.0 / (w[i-start] * w[i-start] + 1e-12)
            sum_nom += m[i] * (e[i] - b) * iwsq
            sum_denom += m[i] * m[i] * iwsq
    scale = sum_nom / max(1.0, sum_denom)
    return scale


def scale_model_to_data(
        model_decay: np.array,
        experimental_decay: np.array,
        start: int,
        stop: int,
        experimental_background: float = 0.0,
        use_weights: bool = True
) -> float:
    """
    Scale a model decay to an experimental decay.

    Parameters
    ----------
    model_decay : numpy.array
        Model decay to scale
    experimental_decay : numpy.array
        Experimental decay to scale to
    start : int
        Start index for scaling
    stop : int
        Stop index for scaling
    experimental_background : float
        Background to subtract from experimental decay
    use_weights : bool
        Whether to use weights for scaling

    Returns
    -------
    float
        Scaling factor
    """
    if use_weights:
        # Calculate weights assuming Poisson noise
        weights = 1.0 / np.sqrt(np.maximum(experimental_decay[start:stop], 1.0))

        # Use the rescale_w_bg function
        scale = rescale_w_bg(
            model_decay=model_decay,
            experimental_decay=experimental_decay,
            experimental_weights=weights,
            experimental_background=experimental_background,
            start=start,
            stop=stop
        )
    else:
        # Simple scaling using ratio of sums
        scale = np.sum(experimental_decay[start:stop] - experimental_background) / np.sum(model_decay[start:stop])

    # Apply scaling
    model_decay *= scale

    return scale
