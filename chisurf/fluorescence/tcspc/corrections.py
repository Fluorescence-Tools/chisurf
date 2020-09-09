from __future__ import annotations

import numba as nb
import numpy as np

import chisurf.math


def compute_linearization_table(
        data: np.ndarray,
        window_length: int,
        window_function_type: str,
        x_min: int,
        x_max: int,
        fill_value: float = 1.0
) -> np.ndarray:
    """Calculate a table to account for differential non-linearities.

    This function can be used to smooth a a measurement of uncorrelated light.
    The resulting table can be used to account for differential non-linearities
    (DNL) in TCSPC measurements. The function reduces the noise. Thus, the
    resulting table can be multiplied with a model function without increasing
    the noise.

    The function uses a smoothing window function function to reduce the high
    frequency noise.

    Parameters
    ----------
    data : numpy-array
        The experimental data that is used to compute a linearization table.
    window_length : int
        The length of the averaging window
    window_function_type : str
        The type of the window function that is used (either 'flat', 'hanning',
        'hamming', 'bartlett', 'blackman').
    x_min : int
        The smallest data point that is considered. Data points smaller than
        this value will be set to the parameter *fill_value*.
    x_max : int
        The largest data point that is considered. Data points larger than
        this value will be set to the parameter *fill_value*.
    fill_value : float
        Values outside of the the range [xmin, xmax] are set to this value
        (default: 1.0).

    Returns
    -------
    numpy-array
        The returned numpy array can be used as a table to perturb a model
        function by DNLs.

    Examples
    --------
    >>> import numpy as np
    >>> from chisurf.fluorescence.tcspc.corrections import compute_linearization_table
    >>> x = np.linspace(0, 40, 128)
    >>> dnl_fraction = 0.01
    >>> counts = 10000
    >>> mean = np.sin(x) * dnl_fraction  * counts + (1 - dnl_fraction) * counts
    >>> np.random.seed(0)
    >>> data = np.random.poisson(mean).astype(np.float64)
    >>> compute_linearization_table(data, 12, "hanning", 10, 90)
    array([1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 0.99980049, 0.99922759, 0.99823618, 0.99681489,
           0.99518504, 0.99393009, 0.99341097, 0.99356403, 0.99429588,
           0.99518651, 0.99570281, 0.99588077, 0.99593212, 0.99620644,
           0.99725223, 0.99907418, 1.00121338, 1.00320785, 1.00486314,
           1.00613769, 1.00711954, 1.00770543, 1.00739009, 1.00586985,
           1.00286934, 0.99849134, 0.99377306, 0.98981455, 0.98744986,
           0.98683355, 0.9874091 , 0.98868764, 0.99063644, 0.99339915,
           0.99669779, 1.00031741, 1.00394849, 1.00699941, 1.00914148,
           1.01040307, 1.01119998, 1.01183167, 1.01202324, 1.01120659,
           1.00905205, 1.0055979 , 1.0012921 , 0.99701193, 0.99370229,
           0.99214721, 0.99253283, 0.99448162, 0.99720137, 0.99972508,
           1.001355  , 1.00164798, 1.00115969, 1.00083382, 1.00120239,
           1.0025449 , 1.00450071, 1.00633953, 1.00702856, 1.00606332,
           1.00373424, 1.00058981, 0.99734557, 0.99442657, 0.9925278 ,
           0.99219378, 0.99332066, 0.9954457 , 0.99805603, 1.00079518,
           1.00285328, 1.00367601, 1.00341471, 1.00277917, 1.00262715,
           1.00308127, 1.00383407, 1.00447143, 1.00466959, 1.00415533,
           1.00289327, 1.00144407, 1.00036365, 0.99984595, 0.99978234,
           0.99991273, 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 1.        , 1.        , 1.        ,
           1.        , 1.        , 1.        ])

    """
    x2 = data.copy()
    x2 /= x2[x_min:x_max].mean()
    mnx = np.ma.array(x2)
    mask2 = np.array([i < x_min or i > x_max for i in range(len(x2))])
    mnx.mask = mask2
    mnx.fill_value = fill_value
    mnx /= mnx.mean()
    yn = mnx.filled()
    return chisurf.math.signal.window(
        data=yn,
        window_len=window_length,
        window_function_type=window_function_type
    )


@nb.jit(nopython=True, nogil=True)
def add_pile_up_to_model(
        data: np.ndarray,
        model: np.ndarray,
        rep_rate: float,
        dead_time: float,
        measurement_time: float,
        modify_inplace: bool
) -> np.ndarray:
    """Add pile up effect to model function.

    Notes
    -----
    This function scales the model function. Thus, the scaling needs to
    be adjusted after adding pile-up to the model. The function uses the
    assumptions as described in ref [1]_.

    Parameters
    ----------
    data : numpy-array
        The array containing the experimental decay
    model : numpy-array
        The array containing the model function
    rep_rate : float
        The repetition-rate in MHz
    dead_time : float
        The dead-time of the system in nanoseconds
    measurement_time : float
        The measurement time in seconds
    verbose : bool
        If this parameter is set to True information is printed to stdout.
    modify_inplace : bool
        If set to True (default) pile-up is added to the input model array and
        the input is modified inplace. If False a copy of the input model array
        is created and pile-up is added to the copy of the model.

    Returns
    -------
    numpy-array
        An array containing a model function with added pile-up.

    Examples
    --------
    >>> from chisurf.fluorescence.tcspc.corrections import add_pile_up_to_model
    >>> tau = 4.0; ampl = 10000
    >>> x = np.linspace(0, 40, 64); y = ampl * np.exp(-x/tau)
    >>> pile_ = add_pile_up_to_model(y, y, 10.0, 200., 5.0, False)

    References
    ----------

    .. [1]  Coates, P.B., The correction for photon pile-up in the measurement
            of radiative lifetimes 1968 J. Phys. E: Sci. Instrum. 1 878

    .. [2]  Walker, J.G., Iterative correction for pile-up in single-photon
            lifetime measurement 2002 Optics Comm. 201 271-277

    """
    rep_rate *= 1e6
    dead_time *= 1e-9
    cum_sum = np.cumsum(data)
    n_pulse_detected = cum_sum[-1]
    total_dead_time = n_pulse_detected * dead_time
    live_time = measurement_time - total_dead_time
    n_excitation_pulses = max(live_time * rep_rate, n_pulse_detected)

    # Coates, 1968, eq. 2
    p = data / (n_excitation_pulses - np.cumsum(data))
    # Coates, 1968, eq. 4
    rescaled_data = -np.log(1.0 - p)
    rescaled_data[rescaled_data == 0] = 1.0

    # instead of rescaling the data, the model function is
    # rescaled, to preserve the counting statistics and the
    # known noise.
    sf = data / rescaled_data
    sf = sf / np.sum(sf) * len(data)
    if modify_inplace:
        model *= sf
        return model
    else:
        a = np.empty(model.shape, dtype=np.float64)
        for i, mv in enumerate(model):
            a[i] = mv * sf[i]
        return a

