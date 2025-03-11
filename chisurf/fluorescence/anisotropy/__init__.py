from chisurf.fluorescence.intensity import nusiance
import chisurf.fluorescence.anisotropy.decay

# Global constants (normally defined elsewhere)
Bp = 10.0     # Background for parallel signal correction
Bs = 5.0      # Background for vertical signal correction
Gfactor = 1.2 # Gain factor to correct detection sensitivity
l1 = 0.1      # Mixing factor for the parallel channel
l2 = 0.2      # Mixing factor for the vertical channel


@nusiance
def r_scatter(
        signal_vertical,
        signal_parallel,
        **kwargs
):
    """
    Computes the scatter anisotropy ratio from vertical and parallel signals.

    The ratio is computed after applying background corrections and a gain
    factor. In detail, the modified parallel and vertical signals are calculated as:

        F_p = (signal_parallel - Bp) * Gfactor
        F_s = signal_vertical - Bs

    and the scatter anisotropy is given by:

        r_scatter = (F_p - F_s) / (F_p * (1 - 3*l2) + F_s * (2 - 3*l1))

    Parameters
    ----------
    signal_vertical : float
        Measured signal in the vertical detection channel.
    signal_parallel : float
        Measured signal in the parallel detection channel.
    **kwargs : dict
        Additional keyword arguments (not used by this function).

    Returns
    -------
    float
        The computed scatter anisotropy ratio.

    Examples
    --------
    >>> Bp = 10.0; Bs = 5.0; Gfactor = 1.2; l1 = 0.1; l2 = 0.2
    >>> # Given a vertical signal of 100 and a parallel signal of 150:
    >>> # F_p = (150 - 10) * 1.2 = 168
    >>> # F_s = 100 - 5 = 95
    >>> # Denom = 168*(1 - 0.6) + 95*(2 - 0.3) = 168*0.4 + 95*1.7 = 67.2 + 161.5 = 228.7
    >>> # r_scatter = (168 - 95) / 228.7 ≈ 73/228.7 ≈ 0.3193...
    >>> r_scatter(signal_vertical=100, signal_parallel=150)
    0.3193...
    """
    Fp = (signal_parallel - Bp) * Gfactor
    Fs = signal_vertical - Bs
    return (Fp - Fs) / (Fp * (1. - 3. * l2) + Fs * (2. - 3. * l1))


@nusiance
def r_exp(
        signal_parallel,
        signal_vertical,
        **kwargs
) -> float:
    """
    Computes the experimental anisotropy from parallel and vertical signals.

    The experimental anisotropy is calculated using the formula:

        F_p = signal_parallel * Gfactor
        r_exp = (F_p - signal_vertical) / (F_p * (1 - 3*l2) + signal_vertical * (2 - 3*l1))

    Parameters
    ----------
    signal_parallel : float
        Measured signal in the parallel detection channel.
    signal_vertical : float
        Measured signal in the vertical detection channel.
    **kwargs : dict
        Additional keyword arguments (not used by this function).

    Returns
    -------
    float
        The computed experimental anisotropy.

    Examples
    --------
    >>> Bp = 10.0; Bs = 5.0; Gfactor = 1.2; l1 = 0.1; l2 = 0.2
    >>> # For signal_parallel=150 and signal_vertical=100:
    >>> # F_p = 150 * 1.2 = 180
    >>> # Denom = 180*(1-0.6) + 100*(2-0.3) = 180*0.4 + 100*1.7 = 72 + 170 = 242
    >>> # r_exp = (180 - 100) / 242 = 80/242 ≈ 0.3306...
    >>> r_exp(signal_parallel=150, signal_vertical=100)
    0.3305...
    """
    Fp = signal_parallel * Gfactor
    return (Fp - signal_vertical) / (Fp * (1. - 3. * l2) + signal_vertical * (2. - 3. * l1))


