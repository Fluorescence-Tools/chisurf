import numpy as np

from chisurf.fluorescence.intensity import nusiance
import chisurf.fluorescence.fret.acceptor as acceptor

# Note: Global parameters assumed to be defined elsewhere in the module/environment:
#   Bg         : Background signal for the green channel.
#   Br         : Background signal for the red channel.
#   crosstalk  : Fraction of green signal that leaks into the red detection channel.
#   phiA       : Acceptor quantum yield.
#   phiD       : Donor quantum yield.
#   Gfactor    : Correction factor for the green channel detector sensitivity.
#   R0         : Förster radius.

@nusiance
def sg_sr(
        Sg,
        Sr,
        **kwargs
) -> float:
    """
    Calculate the raw intensity ratio of the green to red signals.

    This function returns the simple ratio Sg/Sr of the green (Sg) and red (Sr)
    signal intensities. No corrections (such as background subtraction or cross-talk)
    are applied. This ratio is sometimes used as a proxy for overall color balance.

    Parameters
    ----------
    Sg : float
        Measured green signal intensity.
    Sr : float
        Measured red signal intensity.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The ratio Sg/Sr. If Sr is zero, returns NaN.

    References
    ----------
    For general background on fluorescence intensity measurements, see:
    Lakowicz, J.R., *Principles of Fluorescence Spectroscopy*, 3rd ed. (2006).
    """
    if Sr != 0:
        return float(Sg) / float(Sr)
    else:
        return float('nan')


@nusiance
def fg_fr(
        Sg,
        Sr,
        **kwargs
) -> float:
    """
    Calculate the ratio of corrected fluorescence intensities.

    The green fluorescence intensity Fg and the red fluorescence intensity Fr
    are corrected for background and cross-talk:
    
        Fg = Sg - Bg
        Fr = Sr - Br - (Fg * crosstalk)

    The function returns the ratio Fg/Fr, which is used in subsequent FRET analyses.

    Parameters
    ----------
    Sg : float
        Measured green signal intensity.
    Sr : float
        Measured red signal intensity.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The ratio Fg/Fr of the corrected fluorescence intensities.

    References
    ----------
    Lakowicz, J.R., *Principles of Fluorescence Spectroscopy*, 3rd ed. (2006).
    """
    Fg = Sg - Bg
    Fr = Sr - Br - Fg * crosstalk
    return Fg / Fr


@nusiance
def proximity_ratio(
        Sg,
        Sr,
        **kwargs
) -> float:
    """
    Compute the proximity ratio for FRET experiments.

    The proximity ratio is defined as the fraction of the total (uncorrected)
    intensity that is detected in the red channel:
    
        proximity_ratio = Sr / (Sg + Sr)

    This measure is widely used in single-molecule FRET experiments to provide
    a first estimate of the donor–acceptor proximity.

    Parameters
    ----------
    Sg : float
        Measured green signal intensity.
    Sr : float
        Measured red signal intensity.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The proximity ratio, a value between 0 and 1.
    """
    return Sr / (Sg + Sr)


@nusiance
def apparent_fret_efficiency(
        Sg,
        Sr,
        **kwargs
) -> float:
    """
    Calculate the apparent FRET efficiency from corrected fluorescence intensities.

    The apparent FRET efficiency is defined as:
    
        E_app = Fr / (Fg + Fr)

    where the corrected intensities are computed as:
    
        Fg = Sg - Bg
        Fr = Sr - Br - (Fg * crosstalk)

    Note that this value does not account for differences in instrument spectral
    sensitivity or for the fluorescence quantum yields of the dyes.

    Parameters
    ----------
    Sg : float
        Measured green signal intensity.
    Sr : float
        Measured red signal intensity.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The apparent FRET efficiency.
    """
    Fg = Sg - Bg
    Fr = Sr - Br - Fg * crosstalk
    return Fr / (Fg + Fr)


@nusiance
def fret_efficiency(
        Sg,
        Sr,
        **kwargs
) -> float:
    """
    Calculate the corrected FRET efficiency using fluorescence intensity corrections.

    The corrected FRET efficiency is computed by first correcting the measured
    intensities for background and cross-talk, and then scaling them by the
    detector sensitivity (Gfactor) and dye quantum yields. The calculations are:
    
        Fg = Sg - Bg
        Fa = (Sr - Br - Fg * crosstalk) / phiA

        E = Fa / ((Fg / (Gfactor * phiD)) + Fa)

    This corrected efficiency accounts for instrumental factors and differences
    in dye properties.

    Parameters
    ----------
    Sg : float
        Measured green signal intensity.
    Sr : float
        Measured red signal intensity.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The corrected FRET efficiency.

    References
    ----------
    For FRET efficiency corrections and theory, see:
    Lakowicz, J.R., *Principles of Fluorescence Spectroscopy*, 3rd ed. (2006).
    """
    Fg = Sg - Bg
    Fa = (Sr - Br - Fg * crosstalk) / phiA
    return Fa / (Fg / (Gfactor * phiD) + Fa)


@nusiance
def fluorescence_weighted_distance(
        Sg,
        Sr,
        **kwargs
) -> float:
    """
    Compute the fluorescence-weighted donor–acceptor distance.

    This function estimates the donor–acceptor distance (R_DA,E) based on the
    fluorescence intensities corrected for background, cross-talk, and scaled by
    the detector sensitivity and dye quantum yields. The calculation is based on
    the Förster relation:

        R_DA,E = R0 * (Fg / (Gfactor * phiD * Fa))^(1/6)

    where:
      - Fg = Sg - Bg
      - Fa = (Sr - Br - Fg * crosstalk) / phiA

    Parameters
    ----------
    Sg : float
        Measured green signal intensity.
    Sr : float
        Measured red signal intensity.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The fluorescence-weighted distance between donor and acceptor.

    References
    ----------
    Lakowicz, J.R., *Principles of Fluorescence Spectroscopy*, 3rd ed. (2006).
    """
    Fg = Sg - Bg
    Fa = (Sr - Br - Fg * crosstalk) / phiA
    return R0 * np.exp((1.0/6.0) * np.log(Fg / (Gfactor * phiD * Fa)))


@nusiance
def fret_efficency_to_fdfa(
        E,
        **kwargs
):
    """
    Convert FRET transfer efficiency to the donor–acceptor intensity ratio (FD/FA).

    This function converts the FRET efficiency E to the intensity ratio FD/FA using
    the relationship:
    
        FD/FA = (phiA / phiD) * (1/E - 1)

    where phiA and phiD are the acceptor and donor quantum yields, respectively.

    Parameters
    ----------
    E : float
        The FRET transfer efficiency.
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The donor–acceptor intensity ratio (FD/FA).

    References
    ----------
    Lakowicz, J.R., *Principles of Fluorescence Spectroscopy*, 3rd ed. (2006).
    """
    fdfa = phiA / phiD * (1.0 / E - 1.0)
    return fdfa


@nusiance
def fdfa2transfer_efficency(fdfa):
    """
    Convert the donor–acceptor intensity ratio (FD/FA) to the FRET transfer efficiency.

    This function performs the inverse conversion of the relation used in
    fret_efficency_to_fdfa. Specifically, it computes the FRET efficiency E from
    the intensity ratio FD/FA using:

        E = 1 / (1 + (FD/FA) * (phiA / phiD))

    Parameters
    ----------
    fdfa : float
        The donor–acceptor intensity ratio (FD/FA).
    **kwargs : dict
        Additional keyword arguments (not used).

    Returns
    -------
    float
        The FRET transfer efficiency E.

    References
    ----------
    Lakowicz, J.R., *Principles of Fluorescence Spectroscopy*, 3rd ed. (2006).
    """
    r = 1.0 + fdfa * phiA / phiD
    trans = 1.0 / r
    return trans
