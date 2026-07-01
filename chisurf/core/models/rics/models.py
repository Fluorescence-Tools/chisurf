from __future__ import annotations

import numpy as np


def _pos(x: float, floor: float = 1e-12) -> float:
    """Return ``max(abs(x), floor)``.

    RICS fit functions divide by ``N`` and the beam waists, so unconstrained
    optimizer excursions to zero/negative values would produce ``inf``/``nan``.
    Following PAM's ``.miafit`` convention (which uses ``|N|``, ``|D|``, ...),
    magnitudes are taken and divisors floored to keep the model finite for any
    parameter value the optimizer proposes.
    """
    v = abs(float(x))
    return v if v > floor else floor


def rics_simple(
    line_shift: np.ndarray,
    pixel_shift: np.ndarray,
    n: float,
    diffusion_coefficient: float = 2.0,
    offset: float = 0.0,
    pixel_duration: float = 11.1,
    line_duration: float = 3.33,
    pixel_size: float = 40.0,
    w_r: float = 0.2,
    w_z: float = 1.0,
) -> np.ndarray:
    """Simple one-component 3D diffusion RICS model.

    Parameters mirror the tttrlib example
    ``modules/tttrlib/examples/image_correlation/plot_imaging_ics_fit.py``.

    All time parameters are given in the usual microscopy units and internally
    converted to SI units:

    - ``diffusion_coefficient`` in µm^2/s
    - ``pixel_duration`` in µs
    - ``line_duration`` in ms
    - ``pixel_size`` in nm
    - ``w_r``, ``w_z`` in µm
    """
    n = _pos(n)
    D = abs(float(diffusion_coefficient)) * 1.0e-12
    tau_p = float(pixel_duration) * 1.0e-6
    tau_l = float(line_duration) * 1.0e-3
    a = float(pixel_size) * 1.0e-9
    w_r_m = _pos(w_r, 1e-3) * 1.0e-6
    w_z_m = _pos(w_z, 1e-3) * 1.0e-6

    line_shift = np.asarray(line_shift, dtype=float)
    pixel_shift = np.asarray(pixel_shift, dtype=float)

    mv = np.abs(pixel_shift * tau_p + line_shift * tau_l)

    denom_r = 1.0 + 4.0 * D * mv / (w_r_m ** 2)
    denom_z = 1.0 + 4.0 * D * mv / (w_z_m ** 2)

    spatial = np.exp(-a ** 2 * (pixel_shift ** 2 + line_shift ** 2) / (w_r_m ** 2 + 4.0 * D * mv))

    return (
        offset
        + (2.0 ** (-1.5) / n)
        * denom_r ** (-1.0)
        * denom_z ** (-0.5)
        * spatial
    )


def rics_immobile(
    line_shift: np.ndarray,
    pixel_shift: np.ndarray,
    n: float,
    diffusion_coefficient: float = 2.0,
    offset: float = 0.0,
    pixel_duration: float = 11.1,
    line_duration: float = 3.33,
    pixel_size: float = 40.0,
    w_r: float = 0.2,
    w_z: float = 1.0,
    a_immobile: float = 0.0,
) -> np.ndarray:
    """RICS model with a mobile 3D-diffusion component and an immobile fraction.

    Port of PAM's ``RICS_2Comp_Imm`` / ``RICS_2D_Imm_*`` family: a mobile
    diffusing species (as in :func:`rics_simple`) plus a static, non-diffusing
    component that contributes a lag-independent spatial Gaussian
    ``a_immobile * exp(-a^2 (xi^2 + psi^2) / w_r^2)`` (the auto-correlation of the
    immobile structure). ``a_immobile`` is the immobile-component amplitude.
    """
    mobile = rics_simple(
        line_shift, pixel_shift, n, diffusion_coefficient, 0.0,
        pixel_duration, line_duration, pixel_size, w_r, w_z,
    )
    a = float(pixel_size) * 1.0e-9
    w_r_m = _pos(w_r, 1e-3) * 1.0e-6
    line_shift = np.asarray(line_shift, dtype=float)
    pixel_shift = np.asarray(pixel_shift, dtype=float)
    immobile = abs(float(a_immobile)) * np.exp(
        -a ** 2 * (pixel_shift ** 2 + line_shift ** 2) / (w_r_m ** 2)
    )
    return float(offset) + mobile + immobile


def rics_flow(
    line_shift: np.ndarray,
    pixel_shift: np.ndarray,
    n: float,
    diffusion_coefficient: float = 2.0,
    offset: float = 0.0,
    pixel_duration: float = 11.1,
    line_duration: float = 3.33,
    pixel_size: float = 40.0,
    w_r: float = 0.2,
    w_z: float = 1.0,
    v_x: float = 0.0,
    v_y: float = 0.0,
) -> np.ndarray:
    """RICS model with 3D diffusion and uniform flow.

    Port of PAM's ``2D_Gaussian_Cor_Flow``: the scanning displacement
    ``a * xi`` (fast axis) and ``a * psi`` (slow axis) is offset by the flow
    ``v * tau`` accumulated over the scan lag ``tau``. ``v_x`` / ``v_y`` are the
    flow velocities (µm/s) along the fast/slow scan axes.
    """
    n = _pos(n)
    D = abs(float(diffusion_coefficient)) * 1.0e-12
    tau_p = float(pixel_duration) * 1.0e-6
    tau_l = float(line_duration) * 1.0e-3
    a = float(pixel_size) * 1.0e-9
    w_r_m = _pos(w_r, 1e-3) * 1.0e-6
    w_z_m = _pos(w_z, 1e-3) * 1.0e-6
    vx = float(v_x) * 1.0e-6
    vy = float(v_y) * 1.0e-6

    line_shift = np.asarray(line_shift, dtype=float)
    pixel_shift = np.asarray(pixel_shift, dtype=float)

    mv = np.abs(pixel_shift * tau_p + line_shift * tau_l)
    denom_r = 1.0 + 4.0 * D * mv / (w_r_m ** 2)
    denom_z = 1.0 + 4.0 * D * mv / (w_z_m ** 2)

    dx = a * pixel_shift - vx * mv
    dy = a * line_shift - vy * mv
    spatial = np.exp(-(dx ** 2 + dy ** 2) / (w_r_m ** 2 + 4.0 * D * mv))

    return (
        float(offset)
        + (2.0 ** (-1.5) / n) * denom_r ** (-1.0) * denom_z ** (-0.5) * spatial
    )


def rics_full(
    line_shift: np.ndarray,
    pixel_shift: np.ndarray,
    n: float,
    diffusion_coefficient: float = 2.0,
    offset: float = 0.0,
    pixel_duration: float = 11.1,
    line_duration: float = 3.33,
    pixel_size: float = 40.0,
    w_r: float = 0.2,
    w_z: float = 1.0,
    tauT: float = 0.0,
    aT: float = 0.0,
    n_immobile: float = 0.0,
    w_immobile: float = 0.2,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    two_d: bool = False,
) -> np.ndarray:
    """Comprehensive RICS model: mobile diffusion + immobile + blinking + shift.

    Port of PAM's ``RICS_Mod_Imm_with_Shift`` / ``RICS_2D_Imm_with_blinking``
    family. A mobile diffusing species (amplitude ``n``) with optional
    triplet/blinking (``tauT``, ``aT``) plus a static immobile component
    (amplitude ``n_immobile``, width ``w_immobile``); both are laterally shifted
    by ``(shift_x, shift_y)`` [nm] for cross-correlation (ccRICS). With
    ``two_d=True`` the axial (``w_z``) term is dropped (membrane/2D geometry).

    Follows PAM's amplitude normalisation
    ``2^(-3/2) / (n + n_immobile)^2 * (n * mobile + n_immobile * immobile)``.
    """
    n = _pos(n)
    n_imm = abs(float(n_immobile))
    D = abs(float(diffusion_coefficient)) * 1.0e-12
    tau_p = float(pixel_duration) * 1.0e-6
    tau_l = float(line_duration) * 1.0e-3
    a = float(pixel_size) * 1.0e-9
    w_r_m = _pos(w_r, 1e-3) * 1.0e-6
    w_z_m = _pos(w_z, 1e-3) * 1.0e-6
    w_imm_m = _pos(w_immobile, 1e-3) * 1.0e-6
    tauT_s = _pos(tauT, 1e-9) * 1.0e-3
    px_nm = _pos(pixel_size, 1e-6)

    line_shift = np.asarray(line_shift, dtype=float)
    pixel_shift = np.asarray(pixel_shift, dtype=float)

    # Lateral shift (nm -> pixels) applied to both components (ccRICS).
    xs = pixel_shift - float(shift_x) / px_nm
    ys = line_shift - float(shift_y) / px_nm

    mv = np.abs(pixel_shift * tau_p + line_shift * tau_l)

    triplet = 1.0 + (float(aT) / max(1.0 - float(aT), 1e-12)) * np.exp(-mv / tauT_s)
    denom_r = 1.0 + 4.0 * D * mv / (w_r_m ** 2)
    mobile = triplet * denom_r ** (-1.0)
    if not two_d:
        denom_z = 1.0 + 4.0 * D * mv / (w_z_m ** 2)
        mobile = mobile * denom_z ** (-0.5)
    mobile = mobile * np.exp(
        -a ** 2 * (xs ** 2 + ys ** 2) / (w_r_m ** 2 + 4.0 * D * mv)
    )

    immobile = np.exp(-a ** 2 / (w_imm_m ** 2) * (xs ** 2 + ys ** 2))

    denom = _pos(n + n_imm) ** 2
    return float(offset) + 2.0 ** (-1.5) / denom * (n * mobile + n_imm * immobile)


def ics_gaussian_2d(
    line_shift: np.ndarray,
    pixel_shift: np.ndarray,
    amplitude: float = 1.0,
    pixel_size: float = 50.0,
    sigma_1: float = 200.0,
    sigma_2: float = 200.0,
    angle: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Anisotropic 2D-Gaussian spatial correlation (structure sizing).

    Port of PAM's ``2D_Gaussian_2Sigmas_Angle_Cor``: a pure spatial correlation
    (no diffusion/time) with two widths ``sigma_1``/``sigma_2`` [nm] and an
    orientation ``angle`` [rad], used to size elliptical structures / the PSF
    from an image auto-correlation. Distances use ``pixel_size`` [nm].
    """
    px = float(pixel_size)
    line_shift = np.asarray(line_shift, dtype=float)
    pixel_shift = np.asarray(pixel_shift, dtype=float)
    xr = pixel_shift * px - float(x_offset)
    yr = line_shift * px - float(y_offset)
    s1 = _pos(sigma_1, 1.0)
    s2 = _pos(sigma_2, 1.0)
    c, s = np.cos(float(angle)), np.sin(float(angle))
    u = (xr * c + yr * s) / s1
    v = (xr * s + yr * c) / s2
    return float(offset) + abs(float(amplitude)) * np.exp(-(u ** 2) - (v ** 2))


def rics_diffusion_triplet(
    line_shift: np.ndarray,
    pixel_shift: np.ndarray,
    n: float,
    diffusion_coefficient: float = 2.0,
    offset: float = 0.0,
    pixel_duration: float = 11.1,
    line_duration: float = 3.33,
    pixel_size: float = 40.0,
    w_r: float = 0.2,
    w_z: float = 1.0,
    tauT: float = 0.002,
    aT: float = 0.1,
) -> np.ndarray:
    """RICS model with one 3D diffusion component and triplet/blinking.

    Parameters largely follow the tttrlib example ``rics_diffusion_triplet``
    but with consistent SI conversions as in :func:`rics_simple`.

    - ``tauT`` triplet time in ms.
    - ``aT`` triplet amplitude (0–1).
    """
    n = _pos(n)
    D = abs(float(diffusion_coefficient)) * 1.0e-12
    tau_p = float(pixel_duration) * 1.0e-6
    tau_l = float(line_duration) * 1.0e-3
    a = float(pixel_size) * 1.0e-9
    w_r_m = _pos(w_r, 1e-3) * 1.0e-6
    w_z_m = _pos(w_z, 1e-3) * 1.0e-6
    tauT_s = float(tauT) * 1.0e-3

    line_shift = np.asarray(line_shift, dtype=float)
    pixel_shift = np.asarray(pixel_shift, dtype=float)

    mv = np.abs(pixel_shift * tau_p + line_shift * tau_l)

    triplet_factor = 1.0 + (aT / max(1.0 - aT, 1e-12)) * np.exp(-mv / max(tauT_s, 1e-12))

    denom_r = 1.0 + 4.0 * D * mv / (w_r_m ** 2)
    denom_z = 1.0 + 4.0 * D * mv / (w_z_m ** 2)

    spatial = np.exp(-a ** 2 * (pixel_shift ** 2 + line_shift ** 2) / (w_r_m ** 2 + 4.0 * D * mv))

    return (
        offset
        + 2.0 ** (-1.5)
        / (n + tauT_s) ** 2
        * (
            n
            * triplet_factor
            * denom_r ** (-1.0)
            * denom_z ** (-0.5)
            * spatial
        )
    )
