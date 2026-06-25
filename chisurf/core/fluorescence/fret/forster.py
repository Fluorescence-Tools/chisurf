"""Förster radius from spectral overlap (PRD-06 Task 1).

The canonical, grid-agnostic implementation of the spectral overlap integral and the
Förster radius R0 — the primitive behind "look up a donor/acceptor pair and get R0
computed from the overlap", consumed by the fluorophore database (PRD-06) and the FRET
models. The Light Path Simulator plugin has a grid-specific variant
(``lightpath_simulator/backend/crosstalk.py``) that can delegate here.

Definitions (Lakowicz):

    J = ∫ f_D(λ) · ε_A(λ) · λ⁴ dλ            [M⁻¹ cm⁻¹ nm⁴]
    R0 = 0.02108 · (κ² · Q_D · n⁻⁴ · J)^(1/6)  [nm]   (×10 → Å)

where ``f_D`` is the **area-normalized** donor emission (∫ f_D dλ = 1), ``ε_A`` the
acceptor molar extinction coefficient, ``κ²`` the orientation factor (2/3 for freely
rotating dyes), ``Q_D`` the donor quantum yield, and ``n`` the medium refractive index.
"""

from __future__ import annotations

import numpy as np

#: Prefactor giving R0 in nm when J is expressed in M⁻¹ cm⁻¹ nm⁴ (Lakowicz).
_R0_PREFACTOR_NM = 0.02108


def overlap_integral(
    wavelength_nm: np.ndarray,
    donor_emission: np.ndarray,
    acceptor_extinction: np.ndarray,
) -> float:
    """Spectral overlap integral ``J`` in M⁻¹ cm⁻¹ nm⁴.

    The donor emission is area-normalized internally, so its absolute scale is
    irrelevant. ``acceptor_extinction`` must be the molar extinction coefficient
    ε_A(λ) in M⁻¹ cm⁻¹ (i.e. already scaled by ε_max, not peak-normalized).

    Raises ``ValueError`` if the inputs are mismatched or the donor emission has
    non-positive area (no emission to overlap).
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    f_d = np.asarray(donor_emission, dtype=float)
    eps_a = np.asarray(acceptor_extinction, dtype=float)
    if not (wl.shape == f_d.shape == eps_a.shape) or wl.ndim != 1 or wl.size < 2:
        raise ValueError(
            "wavelength_nm, donor_emission and acceptor_extinction must be 1-D arrays "
            "of the same length (>= 2)"
        )
    area = np.trapz(f_d, wl)
    if area <= 0:
        raise ValueError("donor emission spectrum has non-positive area")
    f_d_norm = f_d / area
    return float(np.trapz(f_d_norm * eps_a * wl**4, wl))


def forster_radius(
    overlap_J: float,
    *,
    donor_quantum_yield: float,
    kappa2: float = 2.0 / 3.0,
    refractive_index: float = 1.33,
) -> float:
    """Förster radius R0 in Ångström from the overlap integral ``J``.

    ``overlap_J`` is in M⁻¹ cm⁻¹ nm⁴ (see :func:`overlap_integral`). Raises
    ``ValueError`` on non-physical inputs (negative J/Q_D, non-positive n).
    """
    if overlap_J < 0:
        raise ValueError("overlap integral J must be non-negative")
    if donor_quantum_yield < 0:
        raise ValueError("donor_quantum_yield must be non-negative")
    if refractive_index <= 0:
        raise ValueError("refractive_index must be positive")
    r0_nm = _R0_PREFACTOR_NM * (
        kappa2 * donor_quantum_yield * refractive_index**-4 * overlap_J
    ) ** (1.0 / 6.0)
    return r0_nm * 10.0


def forster_radius_from_spectra(
    wavelength_nm: np.ndarray,
    donor_emission: np.ndarray,
    acceptor_extinction: np.ndarray,
    *,
    donor_quantum_yield: float,
    kappa2: float = 2.0 / 3.0,
    refractive_index: float = 1.33,
) -> tuple[float, float]:
    """Convenience: overlap + R0 in one call.

    Returns ``(R0_angstrom, overlap_J)``.
    """
    overlap_J = overlap_integral(wavelength_nm, donor_emission, acceptor_extinction)
    r0 = forster_radius(
        overlap_J,
        donor_quantum_yield=donor_quantum_yield,
        kappa2=kappa2,
        refractive_index=refractive_index,
    )
    return r0, overlap_J
