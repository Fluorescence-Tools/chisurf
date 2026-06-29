"""Pure k² distribution computation functions.

No Qt, no ZMQ, no GUI dependencies — pure math only.
Returns JSON-safe dicts (lists instead of NumPy arrays).
"""

from __future__ import annotations

import numpy as np

from chisurf.core.fluorescence.anisotropy.kappa2 import (
    kappasq_all,
    kappasq_all_delta,
    kappasq_dwt,
    p_isotropic_orientation_factor,
    s2delta,
)


def compute_kappa2_dist(**params: float | bool | str) -> dict:
    """Compute the k² orientation-factor distribution.

    Parameters
    ----------
    **params
        model_type : str
            ``"cone"`` (WIC), ``"diffusion"`` (DWT), or ``"isotropic"``.
        r_0 : float
            Fundamental anisotropy of the dye (default 0.380).
        r_Dinf : float
            Residual anisotropy of the donor (default 0.050).
        r_Ainf : float
            Residual anisotropy of direct-excited acceptor (default 0.100).
        r_ADinf : float
            Residual anisotropy of FRET-sensitised acceptor (default 0.005).
        kappa2_true : float
            Assumed orientation factor for R_app / R_DA (default 0.667).
        fret_efficiency : float
            FRET efficiency used for the DWT model (default 0.001).
        step : float
            Angular step size in degrees for WIC grid (default 1.5).
        n_bins : int
            Number of bins in the k² histogram (default 131).
        rAD_known : bool
            When True, use the δ angle computed from residual anisotropies
            (default False).

    Returns
    -------
    dict
        JSON-safe result with keys ``k2_scale``, ``k2_hist``, ``k2_values``,
        ``k2_mean``, ``k2_sd``, ``Rapp_mean``, ``RappSD``, ``delta_deg``,
        ``SD2``, ``SA2``, ``delta``.
    """
    model_type = str(params.get("model_type", "cone"))
    r_0 = float(params.get("r_0", 0.380))
    r_Dinf = float(params.get("r_Dinf", 0.050))
    r_Ainf = float(params.get("r_Ainf", 0.100))
    r_ADinf = float(params.get("r_ADinf", 0.005))
    kappa2_true = float(params.get("kappa2_true", 0.667))
    fret_efficiency = float(params.get("fret_efficiency", 0.001))
    step = float(params.get("step", 1.5))
    n_bins = max(int(params.get("n_bins", 131)), 5)
    rAD_known = bool(params.get("rAD_known", False))

    r0_safe = max(r_0, 1e-10)
    sd2 = float(-np.sqrt(max(r_Dinf / r0_safe, 0.0)))
    sa2 = float(np.sqrt(max(r_Ainf / r0_safe, 0.0)))

    _, delta = s2delta(
        s2_donor=sd2,
        s2_acceptor=sa2,
        r_inf_AD=r_ADinf,
        r_0=r_0,
    )

    if model_type == "cone":
        if rAD_known:
            x, k2hist, k2v = kappasq_all_delta(
                delta=delta,
                sD2=sd2,
                sA2=sa2,
                step=step,
                n_bins=n_bins,
            )
        else:
            x, k2hist, k2v = kappasq_all(sd2, sa2, n_bins=n_bins)
    elif model_type == "diffusion":
        x, k2hist, k2v = kappasq_dwt(
            sD2=sd2,
            sA2=sa2,
            fret_efficiency=fret_efficiency,
            n_bins=n_bins,
        )
    elif model_type == "isotropic":
        k2_edges = np.linspace(0.0, 4.0, n_bins)
        k2_centers = 0.5 * (k2_edges[1:] + k2_edges[:-1])
        k2hist = p_isotropic_orientation_factor(k2_centers, normalize=True)
        x = k2_edges
        k2v = k2_centers
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    k2c = x[1:]
    total = max(float(np.sum(k2hist)), 1e-30)
    k2_mean = float(np.dot(k2hist, k2c) / total)
    k2_sd = float(np.sqrt(np.dot(k2hist, (k2c - k2_mean) ** 2) / total))

    k2t = max(kappa2_true, 1e-10)
    r_scale = (k2c / k2t) ** (1.0 / 6.0)
    Rapp_mean = float(np.dot(k2hist, r_scale) / total)
    RappSD = float(np.sqrt(np.dot(k2hist, (r_scale - Rapp_mean) ** 2) / total))

    delta_deg = float(delta * 180.0 / np.pi)

    k2v_list: list = k2v.tolist() if isinstance(k2v, np.ndarray) else list(k2v)  # type: ignore[union-attr]

    return {
        "k2_scale": x.tolist(),
        "k2_hist": k2hist.tolist(),
        "k2_values": k2v_list,
        "k2_mean": k2_mean,
        "k2_sd": k2_sd,
        "Rapp_mean": Rapp_mean,
        "RappSD": RappSD,
        "delta_deg": delta_deg,
        "SD2": sd2,
        "SA2": sa2,
        "delta": float(delta),
    }
