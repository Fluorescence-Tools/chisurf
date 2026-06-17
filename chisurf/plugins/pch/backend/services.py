from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_METHOD_LOAD_TTTR = "pch.load_tttr"
_METHOD_COMPUTE = "pch.compute"
_METHOD_FIT = "pch.fit"


def register_services(dispatcher: Any) -> None:
    dispatcher.register(
        _METHOD_LOAD_TTTR,
        lambda params: _load_tttr_handler(**params),
    )
    dispatcher.register(
        _METHOD_COMPUTE,
        lambda params: _compute_handler(**params),
    )
    dispatcher.register(
        _METHOD_FIT,
        lambda params: _fit_handler(**params),
    )


def _load_tttr_handler(
    filename: str,
    channels: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        import tttrlib
        tttr = tttrlib.TTTR(filename)
        routing = sorted(set(int(c) for c in tttr.routing_channels))
        return {
            "ok": True,
            "result": {
                "filename": filename,
                "n_photons": int(len(tttr)),
                "routing_channels": routing,
                "macro_time_resolution": float(tttr.header.macro_time_resolution),
                "micro_time_range": [0, 65535],
                "has_micro_times": True,
            },
        }
    except Exception as exc:
        logger.exception("Failed to load TTTR file")
        return {"ok": False, "error": str(exc)}


def _compute_handler(
    filename: str,
    channels: list[int] | None = None,
    bin_time_us: float = 100.0,
    micro_time_min: int = 0,
    micro_time_max: int = 65535,
    reading_routine: str = "PTU",
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        import tttrlib
        tttr = tttrlib.TTTR(filename)
        channels = channels or [0, 2]
        mask = np.isin(tttr.routing_channels, channels)
        bin_t = bin_time_us * 1e-6
        masks_mt = (tttr.micro_times >= micro_time_min) & (
            tttr.micro_times <= micro_time_max
        )
        combined_mask = mask & masks_mt
        times = tttr.macro_times[combined_mask] * tttr.header.macro_time_resolution
        t_max = times.max()
        n_bins = int(np.ceil(t_max / bin_t))
        range_end = bin_t * n_bins
        counts, edges = np.histogram(
            times, bins=n_bins, range=(0, range_end)
        )
        tcent = ((edges[:-1] + edges[1:]) / 2).tolist()
        hist_counts = np.bincount(counts, minlength=int(counts.max()) + 1)
        total_bins = int(counts.size)
        k_vals = np.arange(hist_counts.size, dtype=float)
        p_exp = (hist_counts / total_bins).tolist()
        return {
            "ok": True,
            "result": {
                "k_vals": k_vals.tolist(),
                "p_exp": p_exp,
                "hist_counts": hist_counts.tolist(),
                "total_bins": total_bins,
                "trace_t": tcent,
                "trace_counts": counts.tolist(),
                "settings": {
                    "channels": channels,
                    "bin_time_us": bin_time_us,
                    "micro_time_min": micro_time_min,
                    "micro_time_max": micro_time_max,
                },
            },
        }
    except Exception as exc:
        logger.exception("Failed to compute PCH")
        return {"ok": False, "error": str(exc)}


def _fit_handler(
    k_vals: list[float],
    p_exp: list[float],
    hist_counts: list[int] | None = None,
    total_bins: int | None = None,
    n_components: int = 1,
    initial_epsilons: list[float] | None = None,
    initial_Ns: list[float] | None = None,
    fit_low: int = 0,
    fit_high: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        from scipy.optimize import least_squares

        from ..api.algorithms import pch_mixture

        k = np.array(k_vals, dtype=float)
        pe = np.array(p_exp, dtype=float)
        hc = (
            np.array(hist_counts, dtype=float)
            if hist_counts is not None
            else pe * (total_bins or 1)
        )
        tb = total_bins or 1
        fit_high = fit_high if fit_high is not None else int(k[-1])
        init_eps = (
            initial_epsilons
            if initial_epsilons
            else [2.0] * n_components
        )
        init_Ns = (
            initial_Ns if initial_Ns else [3.0] * n_components
        )
        mask_reg = (k >= fit_low) & (k <= fit_high)
        params_init = init_eps + init_Ns

        def resid(p):
            pmod = pch_mixture(k, p[:n_components], p[n_components:])
            pmod /= pmod.sum()
            return pmod[mask_reg] - pe[mask_reg]

        res = least_squares(resid, np.array(params_init), bounds=(0, np.inf))
        epsilons = res.x[:n_components].tolist()
        avg_Ns = res.x[n_components:].tolist()
        Ns_arr = np.array(avg_Ns)
        fractions = (Ns_arr / Ns_arr.sum() * 100.0).tolist()
        p_fit = pch_mixture(k, res.x[:n_components], res.x[n_components:])
        p_fit /= p_fit.sum()
        exp_cnt = p_fit * tb
        mask_chi = mask_reg & (exp_cnt > 0)
        chi2 = float(np.sum((hc[mask_chi] - exp_cnt[mask_chi]) ** 2 / exp_cnt[mask_chi]))
        dof = int(mask_chi.sum()) - (n_components * 2)
        red_chi2 = chi2 / dof if dof > 0 else float("nan")
        return {
            "ok": True,
            "result": {
                "epsilons": epsilons,
                "avg_Ns": avg_Ns,
                "fractions": fractions,
                "chi2": chi2,
                "reduced_chi2": red_chi2,
                "dof": dof,
                "fit_low": fit_low,
                "fit_high": fit_high,
                "p_fit": p_fit.tolist(),
                "n_components": n_components,
            },
        }
    except Exception as exc:
        logger.exception("Failed to fit PCH")
        return {"ok": False, "error": str(exc)}
