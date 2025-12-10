"""MCMC sampling utilities for the MaxEnt TCSPC lifetime / FRET plugin.

This module implements Q-based sampling of an existing MEM solution using
``emcee``. It operates purely on the result dictionary returned by the
MaxEnt solvers and contains no GUI/Qt code so it can be reused from
scripts, notebooks, or the plugin GUI.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import logging
import math
import sys
import numpy as np
import tables

try:  # optional dependency; checked at call time
    import emcee  # type: ignore
except Exception:  # pragma: no cover - handled lazily in the sampler
    emcee = None  # type: ignore[assignment]

from .core import MIN_PROB


logger = logging.getLogger(__name__)


def _log_prob_mem(
    u_vec: np.ndarray,
    p0: np.ndarray,
    H: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    nu_val: float,
    p0_sum: float,
) -> float:
    """Log-posterior for MEM Q-MCMC.

    Defined at module scope so that it can be pickled when using a
    multiprocessing pool with :class:`emcee.EnsembleSampler`.
    """

    u_arr = np.asarray(u_vec, dtype=float).ravel()
    if u_arr.size != p0.size:
        return float("-inf")

    p_raw = p0 * np.exp(u_arr)
    p = np.maximum(p_raw, float(MIN_PROB))
    s = float(np.sum(p))
    if (not np.isfinite(s)) or s <= 0.0:
        return float("-inf")

    p = p * (p0_sum / s)

    L = np.log(np.clip(p / m, a_min=1e-300, a_max=None))
    Hp = H @ p
    chisq_tilde = 0.5 * float(p @ Hp) - float(g0 @ p)
    S = (-L + 1.0) @ p - float(np.sum(m))
    Q_tilde = chisq_tilde - 0.5 * float(nu_val) * float(S)
    if not np.isfinite(Q_tilde):
        return float("-inf")

    return -0.5 * float(Q_tilde)


def _log_prob_mem_vectorized(
    u_vec: np.ndarray,
    p0: np.ndarray,
    H: np.ndarray,
    g0: np.ndarray,
    m: np.ndarray,
    nu_val: float,
    p0_sum: float,
) -> np.ndarray:
    u_arr = np.asarray(u_vec, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, -1)
    if u_arr.shape[1] != p0.size:
        return np.full(u_arr.shape[0], float("-inf"), dtype=float)

    p_raw = p0[None, :] * np.exp(u_arr)
    p = np.maximum(p_raw, float(MIN_PROB))
    s = np.sum(p, axis=1, keepdims=True)
    mask = (s > 0.0) & np.isfinite(s)
    p = p * (p0_sum / s)

    L = np.log(np.clip(p / m[None, :], a_min=1e-300, a_max=None))
    Hp = p @ H.T
    chisq_tilde = 0.5 * np.sum(p * Hp, axis=1) - np.dot(p, g0)
    S = np.sum((-L + 1.0) * p, axis=1) - float(np.sum(m))
    Q_tilde = chisq_tilde - 0.5 * float(nu_val) * S

    logprob = -0.5 * np.asarray(Q_tilde, dtype=float)
    bad = (~mask.ravel()) | (~np.isfinite(logprob))
    if np.any(bad):
        logprob[bad] = float("-inf")
    return logprob


# Progress callback signature: ``cb(done, total) -> bool`` where returning
# ``True`` requests cancellation.
ProgressCallback = Optional[Callable[[int, int], bool]]


def sample_mem_distribution_emcee(
    result: Mapping[str, Any],
    *,
    filename: Optional[str] = None,
    nwalkers: Optional[int] = None,
    steps_total: int = 500,
    thin: int = 5,
    substeps: int = 50,
    progress_cb: ProgressCallback = None,
    nprocs: Optional[int] = None,
    csv_prefix: Optional[str] = None,
    vectorized: Optional[bool] = None,
) -> Dict[str, Any]:
    """Sample the MEM distribution ``p`` using an ``emcee`` ensemble sampler.

    The log-posterior is defined from the MaxEnt objective ``Q = chi^2 - 0.5 * nu * S``
    (up to an additive constant). The state is the distribution ``p`` over the
    current axis (lifetime or distance) re-parameterised via a log-amplitude
    vector ``u`` to enforce positivity.

    Parameters
    ----------
    result:
        Result dictionary returned by the MaxEnt solvers. Must contain at least
        ``p``, ``H``, ``g0`` and either ``tau`` or ``R``. If ``prior`` is
        present it is used for the entropy term; otherwise a uniform prior is
        assumed.
    filename:
        Optional HDF5 file path. If provided, the sampling summary and samples
        are written using PyTables.
    nwalkers:
        Number of walkers. If ``None``, a heuristic is used based on the
        distribution dimension.
    steps_total:
        Total number of MCMC steps per walker.
    thin:
        Thinning factor passed to ``emcee.EnsembleSampler.run_mcmc``.
    substeps:
        Number of steps per chunk. After each chunk the progress callback is
        invoked.
    progress_cb:
        Optional callback ``cb(done, total) -> bool``. It is called with the
        number of completed steps and the total number of steps. If it
        returns ``True`` a ``RuntimeError('MEM sampling cancelled')`` is
        raised.
    nprocs:
        Optional number of worker processes for parallel sampling. If
        ``None`` or ``1``, sampling runs in the current process. If greater
        than ``1``, a ``multiprocessing.Pool`` is used internally.
    csv_prefix:
        Optional path *prefix* for writing TSV files with the thinned
        posterior samples in a format that can be loaded via
        :func:`ndxplorer.reader.read_csv_sampling`. If provided, one or more
        files with names like ``"{csv_prefix}_part01.tsv"`` are written,
        each containing rows = samples and columns = distribution bins
        ("p_0", "p_1", ...).

    Returns
    -------
    stats:
        Dictionary with keys:

        - ``axis``: lifetime or distance axis.
        - ``p_mem``: original MEM distribution.
        - ``p_mean``: posterior mean distribution.
        - ``p_lo`` / ``p_med`` / ``p_hi``: 16th / 50th / 84th percentiles.
        - ``n_samples``: number of posterior samples used for the summary.

        Additional keys may be added in the future.
    """

    if emcee is None:
        raise RuntimeError(
            "The 'emcee' package is required for MEM Q-MCMC sampling. "
            "Install 'emcee' (e.g. via the ChiSurf conda environment) to "
            "enable this feature."
        )

    if vectorized is None:
        try:
            is_win = sys.platform.startswith("win")
        except Exception:
            is_win = False
        vectorized = bool(is_win)

    # Axis and base distribution
    try:
        dist_axis = np.asarray(
            result.get("R", result.get("tau")), dtype=float
        ).ravel()
    except Exception:
        dist_axis = np.zeros(0, dtype=float)

    p0 = np.asarray(result.get("p", []), dtype=float).ravel()
    H = np.asarray(result.get("H", []), dtype=float)
    g0 = np.asarray(result.get("g0", []), dtype=float).ravel()

    if p0.size == 0 or dist_axis.size == 0:
        raise RuntimeError("MEM result does not contain a valid distribution")

    if H.shape != (p0.size, p0.size) or g0.size != p0.size:
        raise RuntimeError("MEM result lacks H/g0 with matching shape for sampling")

    # Prior m used in the entropy term; fall back to uniform if missing.
    prior_arr = result.get("prior", None)
    if prior_arr is None:
        m = np.ones_like(p0, dtype=float)
    else:
        m = np.asarray(prior_arr, dtype=float).ravel()
    if m.size != p0.size:
        m = np.ones_like(p0, dtype=float)
    m = np.clip(m, float(MIN_PROB), np.inf)
    m /= float(np.sum(m))

    nu_val = float(result.get("nu", result.get("nu_input", 0.0)))

    ndim = int(p0.size)
    if ndim <= 0:
        raise RuntimeError("MEM distribution has zero length")

    # emcee's red-blue moves require at least ``2 * ndim`` walkers. For MEM
    # grids this can be sizable, but still tractable for the default
    # ``steps_total``. We therefore always enforce this lower bound.
    if nwalkers is None:
        nwalkers = int(max(2 * ndim, 16))
    else:
        nwalkers = int(max(2 * ndim, nwalkers, 4))

    steps_total = int(max(1, steps_total))
    thin = int(max(1, thin))
    substeps = int(max(1, min(substeps, steps_total)))
    large_steps = max(1, steps_total // substeps)

    p0_sum = float(np.sum(p0)) if p0.size else 1.0

    pool = None
    if (not vectorized) and nprocs is not None and nprocs > 1:
        try:
            import multiprocessing as mp

            cpu_total = mp.cpu_count() or 1
            nprocs_eff = int(max(1, min(int(nprocs), cpu_total)))
            if nprocs_eff > 1:
                try:
                    ctx = mp.get_context("spawn")
                except Exception:
                    ctx = mp
                pool = ctx.Pool(processes=nprocs_eff)
        except Exception as exc:
            logger.warning(
                "MEM sampling: failed to create multiprocessing pool (%s); "
                "falling back to single process",
                exc,
            )
            pool = None

    log_prob_args = (p0, H, g0, m, float(nu_val), float(p0_sum))

    try:
        if vectorized:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                _log_prob_mem_vectorized,
                args=log_prob_args,
                pool=None,
                vectorize=True,
            )
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                _log_prob_mem,
                args=log_prob_args,
                pool=pool,
            )

        scale = 1e-2
        previous_state = scale * np.random.randn(nwalkers, ndim)

        saved_steps = 0
        for i in range(large_steps):
            steps_here = min(substeps, steps_total - saved_steps)
            if steps_here <= 0:
                break

            try:
                previous_state = sampler.run_mcmc(
                    previous_state,
                    nsteps=steps_here,
                    thin_by=thin,
                    skip_initial_state_check=(i > 0),
                    tune=True,
                )
            except ValueError as exc:
                msg = str(exc)
                if "Initial state has a large condition number" in msg:
                    logger.warning(
                        "emcee warning during MEM sampling: %s; "
                        "continuing with skip_initial_state_check=True",
                        msg,
                    )
                    previous_state = sampler.run_mcmc(
                        previous_state,
                        nsteps=steps_here,
                        thin_by=thin,
                        skip_initial_state_check=True,
                        tune=True,
                    )
                else:
                    raise

            saved_steps += steps_here
            if progress_cb is not None:
                try:
                    cancel = bool(progress_cb(saved_steps, steps_total))
                except Exception:
                    cancel = False
                if cancel:
                    raise RuntimeError("MEM sampling cancelled")

        chain = sampler.flatchain
    finally:
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass
    if chain.ndim != 2 or chain.shape[0] == 0:
        raise RuntimeError("No MCMC samples collected")

    u_chain = np.asarray(chain, dtype=float)
    p_samples = p0[None, :] * np.exp(u_chain)
    p_samples = np.maximum(p_samples, float(MIN_PROB))

    s_all = np.sum(p_samples, axis=1, keepdims=True)
    mask = (s_all > 0.0) & np.isfinite(s_all)
    mask = np.asarray(mask, dtype=bool).ravel()
    if not np.any(mask):
        raise RuntimeError("All MCMC samples were invalid")

    p_valid = p_samples[mask]
    s_valid = s_all[mask]
    p_valid = p_valid * (p0_sum / s_valid)

    n_total = int(p_valid.shape[0])
    if n_total > 10:
        p_used = p_valid[n_total // 2 :, :]
    else:
        p_used = p_valid

    p_mean = np.mean(p_used, axis=0)
    p_lo, p_med, p_hi = np.percentile(p_used, [16.0, 50.0, 84.0], axis=0)

    stats: Dict[str, Any] = {
        "axis": dist_axis,
        "p_mem": p0,
        "p_mean": p_mean,
        "p_lo": p_lo,
        "p_med": p_med,
        "p_hi": p_hi,
        "n_samples": int(p_used.shape[0]),
        "nwalkers": int(nwalkers),
        "steps_total": int(steps_total),
        "thin": int(thin),
        "substeps": int(substeps),
        "ndim": int(ndim),
        "vectorized": bool(vectorized),
    }

    # Optional: write smaller TSV files for ndxplorer/ChiSurf sampling loader.
    if csv_prefix is not None:
        try:
            _write_sampling_tsv_stack(str(csv_prefix), p_used)
        except Exception as exc:  # pragma: no cover - best-effort export
            logger.warning("MEM sampling: failed to write TSV stack: %s", exc)

    if filename is not None:
        filters = tables.Filters(complib="zlib", shuffle=True, complevel=1)
        with tables.open_file(str(filename), mode="w", title="MEM sampling", filters=filters) as h5:
            root = h5.root
            h5.create_array(root, "axis", dist_axis, "Lifetime or distance axis")
            h5.create_array(root, "p_mem", p0, "Original MEM distribution")
            h5.create_array(root, "p_mean", p_mean, "Posterior mean distribution")
            h5.create_array(root, "p_lo", p_lo, "Lower credible band (16th percentile)")
            h5.create_array(root, "p_med", p_med, "Median distribution (50th percentile)")
            h5.create_array(root, "p_hi", p_hi, "Upper credible band (84th percentile)")
            h5.create_array(root, "p_samples", p_used, "MCMC samples (thinned)")

            attrs = root._v_attrs
            attrs.mode = "FRET" if "R" in result else "lifetime"
            attrs.nu = float(nu_val)
            attrs.nwalkers = int(nwalkers)
            attrs.steps_total = int(steps_total)
            attrs.thin = int(thin)
            attrs.ndim = int(ndim)
            attrs.sum_p = float(p0_sum)
            attrs.n_samples = int(p_used.shape[0])

    return stats


def _write_sampling_tsv_stack(prefix: str, samples: np.ndarray, max_cols: int = 64) -> None:
    """Write posterior samples to one or more TSV files for ndxplorer.

    Parameters
    ----------
    prefix:
        File name prefix. Files ``"{prefix}.tsv"`` or
        ``"{prefix}_partNN.tsv"`` will be created.
    samples:
        Array of shape (n_samples, ndim) containing the thinned posterior
        samples over the MEM distribution.
    max_cols:
        Maximum number of columns per file. Higher values create fewer but
        wider files; lower values create more, narrower files.
    """

    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return

    n_rows, ndim = arr.shape
    max_cols = int(max(1, max_cols))

    if ndim <= max_cols:
        groups = [(0, ndim)]
    else:
        n_parts = int(math.ceil(ndim / max_cols))
        groups = []
        for k in range(n_parts):
            start = k * max_cols
            stop = min(ndim, (k + 1) * max_cols)
            groups.append((start, stop))

    for idx, (start, stop) in enumerate(groups):
        cols = np.arange(start, stop, dtype=int)
        data = arr[:, cols]
        headers = [f"p_{int(j)}" for j in cols]
        header_line = "\t".join(headers)

        if len(groups) == 1:
            fn = f"{prefix}.tsv"
        else:
            fn = f"{prefix}_part{idx + 1:02d}.tsv"

        try:
            np.savetxt(fn, data, delimiter="\t", header=header_line, comments="")
        except Exception as exc:  # pragma: no cover - best-effort export
            logger.warning("MEM sampling: failed to write TSV '%s': %s", fn, exc)


__all__ = ["sample_mem_distribution_emcee", "ProgressCallback"]
