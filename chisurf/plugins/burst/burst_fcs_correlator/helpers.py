from __future__ import annotations

import pathlib
from typing import List, Tuple

import numpy as np
import tttrlib

from chisurf.core.models.fcs.maxent import fcs_maxent


# ----------------------------------------------------------------------
# Parsing helpers for BUR/BST and TTTR files
# ----------------------------------------------------------------------


def parse_bst_file(path: pathlib.Path) -> Tuple[pathlib.Path | None, List[Tuple[int, int]]]:
    """Return (tttr_path, list_of_(start, end)) for a Burst-ID .bst file.

    The underlying TTTR file is searched in the .bst folder and up to
    three parent folders by stripping the trailing ``.bst`` from the
    filename, following the logic used in WizardTTTRCorrelator.
    """

    if not path.exists() or not path.is_file():
        return None, []

    base_with_ext = path.name[:-4]  # strip '.bst'
    candidates = [path.parent]
    try:
        if path.parent.parent:
            candidates.append(path.parent.parent)
        if path.parent.parent.parent:
            candidates.append(path.parent.parent.parent)
        if path.parent.parent.parent.parent:
            candidates.append(path.parent.parent.parent.parent)
    except Exception:
        pass

    tttr_path: pathlib.Path | None = None
    for folder in candidates:
        cand = folder / base_with_ext
        if cand.exists() and cand.is_file():
            tttr_path = cand
            break

    ranges: List[Tuple[int, int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    s = int(float(parts[0]))
                    e = int(float(parts[1]))
                    if e >= s:
                        ranges.append((s, e))
                except Exception:
                    continue
    except Exception:
        ranges = []

    return tttr_path, ranges


def parse_bur_file(path: pathlib.Path, analysis_root: pathlib.Path) -> Tuple[pathlib.Path | None, List[Tuple[int, int]]]:
    """Return (tttr_path, list_of_(start, end)) for a BUR file."""

    if not path.exists() or not path.is_file():
        return None, []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip()
            if not header:
                return None, []
            cols = [c.strip() for c in header.split("\t") if c.strip()]
            name_to_idx = {c.lower(): i for i, c in enumerate(cols)}

            def _idx(label: str) -> int | None:
                return name_to_idx.get(label.lower(), None)

            idx_first_photon = _idx("first photon")
            idx_last_photon = _idx("last photon")
            idx_first_file = _idx("first file")
            if idx_first_photon is None or idx_last_photon is None or idx_first_file is None:
                return None, []

            ranges: List[Tuple[int, int]] = []
            first_file_name: str | None = None

            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                max_idx = max(idx_first_photon, idx_last_photon, idx_first_file)
                if len(parts) <= max_idx:
                    continue
                s_txt = parts[idx_first_photon].strip()
                e_txt = parts[idx_last_photon].strip()
                f_txt = parts[idx_first_file].strip()
                if not s_txt or not e_txt:
                    continue
                try:
                    s = int(float(s_txt))
                    e = int(float(e_txt))
                except Exception:
                    continue
                if s == 0 and e == 0:
                    continue
                if e < s:
                    continue
                ranges.append((s, e))
                if first_file_name is None and f_txt:
                    first_file_name = f_txt
    except Exception:
        return None, []

    if not ranges or not first_file_name:
        return None, []

    tttr_path: pathlib.Path | None = None
    name_path = pathlib.Path(first_file_name)
    if name_path.is_absolute() and name_path.exists():
        tttr_path = name_path
    else:
        base = analysis_root.parent if analysis_root is not None else path.parent
        candidates: List[pathlib.Path] = []
        try:
            if base is not None:
                candidates.append(base)
                if base.parent is not None:
                    candidates.append(base.parent)
                    if base.parent.parent is not None:
                        candidates.append(base.parent.parent)
        except Exception:
            pass
        if analysis_root is not None:
            candidates.append(analysis_root)
        for folder in candidates:
            try:
                cand = folder / name_path.name
            except Exception:
                continue
            if cand.exists() and cand.is_file():
                tttr_path = cand
                break

    return tttr_path, ranges


def open_tttr(path: pathlib.Path, filetype) -> tttrlib.TTTR | None:
    """Open a TTTR file with simple, extension-aware fallback logic."""

    p_str = path.as_posix()
    ext = path.suffix.lower()
    try:
        if ext == ".spc":
            # Prefer inference for SPC containers
            try:
                ft_int = tttrlib.inferTTTRFileType(p_str)
                if ft_int is not None and ft_int >= 0:
                    return tttrlib.TTTR(p_str, ft_int)
            except Exception:
                pass
            try:
                return tttrlib.TTTR(p_str, "SPC")
            except Exception:
                return tttrlib.TTTR(p_str)
        if isinstance(filetype, str) and filetype.strip():
            try:
                return tttrlib.TTTR(p_str, filetype)
            except Exception:
                pass
        try:
            ft_int = tttrlib.inferTTTRFileType(p_str)
            if ft_int is not None and ft_int >= 0:
                return tttrlib.TTTR(p_str, ft_int)
        except Exception:
            pass
        return tttrlib.TTTR(p_str)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Correlation and fitting helpers
# ----------------------------------------------------------------------


def parse_channel_list(text: str) -> List[int]:
    """Parse a comma / semicolon separated channel list into ints."""

    vals: List[int] = []
    if not text:
        return vals
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except Exception:
            continue
    return vals


def correlate_single_burst(
    tttr: tttrlib.TTTR,
    chs_a: List[int],
    chs_b: List[int],
    micro_a,
    micro_b,
    n_bins: int,
    n_casc: int,
    make_fine: bool,
):
    """Compute a single FCS correlation curve for one burst.

    Returns (tau_ms, G(tau)) or (None, None) if the burst does not
    contain enough photons in either channel.
    """

    try:
        t = tttr.macro_times
    except Exception:
        return None, None
    if t is None or len(t) == 0:
        return None, None

    mask_a = tttrlib.TTTRMask()
    mask_b = tttrlib.TTTRMask()
    mask_a.select_channels(tttr, chs_a, mask=True)
    mask_b.select_channels(tttr, chs_b, mask=True)
    m_a = mask_a.mask.astype(bool)
    m_b = mask_b.mask.astype(bool)

    if micro_a:
        mask_mt_a = tttrlib.TTTRMask()
        mask_mt_a.select_microtime_ranges(tttr, micro_a)
        mask_mt_a.flip()
        m_a = np.logical_and(m_a, mask_mt_a.mask.astype(bool))
    if micro_b:
        mask_mt_b = tttrlib.TTTRMask()
        mask_mt_b.select_microtime_ranges(tttr, micro_b)
        mask_mt_b.flip()
        m_b = np.logical_and(m_b, mask_mt_b.mask.astype(bool))

    w1 = np.array(m_a, dtype=np.float64)
    w2 = np.array(m_b, dtype=np.float64)
    if w1.sum() <= 0.0 or w2.sum() <= 0.0:
        return None, None

    try:
        dT_ms = tttr.header.macro_time_resolution * 1000.0
    except Exception:
        dT_ms = 1.0

    settings = {
        "n_bins": int(n_bins),
        "n_casc": int(n_casc),
        "make_fine": bool(make_fine),
    }

    correlator = tttrlib.Correlator(**settings)
    correlator.set_macrotimes(t, t)
    correlator.set_weights(w1, w2)
    tau = correlator.x_axis * dT_ms
    if make_fine:
        try:
            n_mt = tttr.get_number_of_micro_time_channels()
            mt = tttr.micro_times
            correlator.set_microtimes(mt, mt, n_mt)
            # convert to micro-time channels (matching tttr_correlator)
            tau = tau / (tttr.header.micro_time_resolution / 1000.0)
        except Exception:
            pass
    g = correlator.correlation
    return np.asarray(tau, dtype=float), np.asarray(g, dtype=float)


def fit_diffusion_time(tau: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    """Fit a MaxEnt diffusion-time distribution and summarize it.

    Returns (td_mean_ms, td_peak_ms). On fit failure, both are NaN.
    """

    mask = np.isfinite(tau) & np.isfinite(g) & (tau > 0)
    tau = tau[mask]
    g = g[mask]
    if tau.size < 5:
        return float("nan"), float("nan")
    try:
        result = fcs_maxent(tau=tau, g=g)
    except Exception:
        return float("nan"), float("nan")
    td_grid = np.asarray(result.get("td_grid", []), dtype=float)
    p = np.asarray(result.get("p", []), dtype=float)
    if td_grid.size == 0 or p.size == 0:
        return float("nan"), float("nan")
    p = np.clip(p, 0.0, np.inf)
    if not np.any(p > 0.0):
        td_peak = float(td_grid[np.argmax(p)])
        return float("nan"), td_peak
    td_mean = float(np.sum(td_grid * p) / np.sum(p))
    td_peak = float(td_grid[np.argmax(p)])
    return td_mean, td_peak


def fit_simple_diffusion(
    tau: np.ndarray,
    g: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate single-component diffusion time with a simple 3D Gaussian FCS model.

    Returns (td_ms, tau_used, g_used, g_fit).
    On failure, returns (NaN, empty arrays).
    """

    tau = np.asarray(tau, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    mask = np.isfinite(tau) & np.isfinite(g) & (tau > 0)
    tau = tau[mask]
    g = g[mask]
    if tau.size < 5:
        return float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    s = 3.5

    def _shape(td_val: float) -> np.ndarray:
        x = tau / td_val
        return (1.0 / (1.0 + x)) / np.sqrt(1.0 + x / (s ** 2))

    try:
        order = np.argsort(tau)
        tau = tau[order]
        g = g[order]
    except Exception:
        pass

    try:
        tau_min = float(np.min(tau))
        tau_max = float(np.max(tau))
    except Exception:
        return float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    if not np.isfinite(tau_min) or not np.isfinite(tau_max) or tau_min <= 0.0:
        return float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    td_min = max(tau_min * 0.1, tau_min * 1e-2, 1e-6)
    td_max = max(tau_max * 10.0, td_min * 1.1)

    try:
        td_grid = np.logspace(np.log10(td_min), np.log10(td_max), 60)
    except Exception:
        return float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    best_td = float("nan")
    best_sse = np.inf
    best_g_fit: np.ndarray | None = None

    for td_val in td_grid:
        try:
            f = _shape(float(td_val))
            X = np.column_stack([np.ones_like(f), f])
            beta, _, _, _ = np.linalg.lstsq(X, g, rcond=None)
            b_val = float(beta[0])
            a_val = float(beta[1])
            g_model = b_val + a_val * f
            resid = g - g_model
            sse = float(np.sum(resid * resid))
        except Exception:
            continue
        if not np.isfinite(sse):
            continue
        if sse < best_sse:
            best_sse = sse
            best_td = float(td_val)
            best_g_fit = g_model

    if not np.isfinite(best_td):
        return float("nan"), np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    if best_g_fit is None:
        return best_td, tau, g, np.asarray([], dtype=float)
    return best_td, tau, g, np.asarray(best_g_fit, dtype=float)
