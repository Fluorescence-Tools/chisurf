"""Qt-free core for the burst-wise FCS correlator.

Everything here is pure computation (numpy + tttrlib + the ChiSurf MaxEnt
model) with no GUI dependency, so it can be driven from the backend services /
RPC layer as well as directly from the GUI. The parsing/correlation/fitting
primitives were lifted verbatim from the old ``helpers.py`` (which now re-exports
from here) and wrapped with declarative settings and a file-level orchestrator.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tttrlib

from chisurf.core.models.fcs.maxent import fcs_maxent


# ----------------------------------------------------------------------
# Parsing helpers for BUR/BST and TTTR files
# ----------------------------------------------------------------------


def parse_bst_file(path: pathlib.Path) -> Tuple[Optional[pathlib.Path], List[Tuple[int, int]]]:
    """Return ``(tttr_path, [(start, end), ...])`` for a Burst-ID ``.bst`` file.

    The underlying TTTR file is searched in the ``.bst`` folder and up to three
    parent folders by stripping the trailing ``.bst`` from the filename.
    """
    path = pathlib.Path(path)
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

    tttr_path: Optional[pathlib.Path] = None
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


def parse_bur_file(
    path: pathlib.Path, analysis_root: pathlib.Path
) -> Tuple[Optional[pathlib.Path], List[Tuple[int, int]]]:
    """Return ``(tttr_path, [(start, end), ...])`` for a BUR file."""
    path = pathlib.Path(path)
    if not path.exists() or not path.is_file():
        return None, []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip()
            if not header:
                return None, []
            cols = [c.strip() for c in header.split("\t") if c.strip()]
            name_to_idx = {c.lower(): i for i, c in enumerate(cols)}

            def _idx(label: str) -> Optional[int]:
                return name_to_idx.get(label.lower(), None)

            idx_first_photon = _idx("first photon")
            idx_last_photon = _idx("last photon")
            idx_first_file = _idx("first file")
            if idx_first_photon is None or idx_last_photon is None or idx_first_file is None:
                return None, []

            ranges: List[Tuple[int, int]] = []
            first_file_name: Optional[str] = None

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

    tttr_path: Optional[pathlib.Path] = None
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


def open_tttr(path: pathlib.Path, filetype=None) -> Optional["tttrlib.TTTR"]:
    """Open a TTTR file with simple, extension-aware fallback logic."""
    path = pathlib.Path(path)
    p_str = path.as_posix()
    ext = path.suffix.lower()
    try:
        if ext == ".spc":
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


def parse_channel_list(text: str) -> List[int]:
    """Parse a comma / semicolon separated channel list into ints."""
    vals: List[int] = []
    if not text:
        return vals
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except Exception:
            continue
    return vals


# ----------------------------------------------------------------------
# Correlation and fitting primitives
# ----------------------------------------------------------------------


def correlate_single_burst(
    tttr: "tttrlib.TTTR",
    chs_a: List[int],
    chs_b: List[int],
    micro_a,
    micro_b,
    n_bins: int,
    n_casc: int,
    make_fine: bool,
):
    """Compute a single FCS correlation curve for one burst.

    Returns ``(tau_ms, G(tau))`` or ``(None, None)`` if the burst does not
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

    settings = {"n_bins": int(n_bins), "n_casc": int(n_casc), "make_fine": bool(make_fine)}
    correlator = tttrlib.Correlator(**settings)
    correlator.set_macrotimes(t, t)
    correlator.set_weights(w1, w2)
    tau = correlator.x_axis * dT_ms
    if make_fine:
        try:
            n_mt = tttr.get_number_of_micro_time_channels()
            mt = tttr.micro_times
            correlator.set_microtimes(mt, mt, n_mt)
            tau = tau / (tttr.header.micro_time_resolution / 1000.0)
        except Exception:
            pass
    g = correlator.correlation
    return np.asarray(tau, dtype=float), np.asarray(g, dtype=float)


def fit_diffusion_time(tau: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    """Fit a MaxEnt diffusion-time distribution and summarize it.

    Returns ``(td_mean_ms, td_peak_ms)``; both NaN on failure.
    """
    tau = np.asarray(tau, dtype=float)
    g = np.asarray(g, dtype=float)
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
    tau: np.ndarray, g: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate single-component diffusion time with a 3D Gaussian FCS model.

    Returns ``(td_ms, tau_used, g_used, g_fit)``; ``(NaN, [], [], [])`` on failure.
    """
    tau = np.asarray(tau, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    mask = np.isfinite(tau) & np.isfinite(g) & (tau > 0)
    tau = tau[mask]
    g = g[mask]
    empty = np.asarray([], dtype=float)
    if tau.size < 5:
        return float("nan"), empty, empty, empty

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
        return float("nan"), empty, empty, empty
    if not np.isfinite(tau_min) or not np.isfinite(tau_max) or tau_min <= 0.0:
        return float("nan"), empty, empty, empty

    td_min = max(tau_min * 0.1, tau_min * 1e-2, 1e-6)
    td_max = max(tau_max * 10.0, td_min * 1.1)

    try:
        td_grid = np.logspace(np.log10(td_min), np.log10(td_max), 60)
    except Exception:
        return float("nan"), empty, empty, empty

    best_td = float("nan")
    best_sse = np.inf
    best_g_fit: Optional[np.ndarray] = None

    for td_val in td_grid:
        try:
            f = _shape(float(td_val))
            X = np.column_stack([np.ones_like(f), f])
            beta, _, _, _ = np.linalg.lstsq(X, g, rcond=None)
            g_model = float(beta[0]) + float(beta[1]) * f
            sse = float(np.sum((g - g_model) ** 2))
        except Exception:
            continue
        if not np.isfinite(sse):
            continue
        if sse < best_sse:
            best_sse = sse
            best_td = float(td_val)
            best_g_fit = g_model

    if not np.isfinite(best_td):
        return float("nan"), empty, empty, empty
    if best_g_fit is None:
        return best_td, tau, g, empty
    return best_td, tau, g, np.asarray(best_g_fit, dtype=float)


# ----------------------------------------------------------------------
# Declarative settings + file-level orchestration
# ----------------------------------------------------------------------


@dataclasses.dataclass
class BurstFcsSettings:
    """Correlator + fitting settings for a burst-wise FCS run."""

    n_bins: int = 3
    n_casc: int = 20
    make_fine: bool = False
    padding_ms: float = 100.0
    fit_mode: str = "simple"  # "simple" | "maxent" | "none"
    maxent_reg: float = 0.1
    maxent_td_min: Optional[float] = None
    maxent_td_max: Optional[float] = None
    tmin_fit: Optional[float] = None
    tmax_fit: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BurstFcsSettings":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PairConfig:
    """One FCS channel pair (cross- or auto-correlation)."""

    pair_name: str
    chs_a: List[int]
    chs_b: List[int]
    micro_a: List[Any] = dataclasses.field(default_factory=list)
    micro_b: List[Any] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PairConfig":
        return cls(
            pair_name=str(d.get("pair_name") or d.get("name") or ""),
            chs_a=list(d.get("chs_a", [])),
            chs_b=list(d.get("chs_b", [])),
            micro_a=list(d.get("micro_a", []) or []),
            micro_b=list(d.get("micro_b", []) or []),
        )


def fit_curve(tau: np.ndarray, g: np.ndarray, settings: BurstFcsSettings) -> Dict[str, Any]:
    """Fit one correlation curve per ``settings.fit_mode``.

    Returns a dict with ``tau``, ``g`` (used data), ``g_fit``, ``td_grid``,
    ``p`` (MaxEnt distribution), ``td_mean`` and ``td_peak`` — all as plain
    lists / floats so the result is transport friendly.
    """
    tau_arr = np.asarray(tau, dtype=float)
    g_arr = np.asarray(g, dtype=float)

    if settings.tmin_fit is not None or settings.tmax_fit is not None:
        mask = np.isfinite(tau_arr) & np.isfinite(g_arr)
        if settings.tmin_fit is not None:
            mask &= tau_arr >= settings.tmin_fit
        if settings.tmax_fit is not None:
            mask &= tau_arr <= settings.tmax_fit
        tau_arr = tau_arr[mask]
        g_arr = g_arr[mask]

    out: Dict[str, Any] = {
        "tau": tau_arr.tolist(),
        "g": g_arr.tolist(),
        "g_fit": [],
        "td_grid": [],
        "p": [],
        "td_mean": float("nan"),
        "td_peak": float("nan"),
        "fit_mode": settings.fit_mode,
    }

    if settings.fit_mode == "maxent":
        try:
            res = fcs_maxent(
                tau=tau_arr, g=g_arr,
                td_min=settings.maxent_td_min,
                td_max=settings.maxent_td_max,
                reg=settings.maxent_reg,
            )
        except Exception:
            res = None
        if isinstance(res, dict):
            td_grid = np.asarray(res.get("td_grid", []), dtype=float)
            p = np.clip(np.asarray(res.get("p", []), dtype=float), 0.0, np.inf)
            out["tau"] = np.asarray(res.get("tau", tau_arr), dtype=float).tolist()
            out["g"] = np.asarray(res.get("g", g_arr), dtype=float).tolist()
            out["g_fit"] = np.asarray(res.get("g_fit", []), dtype=float).tolist()
            out["td_grid"] = td_grid.tolist()
            out["p"] = p.tolist()
            if td_grid.size and p.size and np.any(p > 0):
                out["td_mean"] = float(np.sum(td_grid * p) / np.sum(p))
                out["td_peak"] = float(td_grid[np.argmax(p)])
    elif settings.fit_mode == "simple":
        td, tau_used, g_used, g_fit = fit_simple_diffusion(tau_arr, g_arr)
        out["tau"] = np.asarray(tau_used, dtype=float).tolist()
        out["g"] = np.asarray(g_used, dtype=float).tolist()
        out["g_fit"] = np.asarray(g_fit, dtype=float).tolist()
        out["td_mean"] = float(td)
        out["td_peak"] = float(td)

    return out


def correlate_burst_file(
    tttr_path: pathlib.Path,
    ranges: List[Tuple[int, int]],
    pairs: List[PairConfig],
    settings: BurstFcsSettings,
    filetype=None,
) -> List[Dict[str, Any]]:
    """Correlate every (burst × pair) for one TTTR file.

    Opens the TTTR once, applies optional time padding around each burst, slices
    the burst photons, correlates each channel pair and (optionally) fits the
    curve. Returns one transport-friendly dict per produced curve.
    """
    tttr = open_tttr(pathlib.Path(tttr_path), filetype)
    if tttr is None:
        return []

    try:
        n_events = len(tttr)
    except Exception:
        n_events = None

    pad_ms = float(settings.padding_ms or 0.0)
    mt = None
    pad_ticks = 0.0
    if pad_ms > 0.0:
        try:
            macro_res_s = float(tttr.header.macro_time_resolution)
        except Exception:
            macro_res_s = 0.0
        if macro_res_s > 0.0:
            try:
                mt = np.asarray(tttr.macro_times, dtype=float)
            except Exception:
                mt = None
            if mt is not None and mt.size > 0:
                pad_ticks = (pad_ms / 1000.0) / macro_res_s

    curves: List[Dict[str, Any]] = []
    for burst_index, (start, stop) in enumerate(ranges):
        try:
            s = int(start)
            e = int(stop)
        except Exception:
            continue
        if n_events is not None:
            s = max(0, s)
            e = min(e, n_events - 1)
            if e < s:
                continue

        if pad_ms > 0.0 and mt is not None and mt.size > 0 and pad_ticks > 0.0:
            try:
                t_min = max(0.0, float(mt[s]) - pad_ticks)
                t_max = float(mt[e]) + pad_ticks
                s_pad = max(0, int(np.searchsorted(mt, t_min, side="left")))
                e_pad = min(mt.size - 1, int(np.searchsorted(mt, t_max, side="right") - 1))
                if e_pad >= s_pad:
                    s, e = s_pad, e_pad
            except Exception:
                pass

        try:
            tttr_burst = tttr[s:e + 1]
        except Exception:
            continue

        for pair in pairs:
            tau, g = correlate_single_burst(
                tttr_burst,
                chs_a=pair.chs_a, chs_b=pair.chs_b,
                micro_a=pair.micro_a, micro_b=pair.micro_b,
                n_bins=settings.n_bins, n_casc=settings.n_casc,
                make_fine=settings.make_fine,
            )
            if tau is None or g is None:
                continue
            fit = fit_curve(tau, g, settings)
            curves.append({
                "file": pathlib.Path(tttr_path).name,
                "burst_index": burst_index,
                "pair_name": pair.pair_name,
                "tau_raw": np.asarray(tau, dtype=float).tolist(),
                "g_raw": np.asarray(g, dtype=float).tolist(),
                **fit,
            })

    return curves
