import os
import math
import pathlib
from typing import Dict, Any, List

import numpy as np

# Optional deps

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:  # headless plotting
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Repo root: this file is chisurf/plugins/chat/tools/anisotropy.py -> parents[4]
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    w = int(max(1, w))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    y = (cumsum[w:] - cumsum[:-w]) / float(w)
    # pad to original length by repeating edges
    left = np.full((w // 2,), y[0] if y.size else 0.0)
    right = np.full((len(x) - len(y) - len(left),), y[-1] if y.size else 0.0)
    return np.concatenate([left, y, right])


def run_anisotropy_analysis(
    csv_path: str,
    g_factor: float = 1.0,
    time_col: str = "time",
    Ipar_col: str = "I_par",
    Iperp_col: str = "I_perp",
    smooth_window: int = 1,
    out_prefix: str = "anisotropy_out",
) -> Dict[str, Any]:
    if pd is None:
        raise RuntimeError("pandas is required for anisotropy analysis")
    p = pathlib.Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(str(p))
    for col in [time_col, Ipar_col, Iperp_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV")

    t = df[time_col].to_numpy(dtype=float)
    Ipar = df[Ipar_col].to_numpy(dtype=float)
    Iperp = df[Iperp_col].to_numpy(dtype=float)

    # r(t) = (I_par - G*I_perp) / (I_par + 2*G*I_perp)
    num = Ipar - g_factor * Iperp
    den = Ipar + 2.0 * g_factor * Iperp
    # avoid division by zero
    eps = 1e-12
    den = np.where(np.abs(den) < eps, np.nan, den)
    r = num / den

    if smooth_window and smooth_window > 1:
        r_smooth = _moving_average(r.astype(float), int(smooth_window))
    else:
        r_smooth = r.copy()

    out_dir = pathlib.Path(out_prefix).parent if (os.path.sep in out_prefix) else p.parent
    out_stem = pathlib.Path(out_prefix).name if (os.path.sep in out_prefix) else out_prefix
    out_csv = out_dir / f"{out_stem}.csv"
    out_png = out_dir / f"{out_stem}.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({
        time_col: t,
        "r": r,
        "r_smooth": r_smooth,
    })
    out_df.to_csv(str(out_csv), index=False)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.plot(t, r, label='r (raw)', alpha=0.6)
        ax.plot(t, r_smooth, label=f'r (smooth w={smooth_window})', linewidth=1.5)
        ax.set_xlabel(time_col)
        ax.set_ylabel('anisotropy r(t)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out_png))
        plt.close(fig)
    else:
        # Write a minimal valid 1x1 PNG placeholder to ensure output exists
        png_bytes = bytes([
            0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,
            0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
            0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,
            0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,
            0xDE,0x00,0x00,0x00,0x0A,0x49,0x44,0x41,
            0x54,0x08,0xD7,0x63,0xF8,0xCF,0xC0,0x00,
            0x00,0x03,0x01,0x01,0x00,0x18,0xDD,0x8D,
            0xB1,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,
            0x44,0xAE,0x42,0x60,0x82
        ])
        with open(str(out_png), 'wb') as f:
            f.write(png_bytes)

    # summary stats (nan-safe)
    r_valid = r[np.isfinite(r)]
    summary = {
        "n_points": int(r_valid.size),
        "r_mean": float(np.nanmean(r)) if r.size else float('nan'),
        "r_std": float(np.nanstd(r)) if r.size else float('nan'),
        "r_min": float(np.nanmin(r)) if r.size else float('nan'),
        "r_max": float(np.nanmax(r)) if r.size else float('nan'),
        "out_csv": str(out_csv.resolve()),
        "out_png": str(out_png.resolve()),
    }
    return summary


def list_files(dir_path: str, pattern: str = "*.csv", max_items: int = 200) -> List[str]:
    max_items = max(1, int(max_items))
    d = pathlib.Path(dir_path)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    out: List[str] = []
    for p in sorted(d.glob(pattern)):
        if p.is_file():
            out.append(str(p.resolve()))
            if len(out) >= max_items:
                break
    return out


def read_text_head(path: str, n: int = 2000) -> str:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        data = p.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:  # pragma: no cover
        raise RuntimeError(str(e))
    return data[: int(n)]


def batch_run_anisotropy(
    dir_path: str,
    pattern: str = "*.csv",
    g_factor: float = 1.0,
    time_col: str = "time",
    Ipar_col: str = "I_par",
    Iperp_col: str = "I_perp",
    smooth_window: int = 1,
    out_prefix: str = "anisotropy_out",
) -> Dict[str, Any]:
    d = pathlib.Path(dir_path)
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = [str(p.resolve()) for p in sorted(d.glob(pattern)) if p.is_file()]
    results: List[Dict[str, Any]] = []
    for f in files:
        base = pathlib.Path(f).with_suffix("")
        # save next to input
        out_pref = str(base.parent / (base.name + f"_{out_prefix}"))
        try:
            res = run_anisotropy_analysis(
                csv_path=f,
                g_factor=g_factor,
                time_col=time_col,
                Ipar_col=Ipar_col,
                Iperp_col=Iperp_col,
                smooth_window=smooth_window,
                out_prefix=out_pref,
            )
            res_entry = {
                "file": f,
                "r_mean": res.get("r_mean"),
                "r_std": res.get("r_std"),
                "out_csv": res.get("out_csv"),
                "out_png": res.get("out_png"),
            }
            results.append(res_entry)
        except Exception as e:
            results.append({"file": f, "error": str(e)})

    return {"count": len(files), "results": results}
