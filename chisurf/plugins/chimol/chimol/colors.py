from __future__ import annotations

from typing import Optional

import numpy as np
from qtpy import QtCore

from .config import _DISPLAY_CONFIG


_SEQ_COLOR_ROLE = QtCore.Qt.UserRole + 100
_OBJECT_ID_ROLE = QtCore.Qt.UserRole + 101


_AA_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _three_to_one_array(res_names: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if res_names is None:
        return None
    one = []
    for name in res_names:
        key = str(name).strip().upper()
        one.append(_AA_THREE_TO_ONE.get(key, "X"))
    return np.array(one, dtype="U1")


def _color_for_resname(res_name: str) -> tuple[float, float, float, float]:
    """Return an RGBA color for a residue name.

    Colors are loosely grouped by residue type (hydrophobic, polar, charged).
    """

    key = str(res_name).strip().upper()

    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
    polar = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
    positive = {"LYS", "ARG", "HIS"}
    negative = {"ASP", "GLU"}

    colors_cfg = _DISPLAY_CONFIG.get("colors", {})
    aa_cfg = colors_cfg.get("aa_groups", {})

    def _cfg(name: str, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        arr = np.asarray(aa_cfg.get(name, fallback), dtype=float)
        if arr.shape[0] != 4:
            arr = np.asarray(fallback, dtype=float)
        return tuple(float(x) for x in arr)

    if key in hydrophobic:
        return _cfg("hydrophobic", (0.4, 0.8, 0.4, 1.0))
    if key in polar:
        return _cfg("polar", (0.4, 0.7, 0.9, 1.0))
    if key in positive:
        return _cfg("positive", (0.3, 0.3, 0.9, 1.0))
    if key in negative:
        return _cfg("negative", (0.9, 0.3, 0.3, 1.0))
    if key == "GLY":
        return _cfg("gly", (0.8, 0.8, 0.8, 1.0))

    base = np.asarray(_DISPLAY_CONFIG.get("colors", {}).get("base", [0.8, 0.8, 1.0, 1.0]), dtype=float)
    if base.shape[0] != 4:
        base = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)
    return tuple(float(x) for x in base)


def _build_residue_color_array(
    res_names: Optional[np.ndarray], n_points: int
) -> np.ndarray:
    if res_names is None or n_points <= 0:
        base = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)
        return np.tile(base, (max(n_points, 1), 1))

    colors = np.zeros((n_points, 4), dtype=float)
    m = min(len(res_names), n_points)
    for i in range(m):
        colors[i, :] = _color_for_resname(res_names[i])
    if m < n_points:
        colors[m:, :] = colors[m - 1, :]
    return colors


def _color_for_ss(code: str) -> tuple[float, float, float, float]:
    """Return an RGBA color for a secondary-structure code.

    Uses a simple scheme: helices (H) blue, strands (E) red, coil/other
    (C or anything else) light grey/yellow.
    """

    c = str(code).strip().upper()[:1]
    colors_cfg = _DISPLAY_CONFIG.get("colors", {})
    ss_cfg = colors_cfg.get("secondary_structure", {})

    def _cfg(name: str, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        arr = np.asarray(ss_cfg.get(name, fallback), dtype=float)
        if arr.shape[0] != 4:
            arr = np.asarray(fallback, dtype=float)
        return tuple(float(x) for x in arr)

    if c == "H":
        return _cfg("helix", (0.3, 0.3, 0.9, 1.0))
    if c == "E":
        return _cfg("strand", (0.9, 0.3, 0.3, 1.0))
    # coil / turn / other
    return _cfg("coil", (0.9, 0.9, 0.7, 1.0))


def _build_ss_color_array(ss_codes: Optional[np.ndarray], n_points: int) -> np.ndarray:
    if ss_codes is None or n_points <= 0:
        base = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)
        return np.tile(base, (max(n_points, 1), 1))

    colors = np.zeros((n_points, 4), dtype=float)
    m = min(len(ss_codes), n_points)
    for i in range(m):
        colors[i, :] = _color_for_ss(ss_codes[i])
    if m < n_points:
        colors[m:, :] = colors[m - 1, :]
    return colors


def _build_sequence_gradient_colors(n_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.zeros((0, 4), dtype=float)

    colors_cfg = _DISPLAY_CONFIG.get("colors", {})
    gradient_cfg = colors_cfg.get("sequence_gradient", {}) or {}

    default_start = np.array([0.95, 0.45, 0.25, 1.0], dtype=float)
    default_end = np.array([0.25, 0.55, 0.95, 1.0], dtype=float)

    def _gradient_color(key: str, fallback: np.ndarray) -> np.ndarray:
        try:
            arr = np.asarray(gradient_cfg.get(key, fallback), dtype=float)
        except Exception:
            arr = fallback.copy()
        if arr.shape[0] != 4:
            arr = fallback.copy()
        return arr

    start = _gradient_color("start", default_start)
    end = _gradient_color("end", default_end)

    if n_points == 1:
        return start.reshape(1, 4).copy()

    t = np.linspace(0.0, 1.0, n_points, dtype=float)[:, np.newaxis]
    colors = start + (end - start) * t
    colors = np.clip(colors, 0.0, 1.0)
    return colors

