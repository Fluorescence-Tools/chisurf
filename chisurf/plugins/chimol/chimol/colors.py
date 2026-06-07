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


# ------------------------------------------------------------------ #
# PyMOL-compatible named color table
# Extracted from pymol-open-source/layer1/Color.cpp lines 1024-1322
# ------------------------------------------------------------------ #

_PYMOL_COLORS: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "dash": (1.0, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "salmon": (1.0, 0.6, 0.6),
    "lime": (0.5, 1.0, 0.5),
    "slate": (0.5, 0.5, 1.0),
    "hotpink": (1.0, 0.0, 0.5),
    "orange": (1.0, 0.5, 0.0),
    "chartreuse": (0.5, 1.0, 0.0),
    "limegreen": (0.0, 1.0, 0.5),
    "purpleblue": (0.5, 0.0, 1.0),
    "marine": (0.0, 0.5, 1.0),
    "olive": (0.77, 0.7, 0.0),
    "purple": (0.75, 0.0, 0.75),
    "teal": (0.0, 0.75, 0.75),
    "ruby": (0.6, 0.2, 0.2),
    "forest": (0.2, 0.6, 0.2),
    "deepblue": (0.25, 0.25, 0.65),
    "grey": (0.5, 0.5, 0.5),
    "gray": (0.5, 0.5, 0.5),
    "carbon": (0.2, 1.0, 0.2),
    "nitrogen": (0.2, 0.2, 1.0),
    "oxygen": (1.0, 0.3, 0.3),
    "hydrogen": (0.9, 0.9, 0.9),
    "brightorange": (1.0, 0.7, 0.2),
    "sulfur": (0.9, 0.775, 0.25),
    "tv_red": (1.0, 0.2, 0.2),
    "tv_green": (0.2, 1.0, 0.2),
    "tv_blue": (0.3, 0.3, 1.0),
    "tv_yellow": (1.0, 1.0, 0.2),
    "yelloworange": (1.0, 0.87, 0.37),
    "tv_orange": (1.0, 0.55, 0.15),
    "br0": (0.1, 0.1, 1.0),
    "br1": (0.2, 0.1, 0.9),
    "br2": (0.3, 0.1, 0.8),
    "br3": (0.4, 0.1, 0.7),
    "br4": (0.5, 0.1, 0.6),
    "br5": (0.6, 0.1, 0.5),
    "br6": (0.7, 0.1, 0.4),
    "br7": (0.8, 0.1, 0.3),
    "br8": (0.9, 0.1, 0.2),
    "br9": (1.0, 0.1, 0.1),
    "pink": (1.0, 0.65, 0.85),
    "firebrick": (0.698, 0.13, 0.13),
    "chocolate": (0.555, 0.222, 0.111),
    "brown": (0.65, 0.32, 0.17),
    "wheat": (0.99, 0.82, 0.65),
    "violet": (1.0, 0.5, 1.0),
    "lightmagenta": (1.0, 0.2, 0.8),
    "density": (0.1, 0.1, 0.6),
    "paleyellow": (1.0, 1.0, 0.5),
    "aquamarine": (0.5, 1.0, 1.0),
    "deepsalmon": (1.0, 0.5, 0.5),
    "palegreen": (0.65, 0.9, 0.65),
    "deepolive": (0.6, 0.6, 0.1),
    "deeppurple": (0.6, 0.1, 0.6),
    "deepteal": (0.1, 0.6, 0.6),
    "lightblue": (0.75, 0.75, 1.0),
    "lightorange": (1.0, 0.8, 0.5),
    "palecyan": (0.8, 1.0, 1.0),
    "lightteal": (0.4, 0.7, 0.7),
    "splitpea": (0.52, 0.75, 0.0),
    "raspberry": (0.7, 0.3, 0.4),
    "sand": (0.72, 0.55, 0.3),
    "smudge": (0.55, 0.7, 0.4),
    "violetpurple": (0.55, 0.25, 0.6),
    "dirtyviolet": (0.7, 0.5, 0.5),
    "_deepsalmon": (1.0, 0.42, 0.42),
    "lightpink": (1.0, 0.75, 0.87),
    "greencyan": (0.25, 1.0, 0.75),
    "limon": (0.75, 1.0, 0.25),
    "skyblue": (0.2, 0.5, 0.8),
    "bluewhite": (0.85, 0.85, 1.0),
    "warmpink": (0.85, 0.2, 0.5),
    "darksalmon": (0.73, 0.55, 0.52),
    # Element colors
    "helium": (0.850980392, 1.0, 1.0),
    "lithium": (0.8, 0.501960784, 1.0),
    "beryllium": (0.760784314, 1.0, 0.0),
    "boron": (1.0, 0.709803922, 0.709803922),
    "fluorine": (0.701960784, 1.0, 1.0),
    "neon": (0.701960784, 0.890196078, 0.960784314),
    "sodium": (0.670588235, 0.360784314, 0.949019608),
    "magnesium": (0.541176471, 1.0, 0.0),
    "aluminum": (0.749019608, 0.650980392, 0.650980392),
    "silicon": (0.941176471, 0.784313725, 0.62745098),
    "phosphorus": (1.0, 0.501960784, 0.0),
    "chlorine": (0.121568627, 0.941176471, 0.121568627),
    "argon": (0.501960784, 0.819607843, 0.890196078),
    "potassium": (0.560784314, 0.250980392, 0.831372549),
    "calcium": (0.239215686, 1.0, 0.0),
    "scandium": (0.901960784, 0.901960784, 0.901960784),
    "titanium": (0.749019608, 0.760784314, 0.780392157),
    "vanadium": (0.650980392, 0.650980392, 0.670588235),
    "chromium": (0.541176471, 0.6, 0.780392157),
    "manganese": (0.611764706, 0.478431373, 0.780392157),
    "iron": (0.878431373, 0.4, 0.2),
    "cobalt": (0.941176471, 0.564705882, 0.62745098),
    "nickel": (0.31372549, 0.815686275, 0.31372549),
    "copper": (0.784313725, 0.501960784, 0.2),
    "zinc": (0.490196078, 0.501960784, 0.690196078),
    "gallium": (0.760784314, 0.560784314, 0.560784314),
    "germanium": (0.4, 0.560784314, 0.560784314),
    "arsenic": (0.741176471, 0.501960784, 0.890196078),
    "selenium": (1.0, 0.631372549, 0.0),
    "bromine": (0.650980392, 0.160784314, 0.160784314),
    "krypton": (0.360784314, 0.721568627, 0.819607843),
    "rubidium": (0.439215686, 0.180392157, 0.690196078),
    "strontium": (0.0, 1.0, 0.0),
    "yttrium": (0.580392157, 1.0, 1.0),
    "zirconium": (0.580392157, 0.878431373, 0.878431373),
    "niobium": (0.450980392, 0.760784314, 0.788235294),
    "molybdenum": (0.329411765, 0.709803922, 0.709803922),
    "technetium": (0.231372549, 0.619607843, 0.619607843),
    "ruthenium": (0.141176471, 0.560784314, 0.560784314),
    "rhodium": (0.039215686, 0.490196078, 0.549019608),
    "palladium": (0.0, 0.411764706, 0.521568627),
    "silver": (0.752941176, 0.752941176, 0.752941176),
    "cadmium": (1.0, 0.850980392, 0.560784314),
    "indium": (0.650980392, 0.458823529, 0.450980392),
    "tin": (0.4, 0.501960784, 0.501960784),
    "antimony": (0.619607843, 0.388235294, 0.709803922),
    "tellurium": (0.831372549, 0.478431373, 0.0),
    "iodine": (0.580392157, 0.0, 0.580392157),
    "xenon": (0.258823529, 0.619607843, 0.690196078),
    "cesium": (0.341176471, 0.090196078, 0.560784314),
    "barium": (0.0, 0.788235294, 0.0),
    "lanthanum": (0.439215686, 0.831372549, 1.0),
    "cerium": (1.0, 1.0, 0.780392157),
    "praseodymium": (0.850980392, 1.0, 0.780392157),
    "neodymium": (0.780392157, 1.0, 0.780392157),
    "promethium": (0.639215686, 1.0, 0.780392157),
    "samarium": (0.560784314, 1.0, 0.780392157),
    "europium": (0.380392157, 1.0, 0.780392157),
    "gadolinium": (0.270588235, 1.0, 0.780392157),
    "terbium": (0.188235294, 1.0, 0.780392157),
    "dysprosium": (0.121568627, 1.0, 0.780392157),
    "holmium": (0.0, 1.0, 0.611764706),
    "erbium": (0.0, 0.901960784, 0.458823529),
    "thulium": (0.0, 0.831372549, 0.321568627),
    "ytterbium": (0.0, 0.749019608, 0.219607843),
    "lutetium": (0.0, 0.670588235, 0.141176471),
    "hafnium": (0.301960784, 0.760784314, 1.0),
    "tantalum": (0.301960784, 0.650980392, 1.0),
    "tungsten": (0.129411765, 0.580392157, 0.839215686),
    "rhenium": (0.149019608, 0.490196078, 0.670588235),
    "osmium": (0.149019608, 0.4, 0.588235294),
    "iridium": (0.090196078, 0.329411765, 0.529411765),
    "platinum": (0.815686275, 0.815686275, 0.878431373),
    "gold": (1.0, 0.819607843, 0.137254902),
    "mercury": (0.721568627, 0.721568627, 0.815686275),
    "thallium": (0.650980392, 0.329411765, 0.301960784),
    "lead": (0.341176471, 0.349019608, 0.380392157),
    "bismuth": (0.619607843, 0.309803922, 0.709803922),
    "polonium": (0.670588235, 0.360784314, 0.0),
    "astatine": (0.458823529, 0.309803922, 0.270588235),
    "radon": (0.258823529, 0.509803922, 0.588235294),
    "francium": (0.258823529, 0.0, 0.4),
    "radium": (0.0, 0.490196078, 0.0),
    "actinium": (0.439215686, 0.670588235, 0.980392157),
    "thorium": (0.0, 0.729411765, 1.0),
    "protactinium": (0.0, 0.631372549, 1.0),
    "uranium": (0.0, 0.560784314, 1.0),
    "neptunium": (0.0, 0.501960784, 1.0),
    "plutonium": (0.0, 0.419607843, 1.0),
    "americium": (0.329411765, 0.360784314, 0.949019608),
    "curium": (0.470588235, 0.360784314, 0.890196078),
    "berkelium": (0.541176471, 0.309803922, 0.890196078),
    "californium": (0.631372549, 0.211764706, 0.831372549),
    "einsteinium": (0.701960784, 0.121568627, 0.831372549),
    "fermium": (0.701960784, 0.121568627, 0.729411765),
    "mendelevium": (0.701960784, 0.050980392, 0.650980392),
    "nobelium": (0.741176471, 0.050980392, 0.529411765),
    "lawrencium": (0.780392157, 0.0, 0.4),
    "rutherfordium": (0.8, 0.0, 0.349019608),
    "dubnium": (0.819607843, 0.0, 0.309803922),
    "seaborgium": (0.850980392, 0.0, 0.270588235),
    "bohrium": (0.878431373, 0.0, 0.219607843),
    "hassium": (0.901960784, 0.0, 0.180392157),
    "meitnerium": (0.921568627, 0.0, 0.149019608),
    "deuterium": (0.9, 0.9, 0.9),
    "lonepair": (0.5, 0.5, 0.5),
    "pseudoatom": (0.9, 0.9, 0.9),
}

# Add grey00-grey99 and gray00-gray99 (generated programmatically)
for _i in range(100):
    _v = _i / 99.0
    _PYMOL_COLORS[f"grey{_i:02d}"] = (_v, _v, _v)
    _PYMOL_COLORS[f"gray{_i:02d}"] = (_v, _v, _v)


def get_pymol_color(name: str) -> tuple[float, float, float, float]:
    """Look up a PyMOL color name, returning RGBA or raising KeyError."""
    key = name.strip().lower()
    rgb = _PYMOL_COLORS[key]
    return (rgb[0], rgb[1], rgb[2], 1.0)


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


def _build_element_color_array(
    elements: Optional[np.ndarray], n_points: int
) -> np.ndarray:
    """Color atoms by element (CPK style)."""
    if elements is None or n_points <= 0:
        base = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)
        return np.tile(base, (max(n_points, 1), 1))

    colors_cfg = _DISPLAY_CONFIG.get("colors", {})
    cpk_cfg = colors_cfg.get("element_cpk", {})

    default_color = np.asarray(cpk_cfg.get("default", [0.8, 0.8, 0.8, 1.0]), dtype=float)
    
    colors = np.zeros((n_points, 4), dtype=float)
    m = min(len(elements), n_points)

    for i in range(m):
        el = str(elements[i]).strip().upper()
        col = cpk_cfg.get(el)
        if col is not None:
            colors[i, :] = col
        else:
            colors[i, :] = default_color

    if m < n_points:
        colors[m:, :] = default_color
    return colors


def _build_chain_color_array(
    chains: Optional[np.ndarray], n_points: int
) -> np.ndarray:
    """Color atoms by chain ID."""
    if chains is None or n_points <= 0:
        base = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)
        return np.tile(base, (max(n_points, 1), 1))

    # A pleasant multi-color palette
    palette = [
        [0.3, 0.3, 0.9, 1.0], # Blue
        [0.9, 0.3, 0.3, 1.0], # Red
        [0.3, 0.8, 0.3, 1.0], # Green
        [0.9, 0.9, 0.3, 1.0], # Yellow
        [0.9, 0.3, 0.9, 1.0], # Magenta
        [0.3, 0.9, 0.9, 1.0], # Cyan
        [0.9, 0.6, 0.3, 1.0], # Orange
        [0.6, 0.3, 0.9, 1.0], # Purple
    ]

    chain_map = {}
    next_col = 0
    
    colors = np.zeros((n_points, 4), dtype=float)
    m = min(len(chains), n_points)
    
    for i in range(m):
        chid = str(chains[i]).strip()
        if chid not in chain_map:
            chain_map[chid] = palette[next_col % len(palette)]
            next_col += 1
        colors[i, :] = chain_map[chid]

    if m < n_points:
        colors[m:, :] = colors[m - 1, :]
    return colors

