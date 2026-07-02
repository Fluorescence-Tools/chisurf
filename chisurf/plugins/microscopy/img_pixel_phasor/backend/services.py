"""``phasor.*`` RPC service — exposes the phasor toolkit over JSON-RPC (PRD-56 §2.2).

Thin, transport-friendly wrappers over :mod:`..analysis` (pure numpy/scipy). Handlers
take and return JSON-friendly arrays (nested lists), so any RPC client — ChiSurf's
own ``ChisurfClient`` / ``InProcessClient`` or ndXplorer's chisurf-free
``ZmqRpcClient`` — can drive them. Registered with the server's ``ServiceDispatcher``
via the plugin manifest ``entrypoints.services`` (mirrors ``plugins/pch``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .. import analysis

logger = logging.getLogger(__name__)

_DEFAULT_FREQUENCY_MHZ = 80.0
_DEFAULT_TAUS = (0.5, 1.0, 2.0, 4.0, 8.0)

_METHODS = {
    "phasor.describe": lambda p: _describe_handler(**p),
    "phasor.apparent_lifetime": lambda p: _apparent_lifetime_handler(**p),
    "phasor.filter": lambda p: _filter_handler(**p),
    "phasor.component_fraction": lambda p: _component_fraction_handler(**p),
    "phasor.unmix": lambda p: _unmix_handler(**p),
    "phasor.cursor_mask": lambda p: _cursor_mask_handler(**p),
    "phasor.pseudo_color": lambda p: _pseudo_color_handler(**p),
    "phasor.overlays": lambda p: _overlays_handler(**p),
    "phasor.contours": lambda p: _contours_handler(**p),
}


def register_services(dispatcher: Any) -> None:
    """Register every ``phasor.*`` method with the server dispatcher."""
    for name, handler in _METHODS.items():
        dispatcher.register(name, handler)


def _ok(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def _fail(exc: Exception, what: str) -> dict[str, Any]:
    logger.exception("phasor RPC failed: %s", what)
    return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------------------
def _describe_handler(**kwargs: Any) -> dict[str, Any]:
    """Capabilities + defaults, so a client can build its UI without hard-coding."""
    return _ok(
        {
            "methods": sorted(_METHODS),
            "default_frequency_mhz": _DEFAULT_FREQUENCY_MHZ,
            "default_harmonic": 1,
            "default_taus_ns": list(_DEFAULT_TAUS),
            "overlay_sets": list(analysis.OVERLAY_SETS),
            "filters": ["median", "gaussian"],
        }
    )


def _apparent_lifetime_handler(
    g: Any,
    s: Any,
    frequency_mhz: float = _DEFAULT_FREQUENCY_MHZ,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        tau_phi, tau_m = analysis.phasor_to_apparent_lifetime(
            np.asarray(g, dtype=float), np.asarray(s, dtype=float), float(frequency_mhz)
        )
        return _ok({"tau_phi": tau_phi.tolist(), "tau_m": tau_m.tolist()})
    except Exception as exc:
        return _fail(exc, "apparent_lifetime")


def _filter_handler(
    g: Any,
    s: Any,
    kind: str = "median",
    size: int = 3,
    repeat: int = 1,
    sigma: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        g_arr = np.asarray(g, dtype=float)
        s_arr = np.asarray(s, dtype=float)
        if kind == "median":
            gf, sf = analysis.phasor_filter_median(g_arr, s_arr, size=int(size), repeat=int(repeat))
        elif kind == "gaussian":
            gf, sf = analysis.phasor_filter_gaussian(g_arr, s_arr, sigma=float(sigma))
        else:
            raise ValueError(f"unknown filter kind: {kind!r}")
        return _ok({"g": gf.tolist(), "s": sf.tolist(), "kind": kind})
    except Exception as exc:
        return _fail(exc, "filter")


def _component_fraction_handler(
    g: Any,
    s: Any,
    c1: Any,
    c2: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        frac = analysis.phasor_component_fraction(
            np.asarray(g, dtype=float), np.asarray(s, dtype=float), c1, c2
        )
        return _ok({"fraction": frac.tolist()})
    except Exception as exc:
        return _fail(exc, "component_fraction")


def _unmix_handler(
    g: Any,
    s: Any,
    components: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        fractions = analysis.phasor_unmix(
            np.asarray(g, dtype=float), np.asarray(s, dtype=float), components
        )
        return _ok({"fractions": [f.tolist() for f in fractions]})
    except Exception as exc:
        return _fail(exc, "unmix")


def _cursor_mask_handler(
    g: Any,
    s: Any,
    center: Any,
    kind: str = "circular",
    radius: float = 0.05,
    radii: Any = None,
    angle: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        g_arr = np.asarray(g, dtype=float)
        s_arr = np.asarray(s, dtype=float)
        if kind == "circular":
            mask = analysis.mask_from_circular_cursor(g_arr, s_arr, center, float(radius))
        elif kind == "elliptic":
            if radii is None:
                raise ValueError("elliptic cursor requires 'radii'")
            mask = analysis.mask_from_elliptic_cursor(g_arr, s_arr, center, radii, float(angle))
        else:
            raise ValueError(f"unknown cursor kind: {kind!r}")
        return _ok({"mask": mask.tolist(), "n_selected": int(mask.sum())})
    except Exception as exc:
        return _fail(exc, "cursor_mask")


def _pseudo_color_handler(
    masks: Any,
    colors: Any = None,
    intensity: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        mask_arrays = [np.asarray(m, dtype=bool) for m in masks]
        inten = None if intensity is None else np.asarray(intensity, dtype=float)
        rgb = analysis.pseudo_color(mask_arrays, colors=colors, intensity=inten)
        return _ok({"rgb": rgb.tolist(), "shape": list(rgb.shape)})
    except Exception as exc:
        return _fail(exc, "pseudo_color")


def _overlays_handler(
    frequency_mhz: float = _DEFAULT_FREQUENCY_MHZ,
    harmonic: int = 1,
    sets: Any = None,
    taus: Any = None,
    c1: Any = None,
    c2: Any = None,
    tau_d0: float = 4.0,
    e_range: Any = None,
    n_points: int = 256,
    components: Any = None,
    fractions: Any = None,
    cursors: Any = None,
    polar_radii: Any = None,
    polar_angles: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return the requested reference-geometry polylines as ``{name, kind, x, y, style}``."""
    try:
        overlays = analysis.build_overlays(
            frequency_mhz=float(frequency_mhz), harmonic=int(harmonic), sets=sets,
            taus=taus, c1=c1, c2=c2, tau_d0=float(tau_d0), e_range=e_range,
            n_points=int(n_points), components=components, fractions=fractions,
            cursors=cursors, polar_radii=polar_radii, polar_angles=polar_angles,
        )
        return _ok({"frequency_mhz": float(frequency_mhz) * int(harmonic), "overlays": overlays})
    except Exception as exc:
        return _fail(exc, "overlays")


def _contours_handler(
    density: Any,
    g_range: Any,
    s_range: Any,
    levels: Any = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return iso-density contour polylines of a 2-D phasor histogram."""
    try:
        contours = analysis.density_contours(
            np.asarray(density, dtype=float), g_range, s_range, levels=levels
        )
        return _ok({"overlays": contours})
    except Exception as exc:
        return _fail(exc, "contours")
