import chisurf.core.fluorescence.general
import chisurf.core.fluorescence.intensity
import chisurf.core.fluorescence.anisotropy
import chisurf.core.fluorescence.fcs
import chisurf.core.fluorescence.fret
import chisurf.core.fluorescence.tcspc
import chisurf.core.fluorescence.burst

import numpy as np

import chisurf.core.settings
from chisurf.core.settings.settings_utils import build_fret_rda_axis


def rebuild_rda_axis_from_settings() -> np.ndarray:
    """Rebuild the R_DA axis from the current fret settings.

    Parameters
    ----------
    None

    Returns
    -------
    np.ndarray
        The R_DA axis array.
    """
    fret_cfg = getattr(chisurf.core.settings, "fret", {}) or {}
    rda_min = fret_cfg.get("rda_min", 1.0)
    rda_max = fret_cfg.get("rda_max", 130.0)
    rda_res = fret_cfg.get("rda_resolution", 96)
    rda_scale = fret_cfg.get("rda_scale", "log")
    axis = build_fret_rda_axis(rda_min, rda_max, rda_res, rda_scale)
    globals()["rda_axis"] = axis
    return axis


try:
    rda_axis = rebuild_rda_axis_from_settings()
except Exception:
    # Final fallback in case settings or axis construction fail
    rda_axis = build_fret_rda_axis(1.0, 130.0, 96, "log")

