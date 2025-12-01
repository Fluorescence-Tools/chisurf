import chisurf.fluorescence.general
import chisurf.fluorescence.intensity
import chisurf.fluorescence.anisotropy
import chisurf.fluorescence.fcs
import chisurf.fluorescence.fret
import chisurf.fluorescence.simulation
import chisurf.fluorescence.tcspc
import chisurf.fluorescence.burst

import numpy as np

import chisurf.settings
from chisurf.settings.settings_utils import build_fret_rda_axis


def rebuild_rda_axis_from_settings() -> np.ndarray:
    fret_cfg = getattr(chisurf.settings, "fret", {}) or {}
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

