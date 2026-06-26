"""
FCS Confocal Calculator — v3 (PyQt5)

What’s improved versus prior builds
- **Exclusive constraints** via radio buttons (Fix D | Fix r_h | Fix Veff), no accidental multi‑select.
- **Auto‑disable computed fields**: outputs are read‑only per constraint (prevents edits being overwritten).
- **Apply D_ref → D** now also switches to Fix D and focuses D.
- **Robust N↔concentration coupling** with signal‑guarding; both fields are user‑editable.
- **Water η(T) override** disables η field to make it clear it’s computed.
- Optimized UI: Apply D_ref → D placed next to dye combobox; removed Show refs. and Help buttons; Constraint group moved to top.

Core model (same as spec)
- Enter τ (ms). Choose ONE constraint:
    • Fix D  → compute Veff and r_h
    • Fix r_h → compute D and Veff
    • Fix Veff → compute D and r_h
- Temperature/viscosity aware; water η(T) per Kapusta (2010) app note.
- Reference dyes (D@25 °C, water); optional D(T,η) scaling before applying.
- Concentration (nM) ↔ Number of molecules N:  N = 0.602214 × c_nM × V_fL

Units: τ [ms], D [µm²/s], r_h [nm], η [mPa·s], Veff [fL]
"""

from pathlib import Path as _Path

from chisurf.core.plugin import load_manifest as _load_manifest

from .core import compute_confocal  # noqa: F401  (public core entry)
from .gui.client import ConfocalCalcClient  # noqa: F401

_manifest = _load_manifest(_Path(__file__).with_name("manifest.json"))
name = (
    _manifest.display_name if _manifest is not None
    else "Spectroscopy:Fluorescence Correlation Spectroscopy:Diffusion/Volume Calculator"
)

# Hidden from the menu: surfaced inside the FCS Toolbox meta tool.
menu_hidden = True

__all__ = ["ConfocalCalcClient", "compute_confocal", "menu_hidden", "name"]

