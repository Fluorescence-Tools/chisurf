"""Display configuration loading for Chimol."""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import chisurf.settings as _cs_settings
except Exception:  # pragma: no cover - moview can run without chisurf
    _cs_settings = None


def _load_display_config() -> dict:
    """Load Chimol display configuration from JSON file.

    All tunable visual parameters (cartoon radius, AO strength, ball sizes,
    etc.) are collected in ``chimol_display.json``. If the file cannot be
    read, sensible defaults are used. For backward compatibility, legacy
    ``molview_display.json`` or ``protview_display.json`` are also accepted
    if present.
    """

    default = {
        "background": "k",
        "defaults": {
            "color_mode": "by_sequence",
        },
        "grid": {"size": 20.0, "spacing": 1.0},
        "info_overlay": {
            "max_width": 260,
            "min_width": 180,
            "full_height": True,
        },
        "cartoon": {
            "radius_scale": 0.05,
            "min_radius": 0.4,
            "segments_circle": 16,
            "subdivisions": 6,
            "ao_radius": 4.0,
            "ao_max_neighbors": 16,
            "ao_strength": 0.45,
            "style": "ribbon",
            "profile_segments": 20,
            "ribbon_thickness_scale": 0.45,
            "arrow_tip_residues": 2,
            # Slight tension to reduce wiggly beta strands; 0 = classic Catmull-Rom
            "spline_tension": 0.3,
            "ss_shapes": {
                # Rounder/taller helix cross-section similar to NGL cartoon
                "helix": {"width": 0.5, "thickness": 1.35, "profile_power": 1.6},
                "strand": {
                    "width": 1.8,
                    # Flatter cross-section to avoid central specular stripe
                    "thickness": 1.5,
                    "arrow_scale": 0.2,
                    "profile_power": 18.0,
                },
                "coil": {"width": 0.5, "thickness": 0.5, "profile_power": 2.2},
            },
        },
        "balls": {
            "size_scale": 0.04,
            "min_size": 3.0,
            "radius_multiplier": 1.0,
            "ao_radius": 4.0,
            "ao_max_neighbors": 24,
            "ao_strength": 0.5,
            "max_atoms": 8000,
        },
        "dots": {
            "size_px": 8.0,
            "max_points": 250000,
            "alpha": 1.0,
            "base_color": [0.8, 0.8, 1.0, 1.0],
            "px_mode": True,
        },
        "surface": {
            "size_scale": 0.03,
            "min_size": 2.5,
            "ao_radius": 4.5,
            "ao_max_neighbors": 24,
            "ao_strength": 0.6,
            "alpha": 0.8,
            "max_points": 10000,
            "color_mode": "ao_gray",
            "base_color": [0.85, 0.85, 0.92, 1.0],
        },
        "sticks": {
            "width": 2.0,
            "radius": 0.15,
            "segments_circle": 12,
            "max_bonds": 20000,
            "bond_max_length": 1.9,
            "ambient_occlusion": False,
        },
        "colors": {
            "base": [0.8, 0.8, 1.0, 1.0],
            "aa_groups": {
                "hydrophobic": [0.4, 0.8, 0.4, 1.0],
                "polar": [0.4, 0.7, 0.9, 1.0],
                "positive": [0.3, 0.3, 0.9, 1.0],
                "negative": [0.9, 0.3, 0.3, 1.0],
                "gly": [0.8, 0.8, 0.8, 1.0],
            },
            "secondary_structure": {
                "helix": [0.3, 0.3, 0.9, 1.0],
                "strand": [0.9, 0.3, 0.3, 1.0],
                "coil": [0.9, 0.9, 0.7, 1.0],
            },
            "sequence_gradient": {
                "start": [0.95, 0.45, 0.25, 1.0],
                "end": [0.25, 0.55, 0.95, 1.0],
            },
        },
        "selection": {
            "color": [1.0, 1.0, 0.0, 1.0],
            "size_scale": 0.08,
            "min_size": 6.0,
            "max_size": 24.0,
            "alpha": 0.4,
            "click_radius_px": 8.0,
            "px_mode": False,
        },
        "layout": {
            "root_margins": [4, 4, 4, 4],
            "root_spacing": 4,
        },
        "sequence": {
            "residue_tick_step": 20,
            "selection_color": [1.0, 0.95, 0.4, 1.0],
            "selection_text_color": [0.1, 0.1, 0.1, 1.0],
            "number_step": 5,
            "font_family": "Courier New",
            "font_size": 9,
            "font_bold": True,
            "number_font_bold": True,
            "number_height": 16,
            "residue_height": 20,
            "number_color": [0.25, 0.25, 0.25, 1.0],
            "number_bg_color": [0.12, 0.12, 0.12, 1.0],
            "independent_scroll": False,
        },
        "backbone_trace": {
            "protein_atoms": ["CA"],
            "nucleic_atoms": ["P", "C4'", "C1'"],
        },
        "lighting": {
            "light_direction": [0.0, 0.0, 1.0],
            "ambient_strength": 0.1,
            "specular_strength": 0.02,
            "shininess": 4.0,
            "rim_strength": 0.18,
            "rim_power": 2.4,
        },
        "camera": {
            "near_clip": 0.03,
            "far_clip": 2000.0,
            "min_near_clip": 0.005,
            "max_near_clip": 5.0,
            "clip_wheel_scale": 0.85,
        },
    }

    # Prefer a JSON file in the global chisurf settings folder so the user
    # can override display options without touching the source tree. When
    # chisurf is not available (standalone Chimol), fall back to a JSON file
    # shipped next to this module.
    try:
        override_env = os.environ.get("CHIMOL_DISPLAY_CONFIG")
        if override_env:
            override_path = Path(override_env)
            if override_path.is_file():
                path = override_path
            else:
                path = Path(__file__).with_name("chimol_display.json")
        elif _cs_settings is not None:
            settings_dir = _cs_settings.get_path("settings")
            path = settings_dir / "chimol_display.json"
            if not path.is_file():
                legacy = settings_dir / "molview_display.json"
                if legacy.is_file():
                    path = legacy
                else:
                    legacy2 = settings_dir / "protview_display.json"
                    path = legacy2 if legacy2.is_file() else path
                # Fall back to a copy shipped next to this module if present.
                if not path.is_file():
                    path = Path(__file__).with_name("chimol_display.json")
        else:
            path = Path(__file__).with_name("chimol_display.json")
    except Exception:
        path = Path(__file__).with_name("chimol_display.json")

    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return default

    # Shallow-merge user config with defaults to ensure all keys exist.
    for key, sub in default.items():
        if isinstance(sub, dict):
            section = cfg.setdefault(key, {})
            for sk, sv in sub.items():
                section.setdefault(sk, sv)
        else:
            cfg.setdefault(key, sub)
    return cfg


_DISPLAY_CONFIG: dict = _load_display_config()


def reload_display_config() -> None:
    """Reload MolView display configuration JSON into the global cache.

    Existing :class:`MolView` widgets read from the module-level
    :data:`_DISPLAY_CONFIG` inside :meth:`MolView._update_view`, so they
    will pick up new parameters on the next redraw after this function is
    called.
    """

    global _DISPLAY_CONFIG
    _DISPLAY_CONFIG = _load_display_config()
