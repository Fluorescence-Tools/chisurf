"""Display configuration loading for Chimol."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

try:
    import chisurf.core.settings as _cs_settings
except Exception:  # pragma: no cover - moview can run without chisurf
    _cs_settings = None

DISPLAY_CONFIG_VERSION: int = 2
"""Current version of the chimol_display.json schema.

Increment this when keys are added, renamed, or removed so that users
with an older copy in ``~/.chisurf/`` are prompted to update.
"""


_update_listeners: list[Callable[[], None]] = []
"""Registered callbacks to notify when display config is reloaded."""


def register_update_listener(listener: Callable[[], None]) -> None:
    """Register a callback invoked after each config reload."""
    _update_listeners.append(listener)


def unregister_update_listener(listener: Callable[[], None]) -> None:
    """Remove a previously registered callback."""
    try:
        _update_listeners.remove(listener)
    except ValueError:
        pass


def get_package_display_config_path() -> Path:
    """Return the path to the chimol_display.json shipped with the package."""
    return Path(__file__).with_name("chimol_display.json")


def get_user_display_config_path() -> Path | None:
    """Return the expected user ``chimol_display.json`` path, or ``None``.

    When chisurf is available this is ``~/.chisurf/chimol_display.json``;
    otherwise ``None`` is returned (standalone Chimol uses the package copy).
    """
    if _cs_settings is not None:
        try:
            return _cs_settings.get_path("settings") / "chimol_display.json"
        except Exception:
            return None
    return None


def check_for_display_config_update() -> bool:
    """Return ``True`` if the user's ``chimol_display.json`` is outdated.

    Compares the ``_version`` field in the user copy (if any) against
    :data:`DISPLAY_CONFIG_VERSION`.  Returns ``False`` when there is no
    user copy or when the versions match.
    """
    user_path = get_user_display_config_path()
    if user_path is None or not user_path.is_file():
        return False
    try:
        with user_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        user_version = cfg.get("_version", 0)
        return user_version < DISPLAY_CONFIG_VERSION
    except Exception:
        return False


def _load_display_config() -> dict:
    """Load Chimol display configuration from JSON file.

    All tunable visual parameters (cartoon radius, AO strength, ball sizes,
    etc.) are collected in ``chimol_display.json``. If the file cannot be
    read, sensible defaults are used. For backward compatibility, legacy
    ``molview_display.json`` or ``protview_display.json`` are also accepted
    if present.
    """

    default = {
        "_version": DISPLAY_CONFIG_VERSION,
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
            "radius_scale": 0.01,
            "min_radius": 0.5,
            "segments_circle": 18,
            "subdivisions": 7,
            "cartoon_sampling": 7,
            "tube_radius": 0.5,
            "tube_quality": 18,
            "ao_radius": 4.0,
            "ao_max_neighbors": 16,
            "ao_strength": 0.45,
            "style": "ribbon",
            "profile_segments": 20,
            # Slight tension to reduce wiggly beta strands; 0 = classic Catmull-Rom
            "spline_tension": 0.3,
            # PyMOL-style dimensions in scene units, not scaled by molecule size.
            "loop_radius": 0.2,
            "loop_quality": 14,
            "rect_width": 0.4,
            "rect_length": 1.4,
            "oval_width": 0.25,
            "oval_length": 1.35,
            "oval_quality": 20,
            "arrow_sampling": 2,
            # Legacy ss_shapes kept for backward compatibility
            "ss_shapes": {
                "helix": {"width": 0.5, "thickness": 1.35, "profile_power": 1.6},
                "strand": {"width": 1.8, "thickness": 1.5,
                           "arrow_scale": 0.2, "profile_power": 18.0},
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
            "alpha": 0.85,
            "max_points": 10000,
            "color_mode": "ao_gray",
            "base_color": [0.85, 0.85, 0.92, 1.0],
            "grid_spacing": 0.8,
            "iso_value": 0.5,
            "padding": 3.0,
            "max_dim": 96,
            "mesh_sigma_factor": 1.0,
            "mesh_sigma_default": 1.8,
            "method": "gaussian",
            "probe_radius": 1.4,
        },
        "metaball": {
            # Density field function: "wyvill" (compact support, faster) or "gaussian"
            "field_function": "wyvill",
            # Isosurface threshold for marching cubes (lower = larger surface)
            "iso_value": 0.15,
            # Grid resolution in Angstroms (smaller = finer mesh, slower)
            "grid_spacing": 0.6,
            # Extra space around bounding box in Angstroms
            "padding": 5.0,
            # Maximum grid dimension (auto-coarsens spacing if exceeded)
            "max_dim": 128,
            # Mesh transparency (1.0 = opaque, <1.0 = transparent)
            "alpha": 0.6,
            # Ambient occlusion strength (0.0 = off, 1.0 = maximum darkening in crevices)
            "ao_strength": 0.6,
            # AO search radius in Angstroms (larger = broader shadows)
            "ao_radius": 4.5,
            # Material shininess (higher = sharper specular highlights)
            "shininess": 40.0,
            # Specular highlight intensity (0.0 = matte, 1.0 = mirror-like)
            "specular_strength": 0.3,
            # Rim lighting strength (edge glow effect)
            "rim_strength": 0.3,
            # Rim lighting falloff power (higher = sharper edge)
            "rim_power": 4.0,
            # Use only surface-exposed atoms (faster, cleaner surface)
            "surface_only": True,
            # Neighbor search radius for surface classification (Angstroms)
            "surface_radius": 5.0,
            # Max neighbors to be considered surface-exposed
            "surface_max_neighbors": 20,
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
            "element_cpk": {
                "H": [1.0, 1.0, 1.0, 1.0],
                "C": [0.5, 0.5, 0.5, 1.0],
                "N": [0.0, 0.0, 1.0, 1.0],
                "O": [1.0, 0.0, 0.0, 1.0],
                "S": [1.0, 1.0, 0.0, 1.0],
                "P": [1.0, 0.65, 0.0, 1.0],
                "default": [0.8, 0.8, 0.8, 1.0],
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
            # PyMOL-compatible sequence viewer globals
            "seq_view": True,
            "seq_view_gap_mode": 1,
            "seq_view_label_spacing": 5,
            "seq_view_format": 0,
            "seq_view_color": -1,
            "seq_view_fill_color": -1,
            "seq_view_label_color": [1.0, 1.0, 1.0, 1.0],
            "seq_view_overlay": False,
        },
        "backbone_trace": {
            "protein_atoms": ["CA"],
            "nucleic_atoms": ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "C1*"],
        },
        "lighting": {
            "light_direction": [0.0, 0.0, 1.0],
            "ambient_strength": 0.45,
            "specular_strength": 0.25,
            "shininess": 40.0,
            "rim_strength": 0.18,
            "rim_power": 2.4,
        },
        "camera": {
            "near_clip": 0.03,
            "far_clip": 2000.0,
            "min_near_clip": 0.005,
            "max_near_clip": 5.0,
            "clip_wheel_scale": 0.85,
            # Mouse interaction style: "pymol" rotates and pans the object in
            # the camera view (intuitive, follows the cursor); "chimol"
            # rotates and pans the camera/plane so the object moves opposite
            # to the cursor.
            "mouse_mode": "pymol",
        },
        "ray": {
            "ambient": 0.14,
            "diffuse": 0.45,
            "reflect_power": 1.0,
            "specular": 0.25,
            "shininess": 40.0,
            "direct_specular": 0.30,
            "direct_specular_power": 55.0,
            "legacy_lighting": 0.0,
            "antialias": 2,
            "shadow": True,
            "shadow_fudge": 0.001,
            "shadow_decay_factor": 0.2,
            "shadow_decay_range": 1.8,
            "gamma": 2.2,
            "depth_cue": True,
            "fog_start": 0.45,
            "fog_intensity": 1.0,
            "color_blend": True,
            "color_blend_red": 0.17,
            "color_blend_green": 0.25,
            "color_blend_blue": 0.14,
            "light_directions": [
                [0.0, 0.0, 1.0],
                [0.5, 0.3, 1.0]
            ],
        },
        # PyMOL Global Settings (flat namespace)
        # Category: General / Viewport / Camera / Fog
        "bg_rgb": [0.0, 0.0, 0.0],  # Background color of the viewer window.
        "orthoscopic": False,  # Controls whether perspective projection (False) or orthoscopic projection (True) is used.
        "field_of_view": 20.0,  # Vertical field of view in degrees.
        "depth_cue": True,  # Controls whether or not a depth-cue fog effect is used.
        "fog": 1.0,  # Fog density level.
        "fog_start": 0.45,  # Depth coordinate where fog begins.

        # Category: Cartoon Representation
        "cartoon_color": -1,  # Color index of cartoons (-1 = default to atom colors).
        "cartoon_transparency": 0.0,  # Transparency level of cartoons (0.0 = opaque, 1.0 = invisible).
        "cartoon_loop_radius": 0.2,  # Radius of loop segments.
        "cartoon_tube_radius": 0.5,  # Radius of tube segments.
        "cartoon_oval_width": 0.25,  # Width/thickness of oval cartoon profiles (used for alpha helices).
        "cartoon_oval_length": 1.35,  # Length/width of oval cartoon profiles.
        "cartoon_rect_width": 0.4,  # Thickness of rectangular cartoon profiles (used for beta sheets).
        "cartoon_rect_length": 1.4,  # Width of rectangular cartoon profiles.
        "cartoon_fancy_helices": False,  # Whether or not dumbbell/fancy helices are drawn.
        "cartoon_fancy_sheets": False,  # Whether or not beta strands end in fancy arrows.
        "cartoon_flat_sheets": True,  # Whether or not beta strands are flattened.
        "cartoon_smooth_loops": False,  # Whether or not loops are smoothed.
        "cartoon_trace_atoms": False,  # Whether or not cartoons trace through all guide C-alpha atoms.

        # Category: Sphere / Ball Representation
        "sphere_color": -1,  # Color index of sphere representations (-1 = default to atom colors).
        "sphere_scale": 1.0,  # Scale multiplier for sphere representation radii.

        # Category: Stick / Bond Representation
        "stick_color": -1,  # Color index of stick representation (-1 = default to atom/bond colors).
        "stick_radius": 0.25,  # Radius of cylinders used for stick representation.
        "stick_transparency": 0.0,  # Transparency level of sticks.

        # Category: Line Representation
        "line_color": -1,  # Color index of line representation (-1 = default to atom colors).
        "line_width": 1.4,  # Width in pixels of lines.

        # Category: Ribbon Representation
        "ribbon_color": -1,  # Color index of ribbon representation (-1 = default to atom colors).
        "ribbon_width": 0.75,  # Width of ribbons.

        # Category: Raytracing / Lighting
        "ray_trace_mode": 0,  # Raytracing outline mode: 0=normal, 1=outlines, 2=outlines only.
        "ray_trace_frames": 0,  # Controls whether frames are ray-traced during movie compilation.
        "ray_shadow": True,  # Controls whether shadows are cast during raytracing.
        "specular": 0.5,  # Intensity of specular highlights.
        "shininess": 55.0,  # Exponent/power of specular reflections.
        "ambient": 0.2,  # Strength of ambient lighting.
        "direct": 0.45,  # Camera direct light source strength.
        "light_count": 2,  # Number of active light sources.
    }

    # Prefer a JSON file in the global chisurf settings folder so the user
    # can override display options without touching the source tree. When
    # chisurf is not available (standalone Chimol), fall back to a JSON file
    # shipped next to this module.
    package_path = get_package_display_config_path()
    try:
        override_env = os.environ.get("CHIMOL_DISPLAY_CONFIG")
        if override_env:
            override_path = Path(override_env)
            if override_path.is_file():
                path = override_path
            else:
                path = package_path
        elif _cs_settings is not None:
            settings_dir = _cs_settings.get_path("settings")
            user_path = settings_dir / "chimol_display.json"
            if not user_path.is_file():
                legacy = settings_dir / "molview_display.json"
                if legacy.is_file():
                    path = legacy
                else:
                    legacy2 = settings_dir / "protview_display.json"
                    path = legacy2 if legacy2.is_file() else user_path
                # If no user or legacy file exists, copy the package default.
                if not path.is_file() and package_path.is_file():
                    settings_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(package_path, user_path)
                    path = user_path
                elif not path.is_file():
                    path = package_path
            else:
                path = user_path
        else:
            path = package_path
    except Exception:
        path = package_path

    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return default

    # Track the user copy version for the update-prompt feature.
    global _DISPLAY_CONFIG_USER_VERSION
    _DISPLAY_CONFIG_USER_VERSION = cfg.get("_version", 0)

    # Strip meta keys that should not leak into the rendered config.
    cfg.pop("_version", None)

    # Shallow-merge user config with defaults to ensure all keys exist.
    for key, sub in default.items():
        if key == "_version":
            continue
        if isinstance(sub, dict):
            section = cfg.setdefault(key, {})
            for sk, sv in sub.items():
                section.setdefault(sk, sv)
        else:
            cfg.setdefault(key, sub)
    return cfg


_DISPLAY_CONFIG_USER_VERSION: int = 0
"""Version number read from the user's ``chimol_display.json``, or 0."""

_DISPLAY_CONFIG_PACKAGE_VERSION: int = DISPLAY_CONFIG_VERSION
"""Version number shipped with the package."""

_DISPLAY_CONFIG: dict = _load_display_config()


def reload_display_config() -> None:
    """Reload MolView display configuration JSON into the global cache.

    Existing :class:`MolView` widgets read from the module-level
    :data:`_DISPLAY_CONFIG` inside :meth:`MolView._update_view`, so they
    will pick up new parameters on the next redraw after this function is
    called.

    After reloading, all registered :data:`_update_listeners` are invoked so
    that open viewers recompute their representations.
    """

    global _DISPLAY_CONFIG
    _DISPLAY_CONFIG = _load_display_config()
    for listener in list(_update_listeners):
        try:
            listener()
        except Exception:
            pass
