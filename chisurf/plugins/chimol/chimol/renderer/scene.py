from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Geometry:
    """Backend-agnostic geometric data for a single draw call.

    All arrays are NumPy arrays in a standard layout so they can be uploaded
    to any rendering backend (Qt OpenGL, WebGPU, WebGL, etc.) without tying
    Moview to a specific library.
    """

    kind: str  # "mesh" | "line" | "points" | "text"
    positions: np.ndarray
    indices: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    radii: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta["labels"] -> list[str] for kind == "text"


@dataclass
class SceneObject:
    """A logical object in the scene (cartoon, atoms, surface, overlays)."""

    id: str
    geometry: Geometry
    render_mode: str = "opaque"  # "opaque" | "transparent" | "overlay"


@dataclass
class Scene:
    """Complete scene description for the protein viewer.

    The scene is constructed from ChiSurf/Moview geometry helpers and can be
    consumed by any rendering backend. Camera parameters are stored here so
    that CPU-side picking and different GL backends see a consistent view.
    """

    objects: List[SceneObject] = field(default_factory=list)
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    radius: float = 1.0
