from __future__ import annotations

import abc
from typing import Optional

from qtpy import QtWidgets

from .scene import Scene


class Renderer:
    """Abstract interface for Moview rendering backends.

    This class deliberately avoids using :class:`abc.ABC` as a metaclass
    so that renderer implementations can safely inherit from Qt widget
    classes (e.g. :class:`QOpenGLWidget`) without metaclass conflicts. The
    methods still behave like abstract methods by raising
    :class:`NotImplementedError` by default.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._parent = parent

    def widget(self) -> QtWidgets.QWidget:
        """Return the QWidget that should be embedded into the UI."""

        raise NotImplementedError

    def set_scene(self, scene: Optional[Scene]) -> None:
        """Upload a new scene description to the renderer."""

        raise NotImplementedError

    def clear(self) -> None:
        """Clear any previously rendered geometry."""

        # Optional hook; concrete renderers may override.
        return None

    def configure_grid(self, size: float, spacing: float) -> None:
        """Define the logical grid dimensions for the renderer."""

        raise NotImplementedError

    def set_background_color(self, color) -> None:
        """Set the renderer's background color (Qt-compatible value)."""

        raise NotImplementedError

    def set_grid_visible(self, visible: bool) -> None:
        """Toggle visibility of the ground grid / reference plane."""

        raise NotImplementedError

    def fit_to_radius(self, radius: float) -> None:
        """Adjust camera distance to comfortably fit the given radius."""

        raise NotImplementedError

    def reset_view(self, distance: float, elevation: float, azimuth: float) -> None:
        """Reset the camera to the provided spherical coordinates."""

        raise NotImplementedError

    def configure_camera(
        self,
        *,
        near_clip: float,
        far_clip: float,
        min_near_clip: float,
        max_near_clip: float,
        clip_wheel_scale: float,
    ) -> None:
        """Configure camera clipping planes and interaction parameters."""

        # Optional hook for renderers that expose clip controls.
        return None
