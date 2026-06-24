"""Tests for Chimol MolView point-cloud overlay public API."""

from types import MethodType

import numpy as np
import pytest
from qtpy import QtWidgets

from chisurf.plugins.chimol.chimol.geometry.surface import (
    _generate_surface_mesh_from_points,
)
from chisurf.plugins.chimol.chimol.renderer.view import MolView


@pytest.fixture
def qt_app():
    # Make sure a QApplication exists
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_add_point_overlay_updates_dict(qt_app):
    widget = MolView()
    # Mock _update_view to avoid OpenGL calls in headless pytest sessions
    called = []
    widget._update_view = lambda *args, **kwargs: called.append(True)

    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    widget.add_point_overlay("test_key", coords, color=(1.0, 0.0, 0.0, 1.0))

    assert "test_key" in widget._point_overlays
    assert np.allclose(widget._point_overlays["test_key"]["coords"], coords)
    assert widget._point_overlays["test_key"]["color"] == (1.0, 0.0, 0.0, 1.0)
    assert len(called) > 0


def test_remove_point_overlay(qt_app):
    widget = MolView()
    widget._update_view = lambda *args, **kwargs: None

    coords = np.array([[1.0, 2.0, 3.0]])
    widget.add_point_overlay("key_to_remove", coords)

    assert "key_to_remove" in widget._point_overlays
    removed = widget.remove_point_overlay("key_to_remove")
    assert removed is True
    assert "key_to_remove" not in widget._point_overlays

    removed_nonexistent = widget.remove_point_overlay("nonexistent")
    assert removed_nonexistent is False


def test_clear_point_overlays(qt_app):
    widget = MolView()
    widget._update_view = lambda *args, **kwargs: None

    coords = np.array([[1.0, 2.0, 3.0]])
    widget.add_point_overlay("k1", coords)
    widget.add_point_overlay("k2", coords)

    assert len(widget._point_overlays) == 2
    widget.clear_point_overlays()
    assert len(widget._point_overlays) == 0


def test_add_sphere_returns_key_string(qt_app):
    widget = MolView()
    widget._update_view = lambda *args, **kwargs: None

    center = np.array([1.0, 2.0, 3.0])
    key = widget.add_sphere(center, radius=2.5, color=(0.0, 1.0, 0.0, 1.0), label="My Sphere")

    assert key.startswith("sphere_")
    assert key in widget._point_overlays
    entry = widget._point_overlays[key]
    assert np.allclose(entry["coords"], center.reshape(1, 3))
    assert entry["min_size"] == 5.0  # 2 * radius
    assert entry["label"] == "My Sphere"


def test_generate_surface_mesh_from_points_for_av_cloud():
    grid = np.linspace(-2.0, 2.0, 9)
    coords = np.array(
        [
            [x, y, z]
            for x in grid
            for y in grid
            for z in grid
            if x * x + y * y + z * z <= 4.0
        ],
        dtype=float,
    )

    mesh = _generate_surface_mesh_from_points(
        coords,
        grid_spacing=0.5,
        padding=1.0,
        max_dim=48,
    )

    assert mesh is not None
    verts, faces, norms = mesh
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert norms.shape == verts.shape
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0


def test_add_surface_overlay_builds_mesh_scene_object():
    class DummyMolView:
        """Minimal object for testing overlay scene construction."""

    widget = DummyMolView()
    widget._point_overlays = {}
    widget._radius = 10.0
    widget._base_color_single = np.array([1.0, 1.0, 1.0, 1.0])
    widget._update_view = lambda *args, **kwargs: None
    widget._world_to_scene_scale = MethodType(
        MolView._world_to_scene_scale,
        widget,
    )
    widget._transform_world_coords_to_scene = MethodType(
        MolView._transform_world_coords_to_scene,
        widget,
    )
    widget._build_surface_overlay_scene = MethodType(
        MolView._build_surface_overlay_scene,
        widget,
    )

    grid = np.linspace(-1.5, 1.5, 7)
    coords = np.array(
        [
            [x, y, z]
            for x in grid
            for y in grid
            for z in grid
            if x * x + y * y + z * z <= 2.25
        ],
        dtype=float,
    )
    MolView.add_surface_overlay(
        widget,
        "av",
        coords,
        color=(0.0, 1.0, 0.0, 0.6),
        grid_spacing=0.5,
        padding=1.0,
        max_dim=48,
    )

    assert widget._point_overlays["av"]["overlay_kind"] == "surface"
    scene_objects = MolView._update_custom_overlays(widget, {})
    assert scene_objects
    surface = scene_objects[0]
    assert surface.geometry.kind == "mesh"
    assert surface.render_mode == "transparent"


def test_surface_overlay_transforms_angstrom_coords_to_scene_units():
    class DummyMolView:
        """Minimal object for testing overlay coordinate transforms."""

    widget = DummyMolView()
    widget._point_overlays = {}
    widget._raw_center = np.array([1.0, 2.0, 3.0], dtype=float)
    widget._scale_factor = 10.0
    widget._update_view = lambda *args, **kwargs: None
    widget._world_to_scene_scale = MethodType(
        MolView._world_to_scene_scale,
        widget,
    )
    widget._transform_world_coords_to_scene = MethodType(
        MolView._transform_world_coords_to_scene,
        widget,
    )

    coords = np.array([[2.0, 4.0, 6.0], [1.5, 2.5, 3.5]], dtype=float)
    MolView.add_surface_overlay(
        widget,
        "av",
        coords,
        grid_spacing=0.5,
        padding=1.0,
        fallback_min_size=1.5,
    )

    overlay = widget._point_overlays["av"]
    assert np.allclose(
        overlay["coords"],
        np.array([[10.0, 20.0, 30.0], [5.0, 5.0, 5.0]], dtype=float),
    )
    assert overlay["grid_spacing"] == 5.0
    assert overlay["padding"] == 10.0
    assert overlay["min_size"] == 15.0
