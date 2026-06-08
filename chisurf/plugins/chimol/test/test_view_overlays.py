"""Tests for Chimol MolView point-cloud overlay public API.
"""

import numpy as np
import pytest
from qtpy import QtWidgets
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
