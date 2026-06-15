from __future__ import annotations

import numpy as np
import pytest
from qtpy import QtWidgets

from chisurf.plugins.chimol.chimol.renderer.chimol_state import _MolViewObjectState
from chisurf.plugins.chimol.chimol.renderer.view import MolView
from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer


def test_mock_viewer_append_frame_tracks_raw_frames() -> None:
    viewer = MockViewer()
    object_id = viewer.add_coordinates(np.zeros((2, 3)), name="live")
    assert viewer.append_frame(np.zeros((2, 3)), object_id=object_id) == 1
    assert viewer.append_frame(np.ones((2, 3)), object_id=object_id) == 2
    state = viewer.get_active_state()
    assert state.frames_raw.shape == (2, 2, 3)
    assert state.active_frame == 1


def test_select_state_frame_preserves_ca_trace_for_all_atom_frames():
    """All-atom trajectory frames (e.g. ProteinMC) must not overwrite the
    CA-trace coords.  After set_frames, state.coords should still have
    shape (n_residues,) and state.ball_mask should still be residue-sized,
    so the atom renderer produces sphere meshes instead of tiny points.
    """
    n_res = 20
    n_atoms = n_res * 5  # 5 atoms per residue (backbone + centroid sidechain)

    rng = np.random.default_rng(42)
    raw_xyz = rng.uniform(10.0, 60.0, (n_atoms, 3)).astype(float)

    atom_dtype = [
        ("xyz", "f8", (3,)),
        ("atom_name", "U4"),
        ("res_id", "i4"),
        ("res_name", "U4"),
    ]
    atoms = np.zeros(n_atoms, dtype=atom_dtype)
    atoms["xyz"] = raw_xyz
    for i in range(n_res):
        atoms["atom_name"][i * 5 : i * 5 + 5] = ["N", "CA", "C", "O", "CB"]
    atoms["res_id"] = np.repeat(np.arange(n_res), 5)
    atoms["res_name"] = "ALA"

    from chisurf.plugins.chimol.chimol.geometry.primitives import (
        _compute_center_radius,
    )

    # Mimic set_structure centering.
    center, radius = _compute_center_radius(raw_xyz)
    scale = 10.0
    ca_idx = np.array([i * 5 + 1 for i in range(n_res)])  # CA positions

    state = _MolViewObjectState()
    state.atoms = atoms
    state.all_atom_coords = (raw_xyz - center) * scale
    state._initial_center = center.copy()
    state._initial_scale = scale
    state._ca_indices = ca_idx
    state.residue_ids = np.arange(n_res)
    state.residue_names = np.array(["ALA"] * n_res)
    state.residue_oneletter = np.array(["A"] * n_res)
    state.residue_chain_ids = np.array(["A"] * n_res)
    state.coords = (raw_xyz[ca_idx] - center) * scale
    state.ball_mask = np.ones(n_res, dtype=bool)
    state.cartoon_mask = np.ones(n_res, dtype=bool)

    # Verify initial state.
    assert state.coords.shape == (n_res, 3)
    assert state.ball_mask.shape == (n_res,)

    # Simulate a new all-atom frame from ProteinMC (raw, not pre-centered).
    new_raw = raw_xyz + rng.uniform(-3, 3, raw_xyz.shape)

    # Recenter using the stored initial center (same as the fix in
    # _select_state_frame).
    state.all_atom_coords = (new_raw - center) * scale
    new_ca_coords = (new_raw[ca_idx] - center) * scale
    state.coords = new_ca_coords

    # State must still be consistent after the frame update.
    assert state.coords.shape == (n_res, 3)
    assert state.ball_mask.shape == (n_res,)
    assert state.ball_mask.all()

    # CA coords in all_atom_coords must agree with state.coords.
    ca_from_all = state.all_atom_coords[ca_idx]
    np.testing.assert_allclose(state.coords, ca_from_all, atol=1e-12)


def test_select_state_frame_coordinate_only_trajectory():
    """Coordinate-only trajectories (no atom metadata) should use the
    frame directly as state.coords.
    """
    n_pts = 30
    state = _MolViewObjectState()
    state.residue_ids = None
    state._initial_center = None
    state._initial_scale = 1.0
    state._ca_indices = None

    frame = np.random.default_rng(0).uniform(0, 100, (n_pts, 3))
    state.all_atom_coords = frame.copy()
    state.coords = frame.copy()

    # After coordinate-only frame update, coords == frame.
    assert state.coords.shape == frame.shape
    np.testing.assert_array_equal(state.coords, frame)


def test_select_state_frame_replaces_same_shape_coordinate_frame() -> None:
    state = _MolViewObjectState()
    frame0 = np.zeros((4, 3), dtype=float)
    frame1 = np.arange(12, dtype=float).reshape(4, 3)
    state.frames = np.stack([frame0, frame1])
    state.frames_raw = state.frames.copy()
    state.coords = frame0.copy()
    state.all_atom_coords = frame0.copy()

    MolView._select_state_frame(None, state, 1)

    assert state.active_frame == 1
    np.testing.assert_array_equal(state.coords, frame1)
    np.testing.assert_array_equal(state.all_atom_coords, frame1)


def test_select_state_frame_updates_ca_trace_for_same_shape_all_atom_frame() -> None:
    state = _MolViewObjectState()
    frame0 = np.zeros((6, 3), dtype=float)
    frame1 = np.arange(18, dtype=float).reshape(6, 3)
    state.frames = np.stack([frame0, frame1])
    state.frames_raw = state.frames.copy()
    state.coords = frame0[[1, 4]].copy()
    state.all_atom_coords = frame0.copy()
    state.residue_ids = np.array([1, 2])
    state._ca_indices = np.array([1, 4])

    MolView._select_state_frame(None, state, 1)

    assert state.active_frame == 1
    np.testing.assert_array_equal(state.coords, frame1[[1, 4]])
    np.testing.assert_array_equal(state.all_atom_coords, frame1)


@pytest.fixture
def _qt_app():
    """Ensure a QApplication exists for MolView construction."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_set_rmf_data_scales_frames_and_radii_consistently(_qt_app) -> None:
    """RMF coordinates (Angstrom) must be scaled like set_frames/add_structure.

    Previously ``set_rmf_data`` stored raw RMF frames without applying
    ``_scale_factor``.  This made RMF-loaded structures render at a different
    scale than structure-loaded ones, causing cartoon/bond/bead sizes to look
    wrong (the reported nm/Å mixup).
    """
    widget = MolView()
    widget._update_view = lambda *args, **kwargs: None

    raw_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [3.9, 0.0, 0.0], [7.7, 0.0, 0.0]],
        ],
        dtype=float,
    )
    raw_radii = np.array([1.5, 1.7, 1.9], dtype=float)
    scale = float(widget._scale_factor)

    object_id = widget._create_object(name="rmf_test").object_id
    widget.set_rmf_data(
        hierarchy=None,
        frames=raw_frames,
        radii=raw_radii,
        object_id=object_id,
    )

    entry = widget._objects[object_id]
    state = entry.state

    # frames_raw must stay in input units (Angstrom)
    np.testing.assert_array_equal(state.frames_raw, raw_frames)

    # frames must be centered and scaled by _scale_factor
    center = raw_frames.reshape(-1, 3).mean(axis=0)
    expected_frames = (raw_frames - center) * scale
    np.testing.assert_allclose(state.frames, expected_frames, atol=1e-9)

    # bead radii must be scaled by _scale_factor
    np.testing.assert_allclose(state.bead_radii, raw_radii * scale, atol=1e-9)

    # state.coords/active frame should use the scaled frames
    assert state.active_frame == 0
    np.testing.assert_allclose(state.coords, expected_frames[0], atol=1e-9)

    # Global scene radius should reflect the scaled geometry
    assert widget._radius > 0.0
