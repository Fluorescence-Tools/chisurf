"""Unit tests for Phase 2: refine, bootstrap, RMSD, and output writers."""

import os
import tempfile
import numpy as np
import pytest

from ..core import io as _io
from ..core import av as _av
from ..core.refine import run_refinement
from ..core.bootstrap import run_bootstrap
from ..core.engine import RigidBody, DistanceRestraint, SpringParameters
from ..core.results import write_pymol_pml, write_r_table, write_chi2_table, SimulationResult


def _get_test_data():
    """Helper to load a filtered subset of real T4L FPS/PDB example data."""
    # Paths in workspace
    pdb_path = "/Users/tpeulen/dev/chisurf/examples/projects/t4l_chimol/data/3GUN.pdb"
    json_path = "/Users/tpeulen/dev/chisurf/examples/projects/t4l_chimol/data/fret.fps.json"

    if not os.path.exists(pdb_path) or not os.path.exists(json_path):
        pytest.skip("T4L example data not found in workspace")

    # Read full json
    positions, distances, score_sets, extra = _io.read_fps_json(json_path)

    # Filter down to 2 positions and 1 distance for fast testing
    subset_positions = {
        "19D": positions["19D"],
        "132A": positions["132A"]
    }
    subset_distances = {
        "19-132_C3": distances["19-132_C3"]
    }

    # Assign distinct body IDs
    subset_positions["19D"]["body_id"] = 0
    subset_positions["132A"]["body_id"] = 1

    return pdb_path, subset_positions, subset_distances


def test_refinement_one_cycle_no_error():
    """Verify run_refinement completes a cycle without errors."""
    pdb_path, positions, distances = _get_test_data()
    atoms_xyzr = _av.load_structure_with_vdw(pdb_path)

    com = atoms_xyzr[:, :3].mean(axis=0)
    body_a = RigidBody(
        name="body_0",
        atoms_local=atoms_xyzr.copy(),
        com=com.copy(),
        rotation=np.eye(3),
        translation=com.copy(),
    )
    body_b = RigidBody(
        name="body_1",
        atoms_local=atoms_xyzr.copy(),
        com=com.copy() + 5.0,
        rotation=np.eye(3),
        translation=com.copy() + 5.0,
    )

    avs = _av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=pdb_path, disc_step=2.5)
    offset_a = avs["19D"].mean_position - body_a.com
    offset_b = avs["132A"].mean_position - body_b.com

    rst = DistanceRestraint(
        name="19-132_C3",
        body_a=0,
        offset_a=offset_a,
        body_b=1,
        offset_b=offset_b,
        distance_exp=float(distances["19-132_C3"]["distance"]),
        error_neg=float(distances["19-132_C3"]["error_neg"]),
        error_pos=float(distances["19-132_C3"]["error_pos"]),
    )

    params = SpringParameters(max_iterations=10)
    refined = run_refinement(
        bodies=[body_a, body_b],
        restraints=[rst],
        positions=positions,
        atoms_xyzr=atoms_xyzr,
        params=params,
        n_cycles=1,
        disc_step=2.5,
    )
    assert len(refined) == 2
    assert isinstance(refined[0], RigidBody)


def test_refinement_returns_list_of_rigid_bodies():
    """Verify run_refinement returns the list of RigidBody structures."""
    pdb_path, positions, distances = _get_test_data()
    atoms_xyzr = _av.load_structure_with_vdw(pdb_path)
    com = atoms_xyzr[:, :3].mean(axis=0)
    body_a = RigidBody(
        name="body_0",
        atoms_local=atoms_xyzr.copy(),
        com=com,
        rotation=np.eye(3),
        translation=com,
    )
    params = SpringParameters(max_iterations=2)
    refined = run_refinement(
        bodies=[body_a],
        restraints=[],
        positions=positions,
        atoms_xyzr=atoms_xyzr,
        params=params,
        n_cycles=1,
        disc_step=2.5,
    )
    assert len(refined) == 1
    assert refined[0].name == "body_0"


def test_bootstrap_n_iterations_correct():
    """Verify run_bootstrap performs the correct number of iterations."""
    pdb_path, positions, distances = _get_test_data()
    params = SpringParameters(max_iterations=5)
    res = run_bootstrap(
        pdb_path=pdb_path,
        positions=positions,
        distances=distances,
        params=params,
        n_bootstrap=2,
        disc_step=2.5,
    )
    assert res.n_bootstrap == 2
    assert len(res.energies) == 2


def test_bootstrap_model_distances_mean_finite():
    """Verify bootstrap mean model distances are finite numbers."""
    pdb_path, positions, distances = _get_test_data()
    params = SpringParameters(max_iterations=5)
    res = run_bootstrap(
        pdb_path=pdb_path,
        positions=positions,
        distances=distances,
        params=params,
        n_bootstrap=2,
        disc_step=2.5,
    )
    for k in distances.keys():
        assert np.isfinite(res.model_distances_mean[k])
        assert np.isfinite(res.model_distances_std[k])


def test_bootstrap_translations_shape():
    """Verify bootstrap translations and rotations shapes match expectations."""
    pdb_path, positions, distances = _get_test_data()
    params = SpringParameters(max_iterations=5)
    res = run_bootstrap(
        pdb_path=pdb_path,
        positions=positions,
        distances=distances,
        params=params,
        n_bootstrap=2,
        disc_step=2.5,
    )
    assert len(res.translations) == 2
    assert len(res.rotations) == 2
    # Check shape of first iteration, first body translation
    assert res.translations[0][0].shape == (3,)
    assert res.rotations[0][0].shape == (3, 3)


def test_rmsd_identity_is_zero():
    """RMSD between identical coordinate arrays must be zero."""
    coords = np.random.rand(10, 3)
    assert abs(_io.compute_rmsd(coords, coords)) < 1e-7


def test_rmsd_translation_known_value():
    """Verify RMSD computation on known translated point."""
    coords_a = np.zeros((1, 3))
    coords_b = np.array([[3.0, 4.0, 0.0]])
    assert abs(_io.compute_rmsd(coords_a, coords_b) - 5.0) < 1e-7


def test_rmsd_kabsch_removes_rotation():
    """Verify Kabsch alignment resolves known rigid-body rotation to near-zero RMSD."""
    coords_a = np.random.rand(10, 3)
    theta = 0.5
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0]
    ])
    coords_b = coords_a @ R.T

    # Direct comparison should have substantial RMSD
    assert _io.compute_rmsd(coords_a, coords_b, superpose=False) > 0.1
    # Kabsch-aligned comparison should recover near-zero RMSD
    assert abs(_io.compute_rmsd(coords_a, coords_b, superpose=True)) < 1e-7


def test_write_pymol_pml_contains_load_command():
    """Verify PyMOL PML output writer writes correct PDB file load paths."""
    with tempfile.NamedTemporaryFile(suffix=".pml", delete=False) as f:
        path = f.name
    try:
        write_pymol_pml([], [np.zeros((3, 3))], path, pdb_prefix="my_dock")
        with open(path) as f:
            content = f.read()
        assert "load my_dock_body_0.pdb, my_dock_body_0" in content
    finally:
        os.unlink(path)


def test_write_r_table_tab_separated_with_header():
    """Verify R table is tab-separated with expected header & formatted values."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    try:
        sr = SimulationResult(model_distances={"R1": 15.23456})
        write_r_table([sr], {"R1": {}}, path)
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines[0] == "trial\tR1"
        assert lines[1] == "0\t15.2346"
    finally:
        os.unlink(path)


def test_write_chi2_table_tab_separated_with_header():
    """Verify chi2 table output contains per-restraint chi2 and total chi2 columns."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    try:
        sr = SimulationResult(model_distances={"R1": 10.0})
        # dev = 10 - 8 = 2. err_pos = 2. chi2 = (2/2)^2 = 1.0
        write_chi2_table([sr], {"R1": {"distance": 8.0, "error_neg": 2.0, "error_pos": 2.0}}, None, path)
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines[0] == "trial\tR1\tchi2_total"
        assert lines[1] == "0\t1.0000\t1.0000"
    finally:
        os.unlink(path)
