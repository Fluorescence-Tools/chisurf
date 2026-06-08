"""Tests for the FRET Docking & Screening plugin."""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from ..core import distance as _dist
from ..core import io as _io
from ..core import clash as _clash


# ---------------------------------------------------------------------------
# io tests
# ---------------------------------------------------------------------------

def test_write_read_fps_json():
    """Round-trip an fps.json file."""
    positions = {
        "A1": {
            "atom_name": "CA",
            "chain_identifier": "A",
            "residue_seq_number": 1,
            "linker_length": 20.0,
            "linker_width": 1.0,
            "radius1": 3.5,
            "simulation_grid_resolution": 1.5,
            "simulation_type": "AV1",
        }
    }
    distances = {
        "A1_A2": {
            "position1_name": "A1",
            "position2_name": "A2",
            "distance": 45.0,
            "error_neg": 5.0,
            "error_pos": 5.0,
            "distance_type": "RDAMean",
            "Forster_radius": 52.0,
        }
    }
    score_sets = {"chi2_all": {"distances": ["A1_A2"]}}

    with tempfile.NamedTemporaryFile(suffix=".fps.json", mode="w", delete=False) as f:
        path = f.name
        _io.write_fps_json(path, positions, distances, score_sets)

    p2, d2, s2, e2 = _io.read_fps_json(path)
    assert p2 == positions
    assert d2 == distances
    assert "chi2_all" in s2
    assert s2["chi2_all"]["distances"] == ["A1_A2"]
    os.unlink(path)


def test_write_read_pdb():
    """Round-trip a PDB file."""
    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                     dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        path = f.name
        _io.write_pdb(atoms, path)

    assert os.path.getsize(path) > 0
    with open(path) as f:
        content = f.read()
    assert "ATOM" in content
    assert "END" in content
    assert "MODEL" in content
    os.unlink(path)


def test_pdb_transform():
    """PDB write with a translation transform."""
    atoms = np.ones((3, 3), dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        path = f.name
        _io.write_pdb(atoms, path, transform=np.array([10.0, 0.0, 0.0]))

    with open(path) as f:
        content = f.read()
    # Atom 1 should be at ~11,1,1
    lines = content.strip().split("\n")
    atom_lines = [l for l in lines if l.startswith("ATOM") or l.startswith("HETATM")]
    assert len(atom_lines) == 3
    x = float(atom_lines[0][30:38].strip())
    assert abs(x - 11.0) < 0.01
    os.unlink(path)


# ---------------------------------------------------------------------------
# distance tests
# ---------------------------------------------------------------------------

def _make_test_av(points):
    """Helper: create a minimal AccessibleVolume."""
    from ..core.av import AccessibleVolume
    return AccessibleVolume(
        points=points,
        density=np.zeros((5, 5, 5), dtype=np.float32),
        grid_origin=np.zeros(3),
        grid_step=1.0,
        grid_shape=(5, 5, 5),
        attachment_point=np.zeros(3),
    )


def test_distance_between_mean_positions():
    """Rmp between two single-point AVs equals the point distance."""
    av1 = _make_test_av(np.array([[0.0, 0.0, 0.0, 1.0]]))
    av2 = _make_test_av(np.array([[10.0, 0.0, 0.0, 1.0]]))
    d = _dist.distance_between_mean_positions(av1, av2)
    assert abs(d - 10.0) < 1e-10


def test_average_distance_single_point():
    """RDA for single-point AVs equals the point distance."""
    av1 = _make_test_av(np.array([[0.0, 0.0, 0.0, 1.0]]))
    av2 = _make_test_av(np.array([[10.0, 0.0, 0.0, 1.0]]))
    d = _dist.average_distance(av1, av2, n_samples=1000)
    assert abs(d - 10.0) < 0.1


def test_mean_fret_distance():
    """Mean FRET distance < 20 A for R0=52 and 10 A point distance."""
    av1 = _make_test_av(np.array([[0.0, 0.0, 0.0, 1.0]]))
    av2 = _make_test_av(np.array([[10.0, 0.0, 0.0, 1.0]]))
    d = _dist.mean_fret_distance(av1, av2, forster_radius=52.0, n_samples=2000)
    assert d > 0
    assert d < 20.0  # E > 0.99 -> R_E ≈ R0*(1/E-1)^1/6 < 20


def test_fret_efficiency():
    """FRET efficiency at R0 = 0.5"""
    e = _dist.fret_efficiency(52.0, 52.0)
    assert abs(e - 0.5) < 1e-10


def test_distance_from_fret_efficiency():
    """Round-trip distance -> efficiency -> distance"""
    d_in = 45.0
    e = _dist.fret_efficiency(d_in, 52.0)
    d_out = _dist.distance_from_fret_efficiency(e, 52.0)
    assert abs(d_out - d_in) < 1e-10


def test_chi2_score():
    """chi2 = (delta/error)^2"""
    c = _dist.chi2_score(50.0, 45.0, 5.0, 5.0)
    assert abs(c - 1.0) < 1e-10
    c2 = _dist.chi2_score(40.0, 45.0, 5.0, 5.0)
    assert abs(c2 - 1.0) < 1e-10


def test_chi2_asymmetric():
    """chi2 uses asymmetric errors."""
    c = _dist.chi2_score(50.0, 45.0, 3.0, 5.0)
    # delta = +5, uses error_pos = 5
    assert abs(c - 1.0) < 1e-10
    c2 = _dist.chi2_score(40.0, 45.0, 3.0, 5.0)
    # delta = -5, uses error_neg = 3
    assert abs(c2 - (5.0 / 3.0) ** 2) < 1e-10


# ---------------------------------------------------------------------------
# clash tests
# ---------------------------------------------------------------------------

def test_no_clash():
    """Well-separated atoms should have zero clash energy."""
    xyzr1 = np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64)
    xyzr2 = np.array([[100.0, 0.0, 0.0, 1.7]], dtype=np.float64)
    e, f1, f2 = _clash.body_clash_energy(xyzr1, xyzr2, k_clash=10.0)
    assert e == 0.0
    assert np.all(f1 == 0.0)
    assert np.all(f2 == 0.0)


def test_clash_positive():
    """Overlapping atoms should produce positive energy and forces."""
    xyzr1 = np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64)
    xyzr2 = np.array([[0.5, 0.0, 0.0, 1.7]], dtype=np.float64)
    e, f1, f2 = _clash.body_clash_energy(xyzr1, xyzr2, k_clash=10.0)
    assert e > 0.0
    assert np.max(np.abs(f1)) > 0.0
    assert np.max(np.abs(f2)) > 0.0


def test_has_clash():
    """has_clash correctly identifies overlapping configurations."""
    xyzr1 = np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64)
    xyzr2 = np.array([[0.5, 0.0, 0.0, 1.7]], dtype=np.float64)
    assert _clash.has_clash(xyzr1, xyzr2) is True
    xyzr3 = np.array([[100.0, 0.0, 0.0, 1.7]], dtype=np.float64)
    assert _clash.has_clash(xyzr1, xyzr3) is False


def test_total_clash_energy():
    """total_clash_energy matches pair-wise sum."""
    xyzr = np.array([
        [0.0, 0.0, 0.0, 1.7],
        [10.0, 0.0, 0.0, 1.7],
        [0.0, 0.0, 0.0, 1.7],
    ], dtype=np.float64)
    indices = np.array([0, 0, 1], dtype=np.int64)
    e = _clash.total_clash_energy(xyzr, indices, k_clash=10.0)
    # Only atoms 0 and 2 clash (same position, different bodies)
    assert e > 0.0


# ---------------------------------------------------------------------------
# AV tests (if labellib available)
# ---------------------------------------------------------------------------

def _has_labellib():
    try:
        import LabelLib
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_labellib(), reason="LabelLib not available")
def test_av_basic():
    """Basic AV creation and mean position."""
    import LabelLib as ll
    atoms_f32 = np.array([
        [0.0, 0.0, 0.0, 1.7],
        [5.0, 0.0, 0.0, 1.7],
    ], dtype=np.float32)
    src = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    from ..core.av import _HAS_LABELLIB, _LABELLIB_BACKEND, compute_av
    if not _HAS_LABELLIB or not _LABELLIB_BACKEND:
        pytest.skip("LabelLib backend not active on this platform")
    av = compute_av(atoms_f32, src, linker_length=10.0, linker_width=1.0,
                    radii=(3.5, 0.0, 0.0), disc_step=0.5)
    assert av.n_points > 0
    mp = av.mean_position
    assert np.all(np.isfinite(mp))


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

def test_rotation_matrix():
    """Rotation matrix has det = 1."""
    from ..core.engine import _rotation_matrix
    axis = np.array([0.0, 0.0, 1.0])
    rot = _rotation_matrix(axis, math.pi / 2)
    assert abs(np.linalg.det(rot) - 1.0) < 1e-10
    # Rotating [1,0,0] by 90 deg around Z should give [0,1,0]
    v = rot @ np.array([1.0, 0.0, 0.0])
    assert abs(v[0]) < 1e-10
    assert abs(v[1] - 1.0) < 1e-10


def test_rigid_body_global_coords():
    """RigidBody global_coords reflect translation + rotation."""
    from ..core.engine import RigidBody
    body = RigidBody(
        name="test",
        atoms_local=np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
                             dtype=np.float64),
        com=np.array([10.0, 0.0, 0.0]),
        rotation=np.eye(3),
        translation=np.array([10.0, 0.0, 0.0]),
    )
    gc = body.global_coords()
    np.testing.assert_array_almost_equal(gc[0], [10.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(gc[1], [11.0, 0.0, 0.0])


def test_spring_engine_single_restraint():
    """Engine with one restraint converges towards the target."""
    from ..core.engine import SpringEngine, RigidBody, DistanceRestraint, SpringParameters

    # Two bodies, one with a single atom
    body_a = RigidBody(
        name="A",
        atoms_local=np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64),
        com=np.array([0.0, 0.0, 0.0]),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
        mass=0.1,
    )
    body_b = RigidBody(
        name="B",
        atoms_local=np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64),
        com=np.array([100.0, 0.0, 0.0]),
        rotation=np.eye(3),
        translation=np.array([100.0, 0.0, 0.0]),
        mass=0.1,
    )

    rst = DistanceRestraint(
        name="test",
        body_a=0,
        offset_a=body_a.com.copy() - body_a.com,  # zero offset
        body_b=1,
        offset_b=body_b.com.copy() - body_b.com,  # zero offset
        distance_exp=50.0,
        error_neg=10.0,
        error_pos=10.0,
    )

    params = SpringParameters(
        max_iterations=50000,
        max_force=100.0,
        time_step_factor=0.05,
        F_tolerance=1e-2,
        T_tolerance=1e-2,
    )

    engine = SpringEngine([body_a, body_b], [rst], params)
    converged = engine.simulate()
    assert converged, "Engine did not converge"

    final_dist = np.linalg.norm(body_a.com - body_b.com)
    assert abs(final_dist - 50.0) < 10.0, (
        f"Final distance {final_dist:.2f} not close to target 50.0"
    )


def test_engine_convergence_norms():
    """Force and torque norms decrease during simulation."""
    from ..core.engine import SpringEngine, RigidBody, DistanceRestraint, SpringParameters

    body_a = RigidBody(
        name="A",
        atoms_local=np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64),
        com=np.array([0.0, 0.0, 0.0]),
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
    )
    body_b = RigidBody(
        name="B",
        atoms_local=np.array([[0.0, 0.0, 0.0, 1.7]], dtype=np.float64),
        com=np.array([100.0, 0.0, 0.0]),
        rotation=np.eye(3),
        translation=np.array([100.0, 0.0, 0.0]),
    )

    rst = DistanceRestraint(
        name="test",
        body_a=0, offset_a=np.zeros(3),
        body_b=1, offset_b=np.zeros(3),
        distance_exp=50.0, error_neg=10.0, error_pos=10.0,
    )

    engine = SpringEngine([body_a, body_b], [rst], SpringParameters(max_iterations=500))
    engine.simulate()
    assert len(engine._force_norms) > 0
    # Force norms should trend downward
    first_third = len(engine._force_norms) // 3
    if first_third > 0 and len(engine._force_norms) > first_third * 2:
        initial = np.mean(engine._force_norms[:first_third])
        final = np.mean(engine._force_norms[-first_third:])
        assert final <= initial * 1.1 + 1e-6  # non-increasing


# ---------------------------------------------------------------------------
# Integration test (end-to-end with LabelLib)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_labellib(), reason="LabelLib not available"
)
def test_docking_integration():
    """End-to-end docking with a minimal 2-body system."""
    from ..core.av import _HAS_LABELLIB, _LABELLIB_BACKEND
    if not _HAS_LABELLIB or not _LABELLIB_BACKEND:
        pytest.skip("LabelLib backend not active on this platform")
    from ..core.av import load_structure_with_vdw
    from ..core.docking import run_docking

    # Create a minimal PDB with 2 atoms
    pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2      10.000   0.000   0.000  1.00  0.00           C
END
"""
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        pdb_path = f.name
        f.write(pdb_content)

    positions = {
        "P1": {
            "atom_name": "CA",
            "chain_identifier": "A",
            "residue_seq_number": 1,
            "linker_length": 5.0,
            "linker_width": 1.0,
            "radius1": 2.0,
            "simulation_grid_resolution": 1.5,
            "simulation_type": "AV1",
            "body_id": 0,
        },
        "P2": {
            "atom_name": "CA",
            "chain_identifier": "A",
            "residue_seq_number": 2,
            "linker_length": 5.0,
            "linker_width": 1.0,
            "radius1": 2.0,
            "simulation_grid_resolution": 1.5,
            "simulation_type": "AV1",
            "body_id": 1,
        },
    }
    distances = {
        "P1_P2": {
            "position1_name": "P1",
            "position2_name": "P2",
            "distance": 8.0,
            "error_neg": 3.0,
            "error_pos": 3.0,
            "distance_type": "RDAMean",
            "Forster_radius": 52.0,
        },
    }

    from ..core.engine import SpringParameters
    params = SpringParameters(max_iterations=500, max_force=50.0)

    try:
        dock_results, av_dict, body_list = run_docking(
            pdb_path, positions, distances, params=params, n_trials=2
        )
        assert len(dock_results) == 2
        for sr in dock_results:
            assert sr.iterations > 0
    finally:
        os.unlink(pdb_path)


def test_transfer_functions():
    """Test the polynomial and Gaussian transfer functions and their evaluations."""
    # Test polynomial evaluation: y = 2x^2 + 3x + 4
    # coeffs = [2, 3, 4]
    coeffs = np.array([2.0, 3.0, 4.0])
    res = _dist.polynomial_transfer(2.0, coeffs)
    assert abs(res - 18.0) < 1e-10

    # Test Gaussian correction: RDAMean ≈ Rmp + (sigma^2 / (2 * Rmp))
    # rmp = 10, sigma = 2 -> 10 + 4 / 20 = 10.2
    res_g = _dist.gaussian_rmp_to_rda_mean(10.0, 2.0)
    assert abs(res_g - 10.2) < 1e-10


def test_av_pair_statistics_and_polyfit():
    """Test computing AV pair statistics and fitting transfer polynomials."""
    # Create two single-point AVs
    av1 = _make_test_av(np.array([[0.0, 0.0, 0.0, 1.0]]))
    av2 = _make_test_av(np.array([[10.0, 0.0, 0.0, 1.0]]))

    # Rmp, RDAMean, RDAMeanE, sigma should be calculated
    rmp, rda, rda_e, sigma = _dist.av_pair_statistics(av1, av2, forster_radius=52.0, n_samples=100)
    assert abs(rmp - 10.0) < 1e-10
    assert abs(rda - 10.0) < 0.5
    assert abs(rda_e - 10.0) < 0.5
    assert abs(sigma - 0.0) < 0.1

    # Fit polynomial should return identity-like for single point clouds
    coeffs = _dist.fit_transfer_polynomial(av1, av2, distance_type="RDAMean", degree=2, n_samples=100)
    # y ≈ x -> coeffs should evaluate to ~10 for Rmp = 10
    val = _dist.polynomial_transfer(10.0, coeffs)
    assert abs(val - 10.0) < 0.5
