"""Unit tests for Phase 1: bug fixes and backend infrastructure."""

import numpy as np
import pytest

from ..core.av import _HAS_IMP_BFF, _HAS_LABELLIB, select_backend
from ..core.engine import DistanceRestraint, RigidBody
from ..core.sampling import run_metropolis


def _make_toy_system():
    """Create a minimal 2-body system for sampling tests."""
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
        com=np.array([10.0, 0.0, 0.0]),
        rotation=np.eye(3),
        translation=np.array([10.0, 0.0, 0.0]),
    )
    rst = DistanceRestraint(
        name="test_rst",
        body_a=0,
        offset_a=np.zeros(3),
        body_b=1,
        offset_b=np.zeros(3),
        distance_exp=8.0,
        error_neg=1.0,
        error_pos=1.0,
    )
    atoms_xyzr = np.array([[0.0, 0.0, 0.0, 1.7], [10.0, 0.0, 0.0, 1.7]], dtype=np.float64)
    return [body_a, body_b], [rst], atoms_xyzr


def test_metropolis_5_steps_no_error():
    """Verify that run_metropolis executes 5 steps without raising AttributeError or other errors."""
    bodies, restraints, atoms_xyzr = _make_toy_system()
    results = run_metropolis(
        bodies=bodies,
        restraints=restraints,
        positions={},
        avs={},
        atoms_xyzr=atoms_xyzr,
        n_samples=5,
        n_burnin=2,
    )
    assert len(results) == 5
    for r in results:
        assert r.converged
        assert len(r.translations) == 2
        assert len(r.rotations) == 2


def test_metropolis_energy_finite():
    """Verify that energy after 5 steps of Metropolis Monte Carlo is a finite float."""
    bodies, restraints, atoms_xyzr = _make_toy_system()
    results = run_metropolis(
        bodies=bodies,
        restraints=restraints,
        positions={},
        avs={},
        atoms_xyzr=atoms_xyzr,
        n_samples=5,
        n_burnin=2,
    )
    for r in results:
        assert isinstance(r.energy, float)
        assert np.isfinite(r.energy)


def test_select_backend_labellib():
    """Verify select_backend('labellib') works correctly if LabelLib is installed."""
    if not _HAS_LABELLIB:
        pytest.skip("LabelLib is not available on this system")

    select_backend("labellib")
    from ..core import av
    assert av._LABELLIB_BACKEND is True


def test_select_backend_invalid_raises():
    """Verify that select_backend with an invalid name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend name"):
        select_backend("invalid_backend_name")


def test_select_backend_auto_prefers_imp_bff_off_windows(monkeypatch):
    """Verify auto backend selection prefers IMP.bff on non-Windows systems."""
    from ..core import av

    monkeypatch.setattr(av, "_LABELLIB_BACKEND", av._LABELLIB_BACKEND)
    monkeypatch.setattr(av, "_HAS_LABELLIB", True)
    monkeypatch.setattr(av, "_HAS_IMP_BFF", True)
    monkeypatch.setattr(av.sys, "platform", "darwin")

    av.select_backend("auto")

    assert av._LABELLIB_BACKEND is False


def test_select_backend_auto_prefers_labellib_on_windows(monkeypatch):
    """Verify auto backend selection keeps LabelLib first on Windows."""
    from ..core import av

    monkeypatch.setattr(av, "_LABELLIB_BACKEND", av._LABELLIB_BACKEND)
    monkeypatch.setattr(av, "_HAS_LABELLIB", True)
    monkeypatch.setattr(av, "_HAS_IMP_BFF", True)
    monkeypatch.setattr(av.sys, "platform", "win32")

    av.select_backend("auto")

    assert av._LABELLIB_BACKEND is True


def test_select_backend_auto_uses_labellib_when_imp_bff_api_missing(monkeypatch):
    """Verify auto selection treats incomplete IMP.bff as unavailable."""
    from ..core import av

    monkeypatch.setattr(av, "_LABELLIB_BACKEND", av._LABELLIB_BACKEND)
    monkeypatch.setattr(av, "_HAS_LABELLIB", True)
    monkeypatch.setattr(av, "_HAS_IMP_BFF", False)
    monkeypatch.setattr(av.sys, "platform", "darwin")

    av.select_backend("auto")

    assert av._LABELLIB_BACKEND is True


def test_compute_av_falls_back_to_labellib_when_imp_bff_fails(monkeypatch):
    """Verify a broken IMP.bff runtime path falls back to LabelLib."""
    from ..core import av

    calls = []

    def fake_imp_bff(*args, **kwargs):
        raise AttributeError("module 'IMP.bff' has no attribute 'AV'")

    def fake_labellib(atoms, source_xyz, linker_length, linker_width, radii, disc_step):
        calls.append((atoms, source_xyz, linker_length, linker_width, radii, disc_step))
        return av.AccessibleVolume(
            points=np.array([[1.0, 2.0, 3.0, 1.0]], dtype=float),
            density=np.ones((1, 1, 1), dtype=np.float32),
            grid_origin=np.zeros(3),
            grid_step=float(disc_step),
            grid_shape=(1, 1, 1),
            attachment_point=np.asarray(source_xyz, dtype=float),
        )

    monkeypatch.setattr(av, "_LABELLIB_BACKEND", False)
    monkeypatch.setattr(av, "_HAS_IMP_BFF", True)
    monkeypatch.setattr(av, "_HAS_LABELLIB", True)
    monkeypatch.setattr(av, "_av_imp_bff", fake_imp_bff)
    monkeypatch.setattr(av, "_av_labellib", fake_labellib)

    result = av.compute_av(
        atoms=np.array([[0.0, 0.0, 0.0, 1.7]], dtype=float),
        source_xyz=np.array([1.0, 2.0, 3.0], dtype=float),
        linker_length=20.0,
        linker_width=4.5,
        radii=(5.0, 0.0, 0.0),
        disc_step=0.5,
        pdb_path="dummy.pdb",
        source_info={"chain_identifier": "A", "residue_seq_number": 1, "atom_name": "CA"},
    )

    assert result.n_points == 1
    assert calls


def test_info_backends_at_least_one_available():
    """Verify that at least one of the AV backends is available on the system."""
    assert _HAS_LABELLIB or _HAS_IMP_BFF, "Neither LabelLib nor IMP.bff is available"
