"""Headless tests for the IMP + IMP.bff FRET docking engine.

These exercise the real engine when IMP/IMP.bff is available and are skipped
otherwise, so the suite stays green in environments without the bff backend.
"""

from __future__ import annotations

import os

import pytest

from chisurf.plugins.modelling.fret.core import imp_engine

pytestmark = pytest.mark.skipif(
    not imp_engine.has_imp(),
    reason="IMP with the bff module (IMP.bff.AV) is not installed",
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.normpath(os.path.join(_HERE, "..", "examples", "fps_hiv_rt"))
_PROTEIN = os.path.join(_EX, "protein_1R0A.pdb")
_DNA = os.path.join(_EX, "dna.pdb")
_FPS = os.path.join(_EX, "hiv_rt.fps.json")


def _have_example() -> bool:
    return all(os.path.exists(p) for p in (_PROTEIN, _DNA, _FPS))


needs_example = pytest.mark.skipif(not _have_example(), reason="hiv_rt example data missing")


@needs_example
def test_score_builds_avs_and_distances():
    res = imp_engine.score([_PROTEIN, _DNA], _FPS, mean_position_restraint=True)
    assert res.n_avs > 0
    assert res.n_distances > 0
    assert res.score == res.score  # not NaN


@needs_example
def test_dock_runs_and_writes_outputs(tmp_path):
    params = imp_engine.DockingParameters(
        n_frames=4, mc_steps=8, shuffle_max_translation=0.0, n_best=2,
        fixed_body=0, max_translation=3.0,
    )
    res = imp_engine.dock([_PROTEIN, _DNA], _FPS, str(tmp_path), params)
    # the MC run completes, produces a finite score and emits outputs
    assert res.score == res.score  # not NaN
    assert res.n_distances > 0
    assert res.rmf_file is not None and os.path.exists(res.rmf_file)
    assert os.path.exists(res.score_csv)
    assert res.best_pdbs  # PMI wrote best-scoring PDB models


@needs_example
def test_dock_minimize_converges(tmp_path):
    """Conjugate-gradient docking lowers the score and writes a docked PDB.

    Exercises the harmonic point-distance springs on AV-mean point-member
    proxies that let an IMP minimiser drive the rigid bodies.
    """
    import csv

    params = imp_engine.DockingParameters(n_frames=200, shuffle_max_translation=0.0)
    res = imp_engine.dock_minimize([_PROTEIN, _DNA], _FPS, str(tmp_path), params)
    assert res.extra["method"] == "minimize"
    assert res.n_distances > 0
    assert res.best_pdbs and os.path.exists(res.best_pdbs[0])
    conv = res.extra["convergence_csv"]
    rows = list(csv.reader(open(conv)))[1:]
    series = [float(r[1]) for r in rows]
    # minimisation must reduce the energy from its starting value
    assert series[-1] < series[0]


@needs_example
def test_estimate_errors_parallel(tmp_path):
    """Parallel trials produce the same number of valid results as serial."""
    params = imp_engine.DockingParameters(n_frames=150, coarse_clash=True)
    serial = imp_engine.estimate_errors(
        [_PROTEIN, _DNA], _FPS, str(tmp_path / "s"), n_trials=2, params=params, n_workers=1)
    parallel = imp_engine.estimate_errors(
        [_PROTEIN, _DNA], _FPS, str(tmp_path / "p"), n_trials=2, params=params, n_workers=2)
    assert serial["n_workers"] == 1
    assert parallel["n_workers"] == 2  # fork pool actually ran
    assert len(parallel["trial_details"]) == 2
    # both schedules dock to a finite score and identify a best trial
    for res in (serial, parallel):
        assert all(s == s for s in res["scores"])  # no NaNs
        assert res["best_trial"] in (0, 1)


@needs_example
def test_estimate_errors_reports_uncertainty(tmp_path):
    """Repeated docking superposes models and reports per-atom RMSF precision."""
    params = imp_engine.DockingParameters(n_frames=120, coarse_clash=True)
    res = imp_engine.estimate_errors(
        [_PROTEIN, _DNA], _FPS, str(tmp_path), n_trials=2, params=params, n_workers=1)
    unc = res["uncertainty"]
    assert unc is not None and unc["n_models"] == 2
    assert unc["mobile_rmsf_mean"] == unc["mobile_rmsf_mean"]  # finite
    assert os.path.exists(unc["uncertainty_pdb"]) and os.path.exists(unc["uncertainty_csv"])


@needs_example
def test_dock_minimize_saves_distributions(tmp_path):
    """save_distributions exports a P(R_DA) table over the docked structure."""
    import csv
    params = imp_engine.DockingParameters(
        n_frames=100, shuffle_max_translation=0.0, save_distributions=True)
    res = imp_engine.dock_minimize([_PROTEIN, _DNA], _FPS, str(tmp_path), params)
    csv_path = res.extra.get("distributions_csv")
    assert csv_path and os.path.exists(csv_path)
    rows = list(csv.reader(open(csv_path)))
    assert rows[0][0] == "R_DA" and len(rows[0]) > 1  # R_DA + per-pair columns
    assert len(rows) > 10  # several R_DA bins


@needs_example
def test_refine_runs(tmp_path):
    res = imp_engine.refine([_PROTEIN, _DNA], _FPS, str(tmp_path), steps=20)
    assert res.best_pdbs and os.path.exists(res.best_pdbs[0])


def test_rpc_services_register():
    from chisurf.plugins.modelling.fret.backend import services

    class _Disp:
        def __init__(self):
            self.methods = {}

        def register(self, name, fn):
            self.methods[name] = fn

    d = _Disp()
    services.register_services(d)
    assert "fret.dock" in d.methods
    assert "fret.score" in d.methods
    info = d.methods["fret.info_backends"]({})
    assert info["status"] == "ok"
    assert info["data"]["has_imp_bff"] is True
