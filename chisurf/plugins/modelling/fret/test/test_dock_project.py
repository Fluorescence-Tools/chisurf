"""Tests for docking project files (load/save) and running them.

The pure load/save round-trip runs everywhere; the end-to-end run is skipped
when IMP/IMP.bff is unavailable.
"""

from __future__ import annotations

import os

import pytest

from chisurf.plugins.modelling.fret.api import project as _project
from chisurf.plugins.modelling.fret.core import imp_engine

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.normpath(os.path.join(_HERE, "..", "examples", "fps_hiv_rt"))
_PROJECT = os.path.join(_EX, "docking_project.json")


def test_example_project_loads_with_absolute_paths():
    proj = _project.load_docking_project(_PROJECT)
    assert proj.operation == "dock"
    assert len(proj.pdb_paths) == 2
    for p in proj.pdb_paths:
        assert os.path.isabs(p) and os.path.exists(p)
    assert os.path.isabs(proj.fps_json) and os.path.exists(proj.fps_json)
    assert os.path.isabs(proj.output_dir)
    req = proj.to_dock_request()
    assert req["pdb_paths"] == proj.pdb_paths
    assert req["fps_json"] == proj.fps_json
    assert req["n_frames"] == 500


def test_save_round_trip_keeps_paths_relative(tmp_path):
    pdb = tmp_path / "a.pdb"
    fps = tmp_path / "x.fps.json"
    out = tmp_path / "out"
    for p in (pdb, fps):
        p.write_text("")
    dst = tmp_path / "proj.json"
    _project.save_docking_project(
        str(dst), pdb_paths=[str(pdb)], fps_json=str(fps), output_dir=str(out),
        operation="dock", params={"n_frames": 7},
    )
    text = dst.read_text()
    # paths under the project dir are stored relative (portable)
    assert '"a.pdb"' in text and '"x.fps.json"' in text and '"out"' in text
    loaded = _project.load_docking_project(str(dst))
    assert loaded.pdb_paths == [str(pdb)]
    assert loaded.fps_json == str(fps)
    assert loaded.params["n_frames"] == 7


def test_ensure_fps_json_accepts_csharp_txt():
    """A legacy C# LPs .txt is converted to a usable fps.json (FPS-native input)."""
    lps = os.path.join(_EX, "LPs_no_template_old_protein.txt")
    if not os.path.exists(lps):
        pytest.skip("C# LPs example missing")
    pdbs = [os.path.join(_EX, "protein_1R0A.pdb"), os.path.join(_EX, "dna.pdb")]
    out = imp_engine.ensure_fps_json(lps, pdbs)
    assert out.endswith(".json") and os.path.exists(out)
    import json
    payload = json.load(open(out))
    assert len(payload["Positions"]) == 11
    assert len(payload["Distances"]) == 20
    # a native fps.json passes straight through
    assert imp_engine.ensure_fps_json(out, pdbs) == out


@pytest.mark.skipif(not imp_engine.has_imp(), reason="IMP/IMP.bff not installed")
def test_dock_project_runs(tmp_path):
    from chisurf.plugins.modelling.fret.api import operations as ops

    res = ops.dock_project(
        _PROJECT,
        {"output_dir": str(tmp_path), "n_frames": 4, "mc_steps": 4, "n_best": 2},
    )
    assert res["status"] == "ok"
    data = res["data"]
    assert data["n_distances"] > 0
    assert data["score"] == data["score"]  # not NaN
    assert os.path.exists(data["score_csv"])
