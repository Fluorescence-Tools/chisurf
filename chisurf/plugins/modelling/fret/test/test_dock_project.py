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


_POSES = [
    {"body_id": 0, "t": [0.0, 0.0, 0.0], "q": [1.0, 0.0, 0.0, 0.0]},
    {"body_id": 1, "t": [12.5, -3.25, 7.0], "q": [0.5, 0.5, 0.5, 0.5]},
]


def test_save_load_poses_round_trip(tmp_path):
    """A project with docked poses round-trips and yields initial_poses."""
    pdb = tmp_path / "a.pdb"
    fps = tmp_path / "x.fps.json"
    for p in (pdb, fps):
        p.write_text("")
    dst = tmp_path / "proj.json"
    _project.save_docking_project(
        str(dst), pdb_paths=[str(pdb)], fps_json=str(fps), output_dir=str(tmp_path),
        operation="dock", method="minimize", params={"n_frames": 4},
        poses=_POSES, pose_score=42.5, pose_method="minimize",
    )
    # poses are stored as one compact self-describing blob, not raw coordinates
    import json as _json
    payload = _json.loads(dst.read_text())
    assert "codec" in payload["poses"] and payload["poses"]["score"] == 42.5

    loaded = _project.load_docking_project(str(dst))
    assert [b["body_id"] for b in loaded.poses] == [0, 1]
    assert loaded.poses[1]["t"] == pytest.approx(_POSES[1]["t"])
    assert loaded.pose_meta["score"] == 42.5
    req = loaded.to_dock_request()
    assert "initial_poses" in req and len(req["initial_poses"]) == 2


def test_v1_project_loads_without_poses():
    """A legacy project (no poses key) loads with an empty pose list."""
    proj = _project.load_docking_project(_PROJECT)
    assert proj.poses == []
    assert "initial_poses" not in proj.to_dock_request()


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
    # every rigid body reports a compact pose (transform vectors)
    assert len(data["poses"]) == len(_project.load_docking_project(_PROJECT).pdb_paths)


def _pdbs_and_fps():
    proj = _project.load_docking_project(_PROJECT)
    return proj.pdb_paths, proj.fps_json


@pytest.mark.skipif(not imp_engine.has_imp(), reason="IMP/IMP.bff not installed")
def test_capture_reapply_reconstructs_pose(tmp_path):
    """Save, reload and apply reproduce the docked reference frames exactly."""
    pdbs, fps = _pdbs_and_fps()
    params = imp_engine.DockingParameters(n_frames=8, shuffle_max_translation=0.0)
    res = imp_engine.dock_minimize(pdbs, fps, str(tmp_path / "out"), params)
    poses = res.poses
    assert poses

    dst = tmp_path / "docked.json"
    _project.save_docking_project(
        str(dst), pdb_paths=pdbs, fps_json=fps, output_dir=str(tmp_path / "out"),
        method="minimize", poses=poses, pose_score=res.score)

    proj = _project.load_docking_project(str(dst))
    asm = imp_engine.build_assembly(proj.pdb_paths, proj.fps_json,
                                    mean_position_restraint=False)
    imp_engine.apply_poses(asm, proj.poses)
    reapplied = {p["body_id"]: p for p in imp_engine.capture_poses(asm)}
    for saved in poses:
        got = reapplied[saved["body_id"]]
        dt = sum((a - b) ** 2 for a, b in zip(got["t"], saved["t"])) ** 0.5
        assert dt < 1e-4
        dot = abs(sum(a * b for a, b in zip(got["q"], saved["q"])))
        assert abs(dot - 1.0) < 1e-4  # same rotation (quaternion sign-agnostic)


@pytest.mark.skipif(not imp_engine.has_imp(), reason="IMP/IMP.bff not installed")
def test_continue_from_poses_does_not_restart(tmp_path):
    """Resuming from saved poses continues (score no worse than the docked state)."""
    from chisurf.plugins.modelling.fret.api import operations as ops

    pdbs, fps = _pdbs_and_fps()
    params = imp_engine.DockingParameters(n_frames=20, shuffle_max_translation=0.0)
    first = imp_engine.dock_minimize(pdbs, fps, str(tmp_path / "a"), params)

    dst = tmp_path / "docked.json"
    _project.save_docking_project(
        str(dst), pdb_paths=pdbs, fps_json=fps, output_dir=str(tmp_path / "b"),
        method="minimize", poses=first.poses, pose_score=first.score)

    res = ops.dock_project(str(dst), {"output_dir": str(tmp_path / "b"), "n_frames": 20})
    assert res["status"] == "ok"
    # Continuing further minimisation never worsens the score of the docked state.
    assert res["data"]["score"] <= first.score + 1e-6
