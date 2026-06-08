"""Integration tests using reference example files from Olga/FPS.
"""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np

from .. import av as _av
from .. import io as _io
from .. import evaluate as _evaluate

OLGA_T4L_DIR = "/Users/tpeulen/dev/olga/doc/data/T4L"
SCREENING_JSON = os.path.join(OLGA_T4L_DIR, "screening_tutorial.fps.json")
PAIR_SEL_JSON = os.path.join(OLGA_T4L_DIR, "pair_selection_tutorial.fps.json")
PDB_FILE = os.path.join(OLGA_T4L_DIR, "3GUN_NMSim_cl-rep-001.pdb")


def test_load_olga_example_screening_json():
    """Verify we can load the real screening_tutorial.fps.json file."""
    assert os.path.exists(SCREENING_JSON)
    positions, distances, score_sets, extra = _io.read_fps_json(SCREENING_JSON)
    assert len(positions) == 3
    assert "D36" in positions
    assert "A132" in positions
    assert "D60" in positions
    assert len(distances) == 2
    assert "36_132" in distances


def test_compute_avs_on_olga_pdb():
    """Verify we can load the T4L PDB and compute AVs on it using the loaded positions."""
    assert os.path.exists(PDB_FILE)
    positions, _, _, _ = _io.read_fps_json(SCREENING_JSON)
    
    # Load PDB with VdW radii
    atoms_xyzr = _av.load_structure_with_vdw(PDB_FILE)
    assert atoms_xyzr.shape[0] > 0
    
    # Compute AVs for the three positions
    avs = _av.compute_avs_for_structure(atoms_xyzr, positions, pdb_path=PDB_FILE)
    assert len(avs) == 3
    for name, av in avs.items():
        assert av.has_volume
        assert av.n_points > 0
        assert np.all(np.isfinite(av.mean_position))


def test_evaluate_olga_structure():
    """Verify we can evaluate the T4L PDB structure using evaluators read from the JSON."""
    positions, _, _, _ = _io.read_fps_json(SCREENING_JSON)
    evaluators = _io.read_evaluators_json(SCREENING_JSON)
    
    if not evaluators:
        # Construct them manually from distances
        _, distances, _, _ = _io.read_fps_json(SCREENING_JSON)
        from ..evaluators import DistanceEvaluator
        evaluators = [
            DistanceEvaluator(name, d["position1_name"], d["position2_name"], distance_type=d["distance_type"])
            for name, d in distances.items()
        ]
        
    res = _evaluate.evaluate_structure(PDB_FILE, positions, evaluators)
    assert len(res) > 0
    for name, r in res.items():
        assert np.isfinite(r.value)


def test_4w_junction_docking():
    """Verify the 4w junction multi-body docking example works end-to-end."""
    test_dir = os.path.join(
        os.path.dirname(__file__),
        "../../../../../examples/4w_junction/fps_test_data/test_docking/"
    )
    lps_path = os.path.join(test_dir, "LPs_no_template_old_protein.txt")
    protein_pdb = os.path.join(test_dir, "protein_1R0A.pdb")
    dna_pdb = os.path.join(test_dir, "dna.pdb")

    assert os.path.exists(lps_path)
    assert os.path.exists(protein_pdb)
    assert os.path.exists(dna_pdb)

    # 1. Parse configuration
    pdb_paths = [protein_pdb, dna_pdb]
    positions, distances, _, _ = _io.read_fps_json(lps_path, pdb_paths=pdb_paths)

    assert len(positions) > 0
    assert len(distances) > 0
    # Verify body IDs are assigned
    body_ids = {p["body_id"] for p in positions.values()}
    assert body_ids == {0, 1}

    # 2. Run docking
    from ..core import docking as _docking
    from ..core.engine import SpringParameters
    
    params = SpringParameters(max_iterations=5)
    results, avs, bodies = _docking.run_docking(
        pdb_paths,
        positions,
        distances,
        params=params,
        n_trials=1,
    )

    assert len(results) == 1
    assert not results[0].converged
    assert len(bodies) == 2
    assert len(avs) > 0

    # 3. Write output
    from ..core import results as _results
    with tempfile.TemporaryDirectory() as tmpdir:
        atoms_per_body = [b.atoms_local for b in bodies]
        written = _results.write_docking_results_pdb(results, atoms_per_body, tmpdir)
        assert len(written) == 2
        for p in written:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 0


def test_project_save_load():
    """Verify that project save and load functions work correctly via Project schema."""
    from chisurf.core.project import Project
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock project directory
        proj_dir = os.path.join(tmpdir, "test_fret_proj")
        os.makedirs(proj_dir)
        
        # We can simulate the FretDockWizard save/load state
        ui_fret = {
            "fps_json": {"Positions": {}, "Distances": {}},
            "reference_pdb": "protein.pdb",
            "docking": {
                "fps_path": "project.fps.json",
                "pdb_path": "protein.pdb",
                "out_path": "dock_out",
                "max_iter": 1000,
                "max_force": 50.0,
                "n_trials": 2,
                "k_clash": 5.0,
            }
        }
        
        project = Project(
            name="test_proj",
            description="Mock project",
            ui_state={"fret_dock_wizard": ui_fret}
        )
        
        # Save
        project.save(proj_dir)
        assert os.path.exists(os.path.join(proj_dir, "project.json"))
        
        # Load
        loaded_project = Project.load(proj_dir)
        loaded_ui = loaded_project.ui_state.get("fret_dock_wizard", {})
        assert loaded_ui["reference_pdb"] == "protein.pdb"
        assert loaded_ui["docking"]["pdb_path"] == "protein.pdb"
        assert loaded_ui["docking"]["max_iter"] == 1000



def test_click_cli_backends():
    """Verify the click CLI info-backends command runs successfully."""
    from click.testing import CliRunner
    from ..cli import main
    
    runner = CliRunner()
    result = runner.invoke(main, ["info-backends"])
    assert result.exit_code == 0
    assert "AV backends" in result.output
    assert "Active backend" in result.output


def test_click_cli_info():
    """Verify the click CLI info command works on reference fps.json."""
    from click.testing import CliRunner
    from ..cli import main
    
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--fps", SCREENING_JSON])
    assert result.exit_code == 0
    assert "Positions:" in result.output
    assert "Distances:" in result.output
    assert "36_132" in result.output


def test_fastapi_endpoints():
    """Verify the FastAPI routing endpoints return valid responses."""
    from fastapi.testclient import TestClient
    from ..api import app
    
    client = TestClient(app)
    
    # 1. Test /info-backends
    resp = client.get("/fret/info-backends")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_backend" in data
    assert "has_labellib" in data
    
    # 2. Test /info
    resp = client.post("/fret/info", json={"fps_path": SCREENING_JSON})
    assert resp.status_code == 200
    data = resp.json()
    assert data["positions_count"] == 3
    assert data["distances_count"] == 2


