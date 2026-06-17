from __future__ import annotations

import json

from chisurf.core.project import Project, save_project, load_project


def test_project_json_roundtrip(tmp_path):
    project_dir = tmp_path / "test_project"

    # Create a minimal project with some representative content
    p = Project(
        name="unit_test_project",
        description="Project JSON round-trip test",
        chisurf_version="test-version",
        project_format_version=4,
    )
    p.datasets["ds1"] = {"path": "data/file1.dat", "checksum": "abc123"}
    p.experiments["exp1"] = {"type": "tcspc", "dataset_id": "ds1"}
    p.fits.append({"uid": "fit1", "experiment_id": "exp1", "chi2": 1.234})
    p.ui_state["current_experiment_id"] = "exp1"

    archive_path = save_project(p, project_dir)

    # Ensure the archive was created where we expect it
    assert archive_path.is_file()
    assert archive_path.suffix == ".csp"

    # Sanity-check the raw JSON structure
    import zipfile
    with zipfile.ZipFile(archive_path, "r") as zf:
        raw = json.loads(zf.read("project.json"))

    assert raw["meta"]["name"] == "unit_test_project"
    assert raw["project_format_version"] == 4
    assert "datasets" in raw and "ds1" in raw["datasets"]

    # Load back into a Project instance and compare key fields
    loaded = load_project(project_dir)

    assert isinstance(loaded, Project)
    assert loaded.name == p.name
    assert loaded.description == p.description
    assert loaded.chisurf_version == p.chisurf_version
    assert loaded.project_format_version == p.project_format_version
    assert loaded.datasets == p.datasets
    assert loaded.experiments == p.experiments
    assert loaded.fits == p.fits
    assert loaded.ui_state == p.ui_state
