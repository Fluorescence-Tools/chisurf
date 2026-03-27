from __future__ import annotations

import json

from chisurf.project import Project, save_project, load_project


def test_project_json_roundtrip(tmp_path):
    project_dir = tmp_path / "test_project"

    # Create a minimal project with some representative content
    p = Project(
        name="unit_test_project",
        description="Project JSON round-trip test",
        chisurf_version="test-version",
        project_format_version=3,
    )
    p.datasets["ds1"] = {"path": "data/file1.dat", "checksum": "abc123"}
    p.experiments["exp1"] = {"type": "tcspc", "dataset_id": "ds1"}
    p.fits.append({"uid": "fit1", "experiment_id": "exp1", "chi2": 1.234})
    p.ui_state["current_experiment_id"] = "exp1"

    project_json_path = save_project(p, project_dir)

    # Ensure the JSON file was created where we expect it
    assert project_json_path.is_file()
    assert project_json_path.name == "project.json"

    # Sanity-check the raw JSON structure
    with project_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    assert raw["meta"]["name"] == "unit_test_project"
    assert raw["project_format_version"] == 3
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
