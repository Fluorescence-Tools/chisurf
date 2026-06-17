from __future__ import annotations

# Consolidated test file: test_project_roundtrip.py


# --- FROM test_project_fits_roundtrip.py ---

import json
import numpy as np
import pytest

pytest.importorskip("chinet")

from chisurf.core.data import DataCurve
from chisurf.core.fitting.fit import Fit
from chisurf.core.models.model import ModelCurve

from chisurf.core.project import Project, save_project, load_project
from chisurf.core.project.fit_state import make_fit_record, apply_fit_record
from chisurf.core.fitting.parameter import FittingParameter


class DummyLinearModel(ModelCurve):
    """Small concrete model used for Project.fits round-trip tests.

    Uses two fitting parameters ``p0`` and ``p1`` to describe
    ``y = p0 + p1 * x``. This mirrors the setup in ``test_fit_state``
    without depending on any GUI components.
    """

    name = "DummyLinearModelForProject"

    def __init__(self, fit: Fit, **kwargs):  # type: ignore[override]
        super().__init__(fit, **kwargs)
        self.p0 = FittingParameter(name="p0", value=0.5)
        self.p1 = FittingParameter(name="p1", value=1.5)
        self.find_parameters()

    def update_model(self, **kwargs):  # type: ignore[override]
        x = self.fit.data.x
        if x is None:
            x = np.arange(self.fit.data.y.size, dtype=float)
        self.x = x
        self.y = float(self.p0.value) + float(self.p1.value) * x

    def update(self, **kwargs) -> None:  # type: ignore[override]
        super().update(**kwargs)


def _make_dummy_fit() -> Fit:
    x = np.arange(4, dtype=float)
    y = np.ones_like(x)
    data = DataCurve(x=x, y=y)
    return Fit(model_class=DummyLinearModel, data=data)


def test_project_fits_roundtrip_with_single_fit(tmp_path):
    # Prepare a simple project with one dataset, one experiment and one fit
    project_dir = tmp_path / "proj1"

    p = Project(
        name="proj_with_fit",
        description="Project.fits JSON round-trip test",
        chisurf_version="test-version",
    )

    # Minimal dataset / experiment references
    p.datasets["ds1"] = {"path": "data/file1.dat"}
    p.experiments["exp1"] = {"type": "dummy-experiment"}

    fit = _make_dummy_fit()
    # Customize parameter values to check that they are restored later
    params = fit.model.parameters_all_dict
    params["p0"].value = 2.0
    params["p0"].bounds = (0.0, 5.0)
    params["p0"].bounds_on = True

    params["p1"].value = -0.5
    params["p1"].fixed = True

    fit_record = make_fit_record(
        fit_id="fit-uid-1",
        fit=fit,
        dataset_id="ds1",
        experiment_id="exp1",
    )
    fit_record["uid"] = "fit-uid-1"
    p.fits.append(fit_record)

    archive_path = save_project(p, project_dir)
    assert archive_path.is_file()

    # Inspect raw JSON to ensure fits structure is present
    import zipfile
    with zipfile.ZipFile(archive_path, "r") as zf:
        raw = json.loads(zf.read("project.json"))

    assert "fits" in raw
    assert len(raw["fits"]) == 1
    raw_fit = raw["fits"][0]
    assert raw_fit["dataset_id"] == "ds1"
    assert raw_fit["experiment_id"] == "exp1"
    assert "fit_state" in raw_fit

    # Reload project and reconstruct a new Fit from the stored record
    loaded_project = load_project(project_dir)
    assert isinstance(loaded_project, Project)
    assert len(loaded_project.fits) == 1

    loaded_record = loaded_project.fits[0]

    # Make a fresh fit with default parameters and apply the stored record
    fit2 = _make_dummy_fit()
    params2 = fit2.model.parameters_all_dict

    # Ensure defaults differ from the customized values
    assert not np.isclose(params2["p0"].value, 2.0)
    assert not np.isclose(params2["p1"].value, -0.5)

    apply_fit_record(fit2, loaded_record)

    # After applying the record, the parameter state should match
    assert np.isclose(params2["p0"].value, 2.0)
    assert params2["p0"].bounds_on is True
    assert np.allclose(params2["p0"].bounds, [0.0, 5.0])

    assert np.isclose(params2["p1"].value, -0.5)
    assert params2["p1"].fixed is True

# --- FROM test_project_json_roundtrip.py ---

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

# --- FROM test_project_v3.py ---
import json
import os
import tempfile
import unittest

from chisurf.core.project import Project, save_project, load_project


class TestProjectFormat(unittest.TestCase):

    def test_v4_format_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_v4_project")

            p = Project(
                name="v4_test_project",
                description="Project v4 format test",
                chisurf_version="test-version",
                project_format_version=4,
            )
            p.datasets["ds1"] = {"path": "data/file1.dat", "checksum": "abc123", "uid": "ds-uid-1"}
            p.datasets["ds2"] = {"path": "data/file2.dat", "checksum": "def456", "uid": "ds-uid-2"}
            p.experiments["exp1"] = {"type": "tcspc", "dataset_id": "ds-uid-1", "uid": "exp-uid-1"}
            p.fits.append({
                "uid": "fit-uid-1",
                "name": "fit1",
                "dataset_uid": "ds-uid-1",
                "created": "2024-01-01T00:00:00",
                "local_fits": [
                    {"uid": "lf-uid-1", "name": "local1", "parameters": []}
                ]
            })
            p.ui_state["current_experiment_id"] = "exp-uid-1"
            p.metadata["checkpoint_interval"] = 50

            archive_path = save_project(p, project_dir)
            assert archive_path.is_file()

            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                raw = json.loads(zf.read("project.json"))

            self.assertEqual(raw["project_format_version"], 4)
            self.assertIn("meta", raw)
            self.assertEqual(raw["meta"]["name"], "v4_test_project")
            self.assertIn("ds1", raw["datasets"])

            loaded = load_project(project_dir)
            self.assertEqual(loaded.project_format_version, 4)
            self.assertEqual(loaded.name, "v4_test_project")
            self.assertEqual(len(loaded.datasets), 2)
            self.assertEqual(len(loaded.fits), 1)
            self.assertEqual(loaded.fits[0]["uid"], "fit-uid-1")

    def test_v4_deterministic_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_ordering")

            p = Project(
                name="ordering_test",
                description="Test",
                project_format_version=4,
            )
            p.datasets["z_dataset"] = {"uid": "z"}
            p.datasets["a_dataset"] = {"uid": "a"}
            p.datasets["m_dataset"] = {"uid": "m"}

            archive_path = save_project(p, project_dir)

            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                raw = json.loads(zf.read("project.json"))

            dataset_keys = list(raw["datasets"].keys())
            self.assertEqual(dataset_keys, sorted(dataset_keys))

    def test_v4_always_outputs_v4(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "test_v1")

            p = Project(
                name="v4_project",
                description="V4 format",
                project_format_version=4,
            )
            p.datasets["ds1"] = {"path": "data/file1.dat"}

            archive_path = save_project(p, project_dir)

            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                raw = json.loads(zf.read("project.json"))

            self.assertEqual(raw["project_format_version"], 4)
            self.assertIn("meta", raw)

            loaded = load_project(project_dir)
            self.assertEqual(loaded.project_format_version, 4)
            self.assertEqual(loaded.name, "v4_project")

    def test_get_dataset_by_uid(self):
        p = Project(name="test", project_format_version=4)
        p.datasets["uid1"] = {"name": "dataset1"}
        p.datasets["uid2"] = {"name": "dataset2"}

        self.assertEqual(p.get_dataset("uid1")["name"], "dataset1")
        self.assertIsNone(p.get_dataset("nonexistent"))

    def test_get_fit_by_uid(self):
        p = Project(name="test", project_format_version=4)
        p.fits.append({"uid": "fit1", "name": "Fit 1"})
        p.fits.append({"uid": "fit2", "name": "Fit 2"})

        self.assertEqual(p.get_fit("fit1")["name"], "Fit 1")
        self.assertIsNone(p.get_fit("nonexistent"))

    def test_list_uids(self):
        p = Project(name="test", project_format_version=4)
        p.datasets["z"] = {"uid": "z"}
        p.datasets["a"] = {"uid": "a"}
        p.datasets["m"] = {"uid": "m"}
        p.fits.append({"uid": "fit3"})
        p.fits.append({"uid": "fit1"})
        p.fits.append({"uid": "fit2"})

        self.assertEqual(p.list_dataset_uids(), ["a", "m", "z"])
        self.assertEqual(p.list_fit_uids(), ["fit1", "fit2", "fit3"])


