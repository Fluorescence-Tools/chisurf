from __future__ import annotations

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
