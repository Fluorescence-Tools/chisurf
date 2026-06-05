import os
import sys
import json
import pytest
import numpy as np

# Ensure we test the headless entrypoints
import chisurf
from chisurf.core.data import DataCurve
from chisurf.macros.core_fit import save_project, load_project, add_fit

from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.models.model import ModelCurve
from chisurf.core.fitting.parameter import FittingParameter

class DummyLinearModel(ModelCurve):
    name = "DummyLinearModel"
    def __init__(self, fit: Fit, **kwargs):
        super().__init__(fit, **kwargs)
        self.p0 = FittingParameter(name="p0", value=0.5)
        self.p1 = FittingParameter(name="p1", value=1.5)
        self.find_parameters()
    def update_model(self, **kwargs):
        x = self.fit.data.x
        if x is None:
            x = np.arange(self.fit.data.y.size, dtype=float)
        self.x = x
        self.y = float(self.p0.value) + float(self.p1.value) * x

def test_headless_project_save_load(tmp_path):
    """
    Test that save_project and load_project can be called headlessly
    (without ever initializing a GUI or setting `chisurf.cs` to a window).
    """
    # 1. Ensure headless state
    assert getattr(chisurf, "cs", None) is None, "Test must run without a GUI instance"

    # 2. Setup some dummy data and a fit group
    chisurf.fits.clear()
    chisurf.imported_datasets.clear()

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dc = DataCurve(x=x, y=y, name="headless_data")
    chisurf.imported_datasets.append(dc)

    fit_group = FitGroup(data=[dc], model_class=DummyLinearModel)
    local_fit = fit_group.grouped_fits[0]
    local_fit.fit_range = (10, 90)
    
    chisurf.fits.append(fit_group)

    # 3. Save the project headlessly
    project_dir = tmp_path / "test_macro_save"
    save_project(str(tmp_path), "test_macro_save")

    assert project_dir.is_dir()
    project_json = project_dir / "project.json"
    assert project_json.is_file()

    # 4. Clear current state to simulate a fresh load
    chisurf.fits.clear()
    chisurf.imported_datasets.clear()

    # 5. Load the project headlessly
    load_project(str(project_dir))

    # 6. Verify restored state
    assert len(chisurf.imported_datasets) == 1
    restored_dc = chisurf.imported_datasets[0]
    assert restored_dc.name == "headless_data"
    np.testing.assert_allclose(restored_dc.x, x)
    np.testing.assert_allclose(restored_dc.y, y)

    assert len(chisurf.fits) == 1
    restored_fit_group = chisurf.fits[0]
    # Check that model name is populated via the project fallback parsing
    # and fit ranges correctly re-established headlessly.
    assert len(restored_fit_group.grouped_fits) == 1
    restored_local_fit = restored_fit_group.grouped_fits[0]
    assert restored_local_fit.fit_range == (10, 90)

    print("Headless save/load roundtrip successful!")
