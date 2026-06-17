import json

import numpy as np

import chisurf as cs
from chisurf.core.data import DataCurve
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.models.model import ModelCurve
from chisurf.macros.core_fit import (
    _decode_curve_payload,
    _encode_curve_array,
    load_project,
    save_project,
)


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


def test_project_curve_payload_uses_base64_arrays():
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    payload = {
        "x": _encode_curve_array(x),
        "y": _encode_curve_array(np.array([0.0, 1.0, 4.0], dtype=float)),
        "ex": _encode_curve_array(np.zeros_like(x)),
        "ey": _encode_curve_array(np.ones_like(x)),
    }
    assert isinstance(payload["x"], dict)
    assert payload["x"]["encoding"] == "base64"

    dx, dy, dex, dey = _decode_curve_payload(payload)
    np.testing.assert_allclose(dx, x)
    np.testing.assert_allclose(dy, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(dex, np.zeros_like(x))
    np.testing.assert_allclose(dey, np.ones_like(x))

    legacy_payload = {
        "x": x.tolist(),
        "y": [0.0, 1.0, 4.0],
        "ex": np.zeros_like(x).tolist(),
        "ey": np.ones_like(x).tolist(),
    }
    lx, ly, lex, ley = _decode_curve_payload(legacy_payload)
    np.testing.assert_allclose(lx, x)
    np.testing.assert_allclose(ly, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(lex, np.zeros_like(x))
    np.testing.assert_allclose(ley, np.ones_like(x))


def test_headless_project_save_load(tmp_path):
    """
    Test that save_project and load_project can be called headlessly
    (without ever initializing a GUI or setting `cs.cs` to a window).
    """
    # 1. Ensure headless state
    assert getattr(cs, "cs", None) is None, "Test must run without a GUI instance"

    # 2. Setup some dummy data and a fit group
    cs.fits.clear()
    cs.imported_datasets.clear()

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dc = DataCurve(x=x, y=y, name="headless_data")
    cs.imported_datasets.append(dc)

    fit_group = FitGroup(data=[dc], model_class=DummyLinearModel)
    local_fit = fit_group.grouped_fits[0]
    local_fit.fit_range = (10, 90)

    cs.fits.append(fit_group)

    # 3. Save the project headlessly
    project_dir = tmp_path / "test_macro_save"
    archive_path = save_project(str(tmp_path), "test_macro_save")

    assert archive_path.is_file()
    assert archive_path.suffix == ".csp"

    import zipfile
    with zipfile.ZipFile(archive_path, "r") as zf:
        raw = json.loads(zf.read("project.json"))
    ds = raw["datasets"]["ds000"]
    assert isinstance(ds["x"], dict)
    assert ds["x"]["encoding"] == "base64"
    assert "data" in ds["x"]
    assert isinstance(ds["y"], dict)
    assert ds["y"]["encoding"] == "base64"

    # 4. Clear current state to simulate a fresh load
    cs.fits.clear()
    cs.imported_datasets.clear()

    # 5. Load the project headlessly
    load_project(str(project_dir))

    # 6. Verify restored state
    restored_dc = next(dc for dc in cs.imported_datasets if dc.name == "headless_data")
    assert restored_dc.name == "headless_data"
    np.testing.assert_allclose(restored_dc.x, x)
    np.testing.assert_allclose(restored_dc.y, y)
    np.testing.assert_allclose(restored_dc.ex, np.zeros_like(x))
    np.testing.assert_allclose(restored_dc.ey, np.ones_like(y))

    assert len(cs.fits) == 1
    restored_fit_group = cs.fits[0]
    # Check that model name is populated via the project fallback parsing
    # and fit ranges correctly re-established headlessly.
    assert len(restored_fit_group.grouped_fits) == 1
    restored_local_fit = restored_fit_group.grouped_fits[0]
    assert restored_local_fit.fit_range == (10, 90)

    print("Headless save/load roundtrip successful!")
