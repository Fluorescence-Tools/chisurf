from __future__ import annotations

import numpy as np

from chisurf.core.data import DataCurve
from chisurf.core.fitting.fit import Fit
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.models.model import ModelCurve
from chisurf.core.models.global_model.globalfit import GlobalFitModel

from chisurf.core.project.fit_state import global_links_to_state, apply_global_links_state


class DummyLinearModelForGlobal(ModelCurve):
    """Minimal model used to exercise GlobalFitModel link serialization.

    The model is ``y = p0 + p1 * x`` with two :class:`FittingParameter`
    instances that will be referenced from :class:`GlobalFitModel` links.
    """

    name = "DummyLinearModelForGlobal"

    def __init__(self, fit: Fit, **kwargs):  # type: ignore[override]
        super().__init__(fit, **kwargs)
        self.p0 = FittingParameter(name="p0", value=1.0)
        self.p1 = FittingParameter(name="p1", value=2.0)
        self.find_parameters()

    def update_model(self, **kwargs):  # type: ignore[override]
        x = self.fit.data.x
        if x is None:
            x = np.arange(self.fit.data.y.size, dtype=float)
        self.x = x
        self.y = float(self.p0.value) + float(self.p1.value) * x

    def update(self, **kwargs) -> None:  # type: ignore[override]
        super().update(**kwargs)


def _make_fit() -> Fit:
    x = np.linspace(0.0, 3.0, 4, dtype=float)
    y = np.ones_like(x)
    data = DataCurve(x=x, y=y)
    return Fit(model_class=DummyLinearModelForGlobal, data=data)


def test_global_links_roundtrip_and_effect():
    # Create two independent fits to be coupled by a GlobalFitModel
    fit_a1 = _make_fit()
    fit_b1 = _make_fit()

    # Prepare a GlobalFitModel with a single cross-fit link definition
    gm1 = GlobalFitModel(fit=fit_a1, fits=[fit_a1, fit_b1])
    # Origin: fit index 0, parameter "p0"; target: p0 of fit index 1
    gm1.links = [[True, 0, "p0", "f[1]['p0']"]]

    # Serialize only the link table (no GUI or YAML involved)
    state = global_links_to_state(gm1)
    assert "links" in state
    assert state["links"], "Expected at least one serialized link record"

    rec = state["links"][0]
    assert rec["enabled"] is True
    assert rec["origin_fit_index"] == 0
    assert rec["origin_param_name"] == "p0"
    assert rec["formula"] == "f[1]['p0']"

    # Rebuild a fresh pair of fits and a new GlobalFitModel instance
    fit_a2 = _make_fit()
    fit_b2 = _make_fit()
    gm2 = GlobalFitModel(fit=fit_a2, fits=[fit_a2, fit_b2])
    # ``setLinks`` expects this attribute; disable clearing to keep testing simple
    gm2.clear_on_update = False

    # Apply the serialized link configuration
    apply_global_links_state(gm2, state)

    # After restoration, the internal link table should match the original
    assert isinstance(gm2.links, list)
    assert gm2.links == gm1.links

    # And serializing again should reproduce the same state
    state2 = global_links_to_state(gm2)
    assert state2 == state
