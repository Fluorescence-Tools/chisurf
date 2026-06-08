from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import chisurf as cs
import chisurf.macros.core_fit as core_fit
from chisurf.core.data import DataCurve, DataGroup
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.core.models.model import ModelCurve


class DummyLinearModel(ModelCurve):
    name = "DummyLinearModel"

    def __init__(self, fit: Fit, **kwargs):  # type: ignore[override]
        super().__init__(fit, **kwargs)
        self.p0 = FittingParameter(name="p0", value=1.0)
        self.find_parameters()

    def update_model(self, **kwargs):  # type: ignore[override]
        x = self.fit.data.x
        self.x = x
        self.y = float(self.p0.value) * np.ones_like(x)


def _make_fit_group(n: int = 3) -> FitGroup:
    curves = []
    for i in range(n):
        x = np.linspace(0.0, 1.0, 8, dtype=float)
        y = np.ones_like(x) * float(i + 1)
        curves.append(DataCurve(x=x, y=y, name=f"d{i}"))
    data_group = DataGroup(curves)
    return FitGroup(data=data_group, model_class=DummyLinearModel)


def test_link_fit_group_always_uses_first_fit_as_master(monkeypatch):
    fit_group = _make_fit_group(3)

    # Select a non-first fit to emulate middle-checkbox clicks from other rows.
    fit_group.selected_fit = 2

    old_cs = getattr(cs, "cs", None)
    monkeypatch.setattr(cs, "cs", SimpleNamespace(current_fit=fit_group), raising=False)
    try:
        core_fit.link_fit_group("p0", 2)
    finally:
        monkeypatch.setattr(cs, "cs", old_cs, raising=False)

    p0_first = fit_group.grouped_fits[0].model.parameters_all_dict["p0"]
    p0_second = fit_group.grouped_fits[1].model.parameters_all_dict["p0"]
    p0_third = fit_group.grouped_fits[2].model.parameters_all_dict["p0"]

    assert bool(getattr(p0_first, "is_link_master", False)) is True
    assert p0_first.link is None

    assert p0_second.link is p0_first
    assert p0_third.link is p0_first
    assert bool(getattr(p0_second, "is_link_master", False)) is False
    assert bool(getattr(p0_third, "is_link_master", False)) is False
