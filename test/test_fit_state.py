from __future__ import annotations

import numpy as np

from chisurf.data import DataCurve
from chisurf.fitting.fit import Fit
from chisurf.fitting.parameter import FittingParameter
from chisurf.models.model import ModelCurve

import chisurf.models.tcspc.lifetime as lifetime_mod
import chisurf.models.tcspc.fret as fret_mod
import chisurf.models.pda.simple as pda_simple_mod
import chisurf.experiments
import chisurf.fitting

from chisurf.project.fit_state import (
    fit_to_state,
    apply_state_to_fit,
)


class DummyLinearModel(ModelCurve):
    """Minimal concrete model used for testing fit_state helpers.

    The model represents a simple line ``y = p0 + p1 * x`` with two
    :class:`FittingParameter` instances. This keeps the test independent of
    complex experiment‑specific models while still exercising the real
    parameter / chinet plumbing.
    """

    name = "DummyLinearModel"

    def __init__(self, fit: Fit, **kwargs):  # type: ignore[override]
        super().__init__(fit, **kwargs)
        # Two scalar parameters that will be discovered by find_parameters
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
        # Use the default ModelCurve behaviour
        super().update(**kwargs)


def _make_dummy_fit() -> Fit:
    x = np.arange(5, dtype=float)
    y = np.ones_like(x)
    data = DataCurve(x=x, y=y)
    return Fit(model_class=DummyLinearModel, data=data)


def test_fit_state_roundtrip_with_links():
    # Prepare original fit with non‑default parameter settings
    fit1 = _make_dummy_fit()
    m1 = fit1.model
    params1 = m1.parameters_all_dict

    # Sanity: we expect both parameters to be present
    assert "p0" in params1 and "p1" in params1

    params1["p0"].value = 3.14
    params1["p0"].bounds = (0.0, 10.0)
    params1["p0"].bounds_on = True

    params1["p1"].value = -1.23
    params1["p1"].fixed = True
    # Link p1 to p0
    params1["p1"].link = params1["p0"]

    state = fit_to_state(fit1)

    # Basic structure checks
    assert state["model_class"] == DummyLinearModel.__name__
    assert "parameters" in state
    assert set(state["parameters"].keys()) == {"p0", "p1"}

    p1_state = state["parameters"]["p1"]
    assert p1_state["fixed"] is True
    assert p1_state["link_target"] == "p0"

    # Create a fresh fit and apply the stored state
    fit2 = _make_dummy_fit()
    m2 = fit2.model
    params2 = m2.parameters_all_dict

    # Ensure the fresh fit starts out different
    assert not np.isclose(params2["p0"].value, 3.14)
    assert not np.isclose(params2["p1"].value, -1.23)

    apply_state_to_fit(fit2, state)

    # Values, bounds and flags should now match the original
    assert np.isclose(params2["p0"].value, 3.14)
    assert params2["p0"].bounds_on is True
    assert np.allclose(params2["p0"].bounds, [0.0, 10.0])

    assert params2["p1"].fixed is True

    # Linked parameter should share the master's value and point to it
    assert params2["p1"].link is params2["p0"]
    assert np.isclose(params2["p1"].value, params2["p0"].value)


def test_model_get_set_state_roundtrip():
    """Model.get_state/set_state should mirror fit_state helpers."""

    fit1 = _make_dummy_fit()
    m1 = fit1.model
    params1 = m1.parameters_all_dict

    params1["p0"].value = 4.2
    params1["p1"].value = -0.7
    params1["p1"].fixed = True
    params1["p1"].link = params1["p0"]

    state = m1.get_state()

    # Fresh model should start out different
    fit2 = _make_dummy_fit()
    m2 = fit2.model
    params2 = m2.parameters_all_dict
    assert not np.isclose(params2["p0"].value, 4.2)
    assert not np.isclose(params2["p1"].value, -0.7)

    m2.set_state(state)

    assert np.isclose(params2["p0"].value, 4.2)
    assert np.isclose(params2["p1"].value, params2["p0"].value)
    assert params2["p1"].fixed is True
    assert params2["p1"].link is params2["p0"]


def test_fit_get_set_state_roundtrip_and_update_called():
    """Fit.get_state/set_state must round-trip state and trigger update()."""

    fit1 = _make_dummy_fit()
    params1 = fit1.model.parameters_all_dict
    params1["p0"].value = 1.23
    params1["p1"].value = 4.56

    state = fit1.get_state()

    fit2 = _make_dummy_fit()
    params2 = fit2.model.parameters_all_dict

    # Ensure defaults differ
    assert not np.isclose(params2["p0"].value, 1.23)
    assert not np.isclose(params2["p1"].value, 4.56)

    # Track whether update() is invoked
    called = {"update": False}

    def _fake_update(*args, **kwargs):
        called["update"] = True

    fit2.update = _fake_update

    fit2.set_state(state)

    assert np.isclose(params2["p0"].value, 1.23)
    assert np.isclose(params2["p1"].value, 4.56)
    assert called["update"] is True


class _DummyModelForFinalize(ModelCurve):
    """Model stub used to verify Fit.set_state behaviour.

    It records whether ``set_state`` and ``finalize`` have been called so we
    can assert that :meth:`Fit.set_state` delegates correctly.
    """

    name = "DummyFinalizeModel"

    def __init__(self, fit: Fit, **kwargs):  # type: ignore[override]
        super().__init__(fit, **kwargs)
        self._flag_set_state_called = False
        self._flag_finalize_called = False

    def update_model(self, **kwargs):  # type: ignore[override]
        # Minimal implementation: keep y array shape consistent with x
        if getattr(self, "x", None) is None:
            x = self.fit.data.x
            if x is None:
                x = np.arange(self.fit.data.y.size, dtype=float)
            self.x = x
        self.y = np.zeros_like(self.x)

    def update(self, **kwargs) -> None:  # type: ignore[override]
        super().update(**kwargs)

    def get_state(self) -> dict:  # pragma: no cover - trivial
        return {"marker": 1}

    def set_state(self, state: dict) -> None:
        self._flag_set_state_called = True

    def finalize(self):  # type: ignore[override]
        self._flag_finalize_called = True


def test_fit_set_state_calls_model_set_state_and_finalize():
    """Fit.set_state must delegate to model.set_state and then finalize()."""

    x = np.arange(3, dtype=float)
    y = np.ones_like(x)
    data = DataCurve(x=x, y=y)
    fit = Fit(model_class=_DummyModelForFinalize, data=data)

    model = fit.model
    # Sanity: flags start out False
    assert model._flag_set_state_called is False
    assert model._flag_finalize_called is False

    # Apply an arbitrary state; our dummy model only records the calls.
    fit.set_state({"marker": 2})

    assert model._flag_set_state_called is True
    assert model._flag_finalize_called is True


def test_lifetime_model_get_set_state_preserves_lifetime_components():
    """LifetimeModel.get_state/set_state must preserve lifetimes count.

    This exercises the ``lifetimes_n`` structural extra in fit_state.
    """

    # Small dummy TCSPC-like dataset
    x = np.arange(10, dtype=float)
    y = np.ones_like(x)
    data = DataCurve(x=x, y=y)

    fit1 = Fit(model_class=lifetime_mod.LifetimeModel, data=data)
    m1 = fit1.model

    # Ensure multiple lifetime components
    m1.lifetimes.append()
    m1.lifetimes.append()
    n1 = len(m1.lifetimes)
    assert n1 >= 2

    state = m1.get_state()

    fit2 = Fit(model_class=lifetime_mod.LifetimeModel, data=data)
    m2 = fit2.model
    assert len(m2.lifetimes) != n1

    m2.set_state(state)
    assert len(m2.lifetimes) == n1


def test_fret_gaussian_model_get_set_state_preserves_gaussians():
    """GaussianModel.get_state/set_state must preserve Gaussians count."""

    # Minimal TCSPC-like dataset
    x = np.arange(10, dtype=float)
    y = np.exp(-x / 4.0)
    data = DataCurve(x=x, y=y)

    fit1 = Fit(model_class=chisurf.models.tcspc.fret.GaussianModel, data=data)
    m1 = fit1.model
    # Add two Gaussians
    m1.gaussians.append(mean=2.0, sigma=0.5, amplitude=1.0)
    m1.gaussians.append(mean=5.0, sigma=1.0, amplitude=0.5)
    n1 = len(m1.gaussians)
    assert n1 == 2

    state = m1.get_state()

    fit2 = Fit(model_class=chisurf.models.tcspc.fret.GaussianModel, data=data)
    m2 = fit2.model
    # Ensure starting configuration differs
    assert len(m2.gaussians) != n1

    m2.set_state(state)
    assert len(m2.gaussians) == n1


def test_pda_probch0_length_preserved_via_model_state():
    """PdaSimpleModel.get_state/set_state must preserve ProbCh0 count.

    Uses lightweight instances to avoid tttrlib dependencies.
    """

    m1 = pda_simple_mod.PdaSimpleModel.__new__(pda_simple_mod.PdaSimpleModel)
    m1.pch0 = pda_simple_mod.ProbCh0.__new__(pda_simple_mod.ProbCh0)
    m1.pch0._name = "pch0_stub"
    m1.parameters_all_dict = {}

    # Add two discrete PDA species
    m1.pch0.append(amplitude=1.0, pch0=0.2)
    m1.pch0.append(amplitude=2.0, pch0=0.8)
    n1 = len(m1.pch0)
    assert n1 == 2

    state = m1.get_state()

    m2 = pda_simple_mod.PdaSimpleModel.__new__(pda_simple_mod.PdaSimpleModel)
    m2.pch0 = pda_simple_mod.ProbCh0.__new__(pda_simple_mod.ProbCh0)
    m2.pch0._name = "pch0_stub"
    m2.parameters_all_dict = {}

    # Ensure starting configuration differs
    assert len(m2.pch0) != n1

    m2.set_state(state)
    assert len(m2.pch0) == n1


def test_pda_gaussian_distances_length_preserved_via_model_state():
    """PdaGaussianDistanceModel.get_state/set_state must preserve count.

    Uses lightweight instances to avoid tttrlib dependencies.
    """

    m1 = pda_simple_mod.PdaGaussianDistanceModel.__new__(pda_simple_mod.PdaGaussianDistanceModel)
    m1.distances = pda_simple_mod.PdaGaussianDistances.__new__(pda_simple_mod.PdaGaussianDistances)
    m1.distances._name = "pda_distances_stub"
    m1.parameters_all_dict = {}

    m1.distances.append(mean=50.0, sigma=5.0, amplitude=1.0)
    m1.distances.append(mean=60.0, sigma=6.0, amplitude=0.5)
    n1 = len(m1.distances)
    assert n1 == 2

    state = m1.get_state()

    m2 = pda_simple_mod.PdaGaussianDistanceModel.__new__(pda_simple_mod.PdaGaussianDistanceModel)
    m2.distances = pda_simple_mod.PdaGaussianDistances.__new__(pda_simple_mod.PdaGaussianDistances)
    m2.distances._name = "pda_distances_stub"
    m2.parameters_all_dict = {}

    assert len(m2.distances) != n1

    m2.set_state(state)
    assert len(m2.distances) == n1
