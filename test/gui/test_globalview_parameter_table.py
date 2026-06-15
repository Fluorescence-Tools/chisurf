from __future__ import annotations

from qtpy import QtCore

from chisurf.plugins.core.globalview.parameter_table_model import (
    COL_LINK_ROW,
    COL_PARAM,
    COL_ROW,
    ParameterTableModel,
)
from chisurf.plugins.core.globalview.parameter_table_view import ParameterFilterProxy


class _Param:
    """Minimal fitting parameter for Global View table tests."""

    def __init__(self, name: str):
        """Create a fake parameter.

        Parameters
        ----------
        name : str
            Parameter name.
        """
        self.name = name
        self.value = 1.0
        self.fixed = False
        self.bounds = (0.0, 10.0)
        self.bounds_on = False
        self.error_estimate = 0.0
        self.link = None
        self.unique_identifier = f"param-{name}"

    @property
    def is_linked(self) -> bool:
        """Return whether this parameter currently follows another."""
        return self.link is not None


class _Model:
    """Minimal fit model exposing parameter collections."""

    def __init__(self, params: list[_Param]):
        """Create a fake model.

        Parameters
        ----------
        params : list of _Param
            Parameters owned by the model.
        """
        self.parameters_all = params
        self.parameters_all_dict = {p.name: p for p in params}


class _Fit:
    """Minimal fit object for the parameter table."""

    def __init__(self, name: str, params: list[_Param]):
        """Create a fake fit.

        Parameters
        ----------
        name : str
            Fit name.
        params : list of _Param
            Parameters owned by the fit.
        """
        self.name = name
        self.model = _Model(params)
        self.unique_identifier = f"fit-{name}"


class _Client:
    """Fake fitting client that records link calls and mutates fake params."""

    def __init__(self, fits: list[_Fit]):
        """Create a fake fitting client.

        Parameters
        ----------
        fits : list of _Fit
            Fits exposed to the table.
        """
        self.fits = fits
        self.calls = []

    def get_fit_objects(self) -> list[_Fit]:
        """Return fake fit objects."""
        return self.fits

    def link_parameters(self, **kwargs):
        """Record a link request and update the source parameter."""
        self.calls.append(("link", kwargs))
        source = self.fits[kwargs["fit_index"]].model.parameters_all_dict[kwargs["parameter_name"]]
        target = self.fits[kwargs["target_fit_index"]].model.parameters_all_dict[
            kwargs["target_parameter_name"]
        ]
        source.link = target
        return {"ok": True}

    def unlink_parameter(self, **kwargs):
        """Record an unlink request and clear the source parameter."""
        self.calls.append(("unlink", kwargs))
        source = self.fits[kwargs["fit_index"]].model.parameters_all_dict[kwargs["parameter_name"]]
        source.link = None
        return {"ok": True}

    def update_fit(self, **kwargs):
        """Accept table finalization calls."""
        return {"ok": True}

    def model_finalize(self, **kwargs):
        """Accept table finalization calls."""
        return {"ok": True}


def test_link_row_uses_visible_proxy_row_after_sort(qapp, monkeypatch):
    """Entering a link row should resolve against the visible sorted table."""
    import chisurf.plugins.core.globalview.parameter_table_model as table_model

    p_b = _Param("b")
    p_a = _Param("a")
    fits = [_Fit("fit-b", [p_b]), _Fit("fit-a", [p_a])]
    client = _Client(fits)
    monkeypatch.setattr(table_model, "get_fitting_client", lambda: client)

    model = ParameterTableModel()
    model.refresh()

    proxy = ParameterFilterProxy()
    proxy.setSourceModel(model)
    proxy.sort(COL_PARAM, QtCore.Qt.AscendingOrder)

    visible_source_names = [
        proxy.data(proxy.index(row, COL_PARAM))
        for row in range(proxy.rowCount())
    ]
    assert visible_source_names == ["a", "b"]
    assert [proxy.data(proxy.index(row, COL_ROW)) for row in range(proxy.rowCount())] == ["1", "2"]

    b_proxy_index = proxy.index(1, COL_LINK_ROW)
    assert proxy.setData(b_proxy_index, "1", QtCore.Qt.EditRole)

    assert p_b.link is p_a
    assert client.calls[-1] == (
        "link",
        {
            "parameter_name": "b",
            "target_parameter_name": "a",
            "fit_index": 0,
            "target_fit_index": 1,
            "local_idx": None,
            "target_local_idx": None,
        },
    )
    assert proxy.data(b_proxy_index) == "1"


def test_link_row_tooltip_uses_visible_row_after_sort(qapp, monkeypatch):
    """The link tooltip should report the same row shown in the Row column."""
    import chisurf.plugins.core.globalview.parameter_table_model as table_model

    p_b = _Param("b")
    p_a = _Param("a")
    p_b.link = p_a
    fits = [_Fit("fit-b", [p_b]), _Fit("fit-a", [p_a])]
    client = _Client(fits)
    monkeypatch.setattr(table_model, "get_fitting_client", lambda: client)

    model = ParameterTableModel()
    model.refresh()

    proxy = ParameterFilterProxy()
    proxy.setSourceModel(model)
    proxy.sort(COL_PARAM, QtCore.Qt.AscendingOrder)

    tooltip = proxy.data(proxy.index(1, COL_LINK_ROW), QtCore.Qt.ToolTipRole)

    assert tooltip.startswith("Linked to row 1:")


def test_link_row_tooltip_reports_full_dependency(qapp, monkeypatch):
    """The link-row tooltip should include target row and fit context."""
    import chisurf.plugins.core.globalview.parameter_table_model as table_model

    p_source = _Param("source")
    p_target = _Param("target")
    p_source.link = p_target
    fits = [_Fit("source-fit", [p_source]), _Fit("target-fit", [p_target])]
    client = _Client(fits)
    monkeypatch.setattr(table_model, "get_fitting_client", lambda: client)

    model = ParameterTableModel()
    model.refresh()

    tooltip = model.data(model.index(0, COL_LINK_ROW), QtCore.Qt.ToolTipRole)

    assert "Linked to row 2" in tooltip
    assert "fit 1 (target-fit), parameter target" in tooltip
