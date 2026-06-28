"""Tests for the optical components dock and spectrum view."""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from qtpy import QtCore, QtWidgets

from chisurf.gui.autoform.sections.registry import get_section_factory


def _probe_data(probe_id: int, name: str) -> dict:
    return {
        "probe": {"probe_id": probe_id, "chromophore_name": name},
        "spectra": [
            {
                "spectrum_type": "absorption",
                "wavelengths": [400, 500, 600],
                "intensity": [0.0, 1.0, 0.0],
            },
            {
                "spectrum_type": "emission",
                "wavelengths": [450, 550, 650],
                "intensity": [0.0, 1.0, 0.0],
            },
        ],
        "optical_properties": [],
    }


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ---------------------------------------------------------------------------
# SpectrumView – imperative API
# ---------------------------------------------------------------------------


def test_display_single_probe(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    sv = SpectrumView()
    sv.display(_probe_data(1, "Cy3B"))
    assert sv.plot.plotItem.listDataItems()


def test_display_no_spectra_shows_empty(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    sv = SpectrumView()
    sv.display({"probe": {"probe_id": 1, "chromophore_name": "X"}, "spectra": []})
    assert not sv.plot.plotItem.listDataItems()


def test_display_multiple_probes(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    sv = SpectrumView()
    p1 = _probe_data(1, "Cy3B")
    p2 = _probe_data(2, "ATTO647N")
    sv.display_multiple([p1, p2])
    data_items = sv.plot.plotItem.listDataItems()
    assert len(data_items) == 4


def test_clear_removes_all_curves(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    sv = SpectrumView()
    sv.display(_probe_data(1, "Cy3B"))
    assert sv.plot.plotItem.listDataItems()
    sv.clear()
    assert not sv.plot.plotItem.listDataItems()


# ---------------------------------------------------------------------------
# SpectrumView – model-based / AutoForm protocol
# ---------------------------------------------------------------------------


def test_refresh_from_model_single(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    class FakeModel:
        spectra_data = _probe_data(1, "Cy3B")

    sv = SpectrumView(model=FakeModel(), target="spectra_data")
    data_items = sv.plot.plotItem.listDataItems()
    assert len(data_items) == 2


def test_refresh_from_model_list(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    class FakeModel:
        spectra_data = [
            _probe_data(1, "Cy3B"),
            _probe_data(2, "ATTO647N"),
        ]

    sv = SpectrumView(model=FakeModel(), target="spectra_data")
    data_items = sv.plot.plotItem.listDataItems()
    assert len(data_items) == 4


def test_refresh_after_construction(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    sv = SpectrumView()
    assert not sv.plot.plotItem.listDataItems()

    class FakeModel:
        spectra_data = _probe_data(1, "Cy3B")

    sv._model = FakeModel()
    sv._target = "spectra_data"
    sv.refresh()
    assert sv.plot.plotItem.listDataItems()


# ---------------------------------------------------------------------------
# AutoForm registration
# ---------------------------------------------------------------------------


def test_registered_as_custom_section(qapp):
    factory = get_section_factory("spectrum_view")
    assert factory is not None


def test_custom_section_creates_widget(qapp):
    from chisurf.gui.widgets.spectrum_view import SpectrumView

    factory = get_section_factory("spectrum_view")

    class FakeModel:
        spectra_data = _probe_data(1, "Cy3B")

    widget = factory(model=FakeModel(), target="spectra_data")
    assert isinstance(widget, SpectrumView)
    assert widget.plot.plotItem.listDataItems()
