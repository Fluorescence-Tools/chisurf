"""Regression tests for the generalized table/tooltip widgets (gui/widgets).

These widgets were lifted out of the light-path simulator plugin. The crash this
guards against: ``TooltipItem`` was called with the generalized ``name=``/``key=``
keyword API while its ``__init__`` still used the old ``text``/``probe_id``
signature, raising ``TypeError`` at runtime when the probe table populated.
"""
import pytest
from qtpy import QtCore, QtWidgets


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_tooltip_item_generalized_kwargs(qapp):
    """TooltipItem accepts the general (name, key) kwargs and lazily renders."""
    from chisurf.gui.widgets.spectra_tooltip import SpectraTooltipItem, TooltipItem

    # SpectraTooltipItem must remain a backwards-compatible alias.
    assert SpectraTooltipItem is TooltipItem

    calls = []
    item = TooltipItem(name="Alexa488", key=42, render_fn=lambda k: calls.append(k) or f"<b>{k}</b>")
    # tooltip is rendered lazily and cached
    assert item.data(QtCore.Qt.ToolTipRole) == "<b>42</b>"
    assert item.data(QtCore.Qt.ToolTipRole) == "<b>42</b>"
    assert calls == [42], "render_fn(key) must be called exactly once and cached"


def test_filtered_table_populates_with_tooltip_factory(qapp):
    """The exact crash path: FilteredTableWidget + a TooltipItem item factory."""
    from chisurf.gui.widgets.filtered_table import FilteredTableWidget
    from chisurf.gui.widgets.spectra_tooltip import TooltipItem

    table = FilteredTableWidget()
    table.set_item_factory(
        lambda item, key: TooltipItem(name=item.get("name", ""), key=key, render_fn=lambda k: "")
    )
    table.set_data(
        [{"name": "A", "probe_id": 1}, {"name": "B", "probe_id": 2}],
        key_fn=lambda x: x.get("probe_id"),
    )
    assert table.table.rowCount() == 2
    table.set_selected(2)
    assert table.get_selected() == 2


def test_filtered_table_default_factory_is_domain_free(qapp):
    """The generic table's default factory must not depend on the spectra item."""
    from chisurf.gui.widgets.filtered_table import FilteredTableWidget
    from chisurf.gui.widgets.spectra_tooltip import TooltipItem

    table = FilteredTableWidget()
    table.set_data([{"name": "X", "probe_id": 9}], key_fn=lambda x: x.get("probe_id"))
    assert table.table.rowCount() == 1
    widget_item = table.table.item(0, 0)
    assert widget_item.text() == "X"
    # default item must be a plain item, not the tooltip specialization
    assert not isinstance(widget_item, TooltipItem)
