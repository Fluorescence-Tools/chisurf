"""Offscreen-Qt tests for AutoForm generic table sections."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy import QtWidgets
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"qtpy unavailable: {exc}")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_table_section_loads_from_json():
    import chisurf.core.dataspec as ds

    view = ds.load_view_spec({
        "sections": [{
            "type": "table",
            "source": "rows",
            "selected_attr": "selected",
            "columns": [
                {"key": "sample_id", "label": "Sample"},
                {"key": "description", "label": "Description"},
            ],
        }]
    })

    section = view.sections[0]
    assert isinstance(section, ds.TableSection)
    assert section.columns[0]["key"] == "sample_id"
    assert section.selected_attr == "selected"


def test_autoform_renders_table_and_selection(qapp):
    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import TableWidget

    class Model:
        selected = {}

        def __init__(self):
            self.activated = None
            self._rows = [
                {"sample_id": "s1", "description": "first"},
                {"sample_id": "s2", "description": "second"},
            ]

        def rows(self):
            return self._rows

        def open_row(self, row):
            self.activated = row

        def view_spec(self):
            return ds.ModelView(sections=(ds.TableSection(
                source="rows",
                selected_attr="selected",
                activated_call="open_row",
                columns=(
                    {"key": "sample_id", "label": "Sample"},
                    {"key": "description", "label": "Description"},
                ),
            ),))

    model = Model()
    form = AutoForm(model)
    table = form.findChild(TableWidget)
    assert table is not None
    assert table.rowCount() == 2
    assert table.item(1, 0).text() == "s2"

    table.selectRow(1)
    assert model.selected == {"sample_id": "s2", "description": "second"}
    table._activate_current_row()
    assert model.activated == {"sample_id": "s2", "description": "second"}


def test_table_refreshes_from_model(qapp):
    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import TableWidget

    class Model:
        def __init__(self):
            self._rows = [{"id": "a"}]

        def rows(self):
            return self._rows

        def view_spec(self):
            return ds.ModelView(sections=(ds.TableSection(
                source="rows",
                columns=({"key": "id", "label": "ID"},),
            ),))

    model = Model()
    form = AutoForm(model)
    table = form.findChild(TableWidget)
    assert table.rowCount() == 1
    model._rows.append({"id": "b"})
    form.refresh_plots()
    assert table.rowCount() == 2
