"""Offscreen-Qt tests for AutoForm ValueSection ``text`` and ``date`` kinds."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    try:
        from qtpy import QtWidgets
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"qtpy unavailable: {exc}")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _model(view, **attrs):
    return SimpleNamespace(view_spec=lambda: view, **attrs)


def test_text_kind_renders_multiline_and_roundtrips(qapp):
    from qtpy import QtWidgets

    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ValueWidget

    view = ds.ModelView(sections=(
        ds.ValueSection(attr="notes", kind="text", label="Notes"),
    ))
    model = _model(view, notes="line 1\nline 2")
    form = AutoForm(model)

    vw = form.findChildren(ValueWidget)[0]
    assert isinstance(vw.editor, QtWidgets.QPlainTextEdit)
    assert vw.editor.toPlainText() == "line 1\nline 2"

    # editing + focus-out commits back to the model
    vw.editor.setPlainText("edited\ntext")
    vw.editor.editingFinished.emit()
    assert model.notes == "edited\ntext"

    # sync re-reads the model value
    model.notes = "synced"
    form.sync_fields()
    assert vw.editor.toPlainText() == "synced"


def test_text_kind_read_only_disables_editing(qapp):
    from qtpy import QtWidgets

    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ValueWidget

    view = ds.ModelView(sections=(
        ds.ValueSection(attr="ro", kind="text", label="RO", read_only=True),
    ))
    model = _model(view, ro="immutable")
    form = AutoForm(model)
    vw = form.findChildren(ValueWidget)[0]
    assert isinstance(vw.editor, QtWidgets.QPlainTextEdit)
    assert vw.editor.isReadOnly()


def test_date_kind_renders_and_roundtrips(qapp):
    from qtpy import QtWidgets

    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ValueWidget

    view = ds.ModelView(sections=(
        ds.ValueSection(attr="when", kind="date", label="When"),
    ))
    model = _model(view, when="2026-06-28")
    form = AutoForm(model)
    vw = form.findChildren(ValueWidget)[0]
    assert isinstance(vw.editor, QtWidgets.QDateEdit)
    assert vw.editor.date().toString("yyyy-MM-dd") == "2026-06-28"

    vw.editor.setDate(vw.editor.date().addDays(1))
    assert model.when == "2026-06-29"
