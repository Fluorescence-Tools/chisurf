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

    view = ds.ModelView(sections=(ds.ValueSection(attr="notes", kind="text", label="Notes"),))
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

    view = ds.ModelView(
        sections=(ds.ValueSection(attr="ro", kind="text", label="RO", read_only=True),)
    )
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

    view = ds.ModelView(sections=(ds.ValueSection(attr="when", kind="date", label="When"),))
    model = _model(view, when="2026-06-28")
    form = AutoForm(model)
    vw = form.findChildren(ValueWidget)[0]
    assert isinstance(vw.editor, QtWidgets.QDateEdit)
    assert vw.editor.date().toString("yyyy-MM-dd") == "2026-06-28"

    vw.editor.setDate(vw.editor.date().addDays(1))
    assert model.when == "2026-06-29"


def test_secret_kind_masks_and_reveals(qapp):
    from qtpy import QtWidgets

    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ValueWidget

    view = ds.ModelView(sections=(ds.ValueSection(attr="api_key", kind="secret", label="API Key"),))
    model = _model(view, api_key="s3cr3t")
    form = AutoForm(model)
    vw = form.findChildren(ValueWidget)[0]

    # starts masked, holds the value, and edits commit back
    assert isinstance(vw.editor, QtWidgets.QLineEdit)
    assert vw.editor.echoMode() == QtWidgets.QLineEdit.Password
    assert vw.editor.text() == "s3cr3t"
    vw.editor.setText("new-key")
    vw.editor.editingFinished.emit()
    assert model.api_key == "new-key"

    # the reveal toggle flips the echo mode both ways
    assert hasattr(vw, "reveal")
    vw.reveal.setChecked(True)
    assert vw.editor.echoMode() == QtWidgets.QLineEdit.Normal
    vw.reveal.setChecked(False)
    assert vw.editor.echoMode() == QtWidgets.QLineEdit.Password


def test_editable_choice_commits_free_text(qapp):
    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ChoiceWidget

    view = ds.ModelView(
        sections=(
            ds.ChoiceSection(attr="model_id", label="Model", options=("a", "b"), editable=True),
        )
    )
    model = _model(view, model_id="a")
    form = AutoForm(model)
    cw = form.findChildren(ChoiceWidget)[0]
    assert cw.combo.isEditable()

    # picking a listed option commits it
    cw.combo.setCurrentIndex(1)
    assert model.model_id == "b"

    # a hand-typed value not among the options commits verbatim on editing-finished
    cw.combo.setEditText("custom-model")
    cw.combo.lineEdit().editingFinished.emit()
    assert model.model_id == "custom-model"

    # sync re-displays a custom value that is not in the option list
    model.model_id = "another-custom"
    form.sync_fields()
    assert cw.combo.currentText() == "another-custom"


def test_button_row_flushes_pending_field_edit(qapp):
    """A value typed but not yet committed is flushed before the action runs.

    Reproduces the AI-settings 401: the API key was typed but a NoFocus tool
    button did not blur the field, so the un-committed key was missing when the
    action read the model.
    """
    import chisurf.core.dataspec as ds
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import ButtonRowWidget, ValueWidget

    view = ds.ModelView(
        sections=(
            ds.ValueSection(attr="api_key", kind="secret", label="API Key"),
            ds.ButtonRowSection(buttons=({"label": "Test", "action": "do_test"},)),
        )
    )
    seen = {}
    model = _model(view, api_key="", do_test=lambda: seen.update(key=model.api_key))
    form = AutoForm(model)
    form.show()

    vw = form.findChildren(ValueWidget)[0]
    vw.editor.setFocus()
    qapp.processEvents()  # let the focus event register (as it is in the live app)
    vw.editor.setText("sk-typed-not-committed")
    assert model.api_key == ""  # not committed yet (no focus-out)

    btn = form.findChildren(ButtonRowWidget)[0].findChildren(_tool_button())[0]
    btn.click()
    assert model.api_key == "sk-typed-not-committed"
    assert seen.get("key") == "sk-typed-not-committed"


def _tool_button():
    from qtpy import QtWidgets

    return QtWidgets.QToolButton
