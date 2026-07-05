"""Offscreen-Qt tests for the AutoForm ``wizard`` and ``info`` sections."""

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


def _wizard_view():
    import chisurf.core.dataspec as ds

    return ds.load_view_spec(
        {
            "sections": [
                {
                    "type": "wizard",
                    "linear": True,
                    "steps": [
                        {
                            "title": "Intro",
                            "icon": "👋",
                            "subtitle": "hello",
                            "sections": [{"type": "info", "text": "<b>hi</b>"}],
                        },
                        {
                            "title": "Act",
                            "sections": [
                                {"type": "info", "source": "status"},
                                {
                                    "type": "button_row",
                                    "buttons": [{"label": "Do it", "action": "do_it"}],
                                },
                            ],
                            "complete_when": {"attr": "done", "equals": "True"},
                        },
                        {
                            "title": "End",
                            "optional": True,
                            "sections": [{"type": "info", "text": "bye"}],
                        },
                    ],
                }
            ]
        }
    )


def _make_model():
    m = SimpleNamespace(done=False)
    m.view_spec = _wizard_view
    m.status = lambda: "ready" if m.done else "not ready"

    def do_it():
        m.done = True

    m.do_it = do_it
    return m


def test_wizard_renders_two_columns_and_steps(qapp):
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.wizard_section import WizardWidget

    form = AutoForm(_make_model())
    wiz = form.findChild(WizardWidget)
    assert wiz is not None
    assert wiz.nav_list.count() == 3
    # icon + title in the nav label
    assert "Intro" in wiz.nav_list.item(0).text()
    # a step with no complete_when is treated as complete (informational)
    assert wiz.nav_list.item(0).text().startswith("✓")


def test_wizard_linear_gate_and_completion(qapp):
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.wizard_section import WizardWidget

    model = _make_model()
    form = AutoForm(model)
    wiz = form.findChild(WizardWidget)

    # step 0 (info-only, implicitly complete) -> Next enabled
    wiz.nav_list.setCurrentRow(0)
    assert wiz.next_btn.isEnabled()

    # step 1 gated by complete_when (done is False) -> Next disabled
    wiz.nav_list.setCurrentRow(1)
    assert not wiz.next_btn.isEnabled()

    # perform the action, refresh, and the gate lifts + ✓ appears
    model.do_it()
    wiz.refresh()
    assert wiz.next_btn.isEnabled()
    assert wiz.nav_list.item(1).text().startswith("✓")

    # optional last step never blocks; Finish replaces Next. (isHidden reflects the
    # explicit setVisible() calls even though the top-level window is never shown.)
    wiz.nav_list.setCurrentRow(2)
    assert wiz.next_btn.isHidden()
    assert not wiz.finish_btn.isHidden()


def test_info_section_static_and_dynamic(qapp):
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.builtin import InfoWidget

    model = _make_model()
    form = AutoForm(model)
    infos = form.findChildren(InfoWidget)
    assert infos, "no InfoWidget rendered"
    texts = [w.toPlainText() for w in infos]
    assert any("hi" in t for t in texts)

    # dynamic source reflects the model and updates on refresh
    dyn = next(w for w in infos if getattr(w._section, "source", ""))
    assert "not ready" in dyn.toPlainText()
    model.do_it()
    dyn.refresh()
    assert "ready" in dyn.toPlainText()
