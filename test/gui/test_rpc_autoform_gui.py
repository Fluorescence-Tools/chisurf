"""Offscreen-Qt tests for ``AutoForm.from_rpc_method``.

The headless half of the contract lives in ``test/models/test_rpc_method_view.py``;
here we assert the Qt half: building a form from an RPC method declaration
surfaces the method's ``description`` and each parameter's JSON-Schema
``description`` as widget tooltips, and edits reach ``params()``.
"""

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


def _method():
    return {
        "name": "demo.run",
        "summary": "Run the demo.",
        "description": "Run the demo computation on the selected file.",
        "params_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "default": 1.5,
                    "description": "Detection threshold in photons per bin.",
                },
                "verbose": {"type": "boolean", "description": "Emit progress messages."},
            },
            "required": [],
        },
    }


def test_descriptions_surface_as_qt_tooltips(qapp):
    from qtpy import QtWidgets

    from chisurf.gui.autoform import AutoForm

    form = AutoForm.from_rpc_method(_method())
    tooltips = [w.toolTip() for w in form.findChildren(QtWidgets.QWidget) if w.toolTip()]
    joined = "\n".join(tooltips)
    assert "Run the demo computation on the selected file." in joined  # method-level
    assert "Detection threshold in photons per bin." in joined  # per-parameter
    assert "Emit progress messages." in joined


def test_edited_values_reach_params(qapp):
    from chisurf.gui.autoform import AutoForm

    form = AutoForm.from_rpc_method(_method())
    group = form.model._params_group
    group.threshold = 2.5
    group.verbose = True
    assert form.model.params() == {"threshold": 2.5, "verbose": True}
