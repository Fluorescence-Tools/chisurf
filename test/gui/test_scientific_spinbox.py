"""Offscreen-Qt tests for the ScientificDoubleSpinBox.

Tests that the Qt-native scientific double spinbox performs basic value setting,
clamping against min/max bounds, coercing integers in integer mode, and steps
appropriately in additive and decimal modes.
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


def test_spinbox_basics(qapp):
    from chisurf.gui.widgets.fitting.scientific_spinbox import ScientificDoubleSpinBox

    sb = ScientificDoubleSpinBox(value=1.5, min=0.0, max=10.0)
    assert sb.value() == 1.5

    sb.setValue(5.0)
    assert sb.value() == 5.0

    # Check clamping
    sb.setValue(15.0)
    assert sb.value() == 10.0

    sb.setValue(-5.0)
    assert sb.value() == 0.0


def test_spinbox_int_mode(qapp):
    from chisurf.gui.widgets.fitting.scientific_spinbox import ScientificDoubleSpinBox

    sb = ScientificDoubleSpinBox(value=3.2, int=True)
    assert sb.value() == 3  # coerced to int

    sb.setValue(5.7)
    assert sb.value() == 6


def test_spinbox_stepping(qapp):
    from chisurf.gui.widgets.fitting.scientific_spinbox import ScientificDoubleSpinBox

    # In decimal mode, stepping by 1 should multiply by 1.01
    sb = ScientificDoubleSpinBox(value=100.0, dec=True)
    sb.stepBy(1)
    assert sb.value() == 101.0

    # In standard mode with step=5.0
    sb = ScientificDoubleSpinBox(value=10.0, dec=False, step=5.0)
    sb.stepBy(1)
    assert sb.value() == 15.0
