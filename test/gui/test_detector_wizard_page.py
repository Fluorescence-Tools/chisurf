"""Construction smoke test for the detector channel-definition page.

The headless repository/load tests do not build the widget, so a missing
Qt import (e.g. ``QComboBox`` for the calibration-date combobox) would slip
through. This test constructs the page to catch import/NameError regressions.
"""
from __future__ import annotations


def test_detector_wizard_page_constructs(qapp):
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_channel_definition import (
        DetectorWizardPage,
    )

    page = DetectorWizardPage(allow_finish=False)
    # The calibration-date combobox added for time-versioned calibration must
    # exist and start with a "Latest" entry.
    assert page.calibration_combo is not None
    assert page.calibration_combo.itemText(0) == "Latest"
