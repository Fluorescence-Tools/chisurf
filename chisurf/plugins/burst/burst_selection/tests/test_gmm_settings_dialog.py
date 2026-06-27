"""Offscreen tests for the GMM settings dialog migrated onto AutoForm (PRD-40).

Proves the declarative AutoForm form (a) renders one control per field, (b)
reflects the initial settings, and (c) commits edits back so ``get_settings``
returns them — i.e. AutoForm works as a non-model, fixed-form consumer.
"""
import pytest
from qtpy import QtWidgets


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


CUSTOM = {
    "covariance_type": "diag",
    "random_state": 7,
    "max_iter": 500,
    "n_init": 3,
    "tol": 1e-4,
    "max_components": 5,
    "reg_covar": 1e-5,
}


def test_gmm_dialog_renders_and_reflects_settings(qapp):
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.burst.burst_selection.gui.gmm_settings_dialog import (
        GMMSettingsDialog,
    )

    dialog = GMMSettingsDialog(gmm_settings=dict(CUSTOM))

    forms = dialog.findChildren(AutoForm)
    assert len(forms) == 1
    # the simple fields are grouped into a single aligned QFormLayout
    form_layout = dialog.findChild(QtWidgets.QFormLayout)
    assert form_layout is not None
    assert form_layout.rowCount() == len(CUSTOM)

    # one editor per settings field, reflecting the provided initial values
    combos = dialog.findChildren(QtWidgets.QComboBox)
    spins = dialog.findChildren(QtWidgets.QSpinBox)
    dspins = dialog.findChildren(QtWidgets.QDoubleSpinBox)
    assert len(combos) + len(spins) + len(dspins) == len(CUSTOM)
    assert any(c.currentText() == "diag" for c in combos)
    assert any(s.value() == 500 for s in spins)

    # initial round-trip is loss-less
    assert dialog.get_settings() == CUSTOM


def test_gmm_dialog_commits_edits_without_a_fit(qapp):
    """Editing a control updates get_settings; no fit => no fit.update side effect."""
    from chisurf.plugins.burst.burst_selection.gui.gmm_settings_dialog import (
        GMMSettingsDialog,
    )

    dialog = GMMSettingsDialog(gmm_settings=dict(CUSTOM))
    max_iter_spin = next(
        s for s in dialog.findChildren(QtWidgets.QSpinBox) if s.value() == 500
    )
    max_iter_spin.setValue(999)  # valueChanged -> ValueWidget._commit -> setattr
    assert dialog.get_settings()["max_iter"] == 999


def test_new_gui_exposes_and_applies_gmm_settings(qapp, monkeypatch):
    """The new GUI's GMM-settings button updates settings and _gmm_model honours them.

    Regression: the dialog was previously only reachable from the legacy GUI, so
    the new GUI had no way to display or apply advanced GMM settings.
    """
    from chisurf.plugins.burst.burst_selection.gui import tool as tool_mod
    from chisurf.plugins.burst.burst_selection.gui.tool import BurstSelectionTool

    tool = BurstSelectionTool()
    try:
        # the settings button exists and is connected
        assert hasattr(tool, "gmm_settings_button")
        assert tool.gmm_settings == dict(tool_mod.DEFAULT_GMM_SETTINGS)

        # simulate accepting the dialog with edited settings
        edited = dict(tool.gmm_settings)
        edited.update(covariance_type="diag", max_iter=123, random_state=7)

        class _FakeDialog:
            def __init__(self, *a, **k):
                pass

            def exec_(self):
                return QtWidgets.QDialog.Accepted

            def get_settings(self):
                return edited

        monkeypatch.setattr(tool_mod, "GMMSettingsDialog", _FakeDialog)
        monkeypatch.setattr(tool, "_fit_gmm", lambda: None)  # avoid needing data
        tool._show_gmm_settings()

        assert tool.gmm_settings["covariance_type"] == "diag"
        assert tool.gmm_settings["max_iter"] == 123

        # _gmm_model builds an estimator carrying the configured settings
        model = tool._gmm_model(2)
        assert model.covariance_type == "diag"
        assert model.max_iter == 123
        assert model.random_state == 7
    finally:
        tool.close()
