from __future__ import annotations

from pathlib import Path


def test_csv_tcspc_dt_uses_spinbox_value_when_not_scaled_contract():
    src = Path("chisurf/gui/widgets/experiments/tcspc/csv_tcspc_widget.py").read_text(encoding="utf-8")
    assert "dt = dt_base * rebin if self.checkBox_2.isChecked() else dt_base" in src


def test_model_macros_expose_unload_irf_contract():
    src = Path("chisurf/macros/model.py").read_text(encoding="utf-8")
    assert "def unload_irf(" in src


def test_model_actions_irf_handlers_use_fit_object_contract():
    src = Path("chisurf/core/actions/model_actions.py").read_text(encoding="utf-8")
    assert "return model_macros.change_irf(int(irf_idx), str(irf_name), fit=fit)" in src
    assert "return model_macros.unload_irf(fit=fit)" in src


def test_convolve_widget_change_irf_does_not_immediately_unload_contract():
    src = Path("chisurf/gui/widgets/models/tcspc/convolve.py").read_text(encoding="utf-8")
    assert "name=\"model.change_irf\"" in src
    assert "name=\"model.unload_irf\"" in src
    assert "def _resolve_fit_group_index" in src
    assert "payload[\"fit_index\"]" in src
