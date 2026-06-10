from __future__ import annotations

from pathlib import Path


def test_core_data_global_fit_guard_contracts():
    src = Path("chisurf/macros/core_data.py").read_text(encoding="utf-8")
    assert "def _is_global_fit_dataset(" in src
    assert "def restore_global_fit_dataset(" in src
    assert "dataset_restore_global_fit" in src


def test_dataset_actions_exposes_restore_global_fit_action_contract():
    src = Path("chisurf/core/actions/dataset_actions.py").read_text(encoding="utf-8")
    assert "@action(\"dataset.restore_global_fit\")" in src
