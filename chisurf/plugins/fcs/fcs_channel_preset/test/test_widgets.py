from __future__ import annotations

from pathlib import Path

import pytest
from qtpy import QtWidgets


def test_fcs_channel_dialog(qapp, qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Construction uses an injected temp database and known user."""
    db_path = str(tmp_path / "sample_management.db")

    monkeypatch.setattr(
        "chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils.resolve_active_user_id",
        lambda: "user_default",
    )
    monkeypatch.setattr(
        "chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups.resolve_active_user_id",
        lambda: "user_default",
    )

    from chisurf.plugins.fcs.fcs_channel_preset import FCSChannelWidget
    widget = FCSChannelWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "FCS Channel Definitions" in widget.windowTitle()
