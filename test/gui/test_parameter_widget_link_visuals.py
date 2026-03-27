from pathlib import Path


def test_group_unlink_refreshes_visual_state_for_related_widgets_contract():
    path = Path(__file__).resolve().parents[1] / "chisurf" / "gui" / "widgets" / "fitting" / "parameter_widgets.py"
    src = path.read_text(encoding="utf-8")

    assert "def _refresh_group_link_visuals(self):" in src
    assert "QtCore.QTimer.singleShot(100, self._refresh_group_link_visuals)" in src
    assert "self._refresh_group_link_visuals()" in src
