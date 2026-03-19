from pathlib import Path


def _lineplot_source() -> str:
    path = Path(__file__).resolve().parents[1] / "chisurf" / "plots" / "lineplot" / "lineplot.py"
    return path.read_text(encoding="utf-8")


def test_group_display_methods_exist():
    src = _lineplot_source()
    assert "def _plot_group_curves(self" in src
    assert "def _plot_single_fit_curves(self" in src
    assert "def _plot_active_fit_only(self" in src


def test_group_display_alpha_and_setalpha_contract():
    src = _lineplot_source()
    assert "line.setAlpha(int(alpha * 255), auto=False)" in src
    assert "alpha = 1.0" in src
    assert "alpha = 0.4" in src


def test_group_display_uses_selected_fit_when_available():
    src = _lineplot_source()
    assert "selected_fit" in src
    assert "hasattr(self.fit, 'grouped_fits')" in src or "hasattr(fit, 'grouped_fits')" in src
