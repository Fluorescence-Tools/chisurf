from pathlib import Path


def test_add_fit_batches_ui_updates_for_multi_dataset_adds_contract():
    path = Path(__file__).resolve().parents[1] / "chisurf" / "macros" / "core_fit.py"
    src = path.read_text(encoding="utf-8")

    assert "_defer_cs_update" in src
    assert "_ui_updates_frozen" in src
    assert "dataset_indices=[idx]" in src
    assert "_defer_cs_update=True" in src
    assert "_ui_updates_frozen=True" in src


def test_table_plot_avoids_resize_to_contents_and_hidden_refresh_contract():
    path = Path(__file__).resolve().parents[1] / "chisurf" / "plots" / "table_plot.py"
    src = path.read_text(encoding="utf-8")

    assert "QHeaderView.Interactive" in src
    assert "copy_curves=False" in src
    assert "if not self.isVisible():" in src
    assert "self._refresh_pending = True" in src
