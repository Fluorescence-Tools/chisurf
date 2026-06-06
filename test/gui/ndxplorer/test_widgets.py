import sys
import pathlib

import pytest


@pytest.fixture(scope="session")
def qapp():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _ensure_ndxplorer_path():
    ndxplorer_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "modules" / "ndxplorer"
    )
    if str(ndxplorer_path) not in sys.path:
        sys.path.insert(0, str(ndxplorer_path))


class TestUIComponents:
    def test_histogram_controls_creation(self, qapp):
        _ensure_ndxplorer_path()
        try:
            from ndxplorer.ui.histogram_controls import HistogramControls
            widget = HistogramControls()
            assert widget is not None
        except ImportError as e:
            pytest.skip(f"ndxplorer import failed: {e}")

    def test_selection_panel_creation(self, qapp):
        _ensure_ndxplorer_path()
        try:
            from ndxplorer.ui.selection_panel import SelectionPanel
            widget = SelectionPanel()
            assert widget is not None
        except ImportError as e:
            pytest.skip(f"ndxplorer import failed: {e}")

    def test_parameter_editor_creation(self, qapp):
        _ensure_ndxplorer_path()
        try:
            from ndxplorer.ui.parameter_editor import ParameterEditor
            widget = ParameterEditor()
            assert widget is not None
        except ImportError as e:
            pytest.skip(f"ndxplorer import failed: {e}")
