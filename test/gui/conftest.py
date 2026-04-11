import pytest
import qtpy

# ensure we are using pytest-qt properly
# qtbot is provided by pytest-qt plugin
import chisurf.gui


@pytest.fixture
def chisurf_app(qtbot):
    """
    Bootstrap ChiSurf's main window and register it with qtbot for 
    proper teardown and event processing.
    """
    app = chisurf.gui.get_app()
    # We add the main window to qtbot so it simulates closing properly
    qtbot.addWidget(app.cs)
    yield app
