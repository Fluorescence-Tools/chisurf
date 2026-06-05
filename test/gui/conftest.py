import pytest
from qtpy.QtWidgets import QApplication

import chisurf.gui


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow tests (reader-heavy PDB, TTTR, etc.)",
    )
    parser.addoption(
        "--run-xfail", action="store_true", default=False,
        help="Run expected-to-fail (xfail) tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Use --run-slow to include")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--run-xfail"):
        skip_xfail = pytest.mark.skip(reason="Use --run-xfail to include")
        for item in items:
            if "xfail" in item.keywords:
                item.add_marker(skip_xfail)


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication fixture for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def chisurf_app(qtbot):
    """Bootstrap ChiSurf's main window and register with qtbot."""
    app = chisurf.gui.get_app()
    qtbot.addWidget(app.cs)
    yield app
