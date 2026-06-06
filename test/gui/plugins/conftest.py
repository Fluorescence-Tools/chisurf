import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication fixture for plugin widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
