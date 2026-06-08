import sys
import pathlib
import pytest

_topdir = pathlib.Path(__file__).resolve().parents[2]
if str(_topdir) not in sys.path:
    sys.path.insert(0, str(_topdir))


@pytest.fixture(scope="session")
def qapp():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
