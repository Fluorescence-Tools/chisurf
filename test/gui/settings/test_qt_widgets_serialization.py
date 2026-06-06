import logging
import pytest
from qtpy.QtWidgets import QSpinBox
from chisurf.core.base import Base


class QtSerializationWidget(Base):
    def __init__(self):
        super().__init__(name="TestWidget")
        self.spinbox = QSpinBox()
        self.regular_value = 42
        self.text_value = "This is a test"


@pytest.fixture
def qapp():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_serialization_fails_without_skip_qt_widgets(qapp, caplog):
    caplog.set_level(logging.DEBUG)
    test_widget = TestWidget()

    logging.info("Attempting to serialize without skipping Qt widgets...")
    with pytest.raises(Exception):
        test_widget.to_dict(convert_values_to_elementary=True)


def test_serialization_succeeds_with_skip_qt_widgets(qapp, caplog):
    caplog.set_level(logging.DEBUG)
    test_widget = TestWidget()

    logging.info("Attempting to serialize with skipping Qt widgets...")
    result = test_widget.to_dict(convert_values_to_elementary=True, skip_qt_widgets=True)
    logging.info(f"Result: {result}")

    assert result is not None
    assert "spinbox" not in result
    assert result["regular_value"] == 42
    assert result["text_value"] == "This is a test"


def test_save_to_yaml_with_skip_qt_widgets(qapp, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    test_widget = TestWidget()

    yaml_path = tmp_path / "test_widget.yaml"
    logging.info("Attempting to save to YAML with skipping Qt widgets...")
    test_widget.save(str(yaml_path), skip_qt_widgets=True)

    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "spinbox" not in content
    assert "regular_value" in content
    assert "text_value" in content
