import time
import pytest
from qtpy.QtWidgets import QMainWindow, QPlainTextEdit, QLineEdit, QVBoxLayout, QWidget, QPushButton, QLabel
from qtpy.QtWidgets import QCheckBox

from chisurf.gui import misc_helpers
from chisurf.gui.widgets.general import LogListWidget


class LogFilterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log Filter Test")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.plainTextEditLog = QPlainTextEdit()
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.plainTextEditLog)

        layout.addWidget(QLabel("Filter:"))
        self.lineEdit_LogFilter = QLineEdit()
        self.lineEdit_LogFilter.textChanged.connect(self.filter_log_content)
        layout.addWidget(self.lineEdit_LogFilter)

        add_log_button = QPushButton("Add Log Entry")
        add_log_button.clicked.connect(self.add_log_entry)
        layout.addWidget(add_log_button)

        clear_filter_button = QPushButton("Clear Filter")
        clear_filter_button.clicked.connect(self.clear_filter)
        layout.addWidget(clear_filter_button)

        self._original_log_content = ""

        for i in range(5):
            self.plainTextEditLog.appendPlainText(f"Initial log entry {i+1}")

        self._original_log_content = self.plainTextEditLog.toPlainText()

    def filter_log_content(self):
        filter_text = self.lineEdit_LogFilter.text().strip().lower()

        if not hasattr(self, '_original_log_content'):
            self._original_log_content = ""

        current_content = self.plainTextEditLog.toPlainText()

        if not filter_text or len(current_content) > len(self._original_log_content):
            self._original_log_content = current_content

        if not filter_text:
            self.plainTextEditLog.setPlainText(self._original_log_content)
            return

        lines = self._original_log_content.split('\n')
        filtered_lines = [line for line in lines if filter_text in line.lower()]

        self.plainTextEditLog.clear()

        if filtered_lines:
            self.plainTextEditLog.setPlainText('\n'.join(filtered_lines))
        else:
            self.plainTextEditLog.setPlainText("No matching log entries found.")

    def add_log_entry(self):
        import random
        prefixes = ["INFO", "DEBUG", "WARNING", "ERROR", "TEST"]
        prefix = random.choice(prefixes)
        self.plainTextEditLog.appendPlainText(f"{prefix}: New log entry at {time.strftime('%H:%M:%S')}")
        self.update_log_filter()

    def update_log_filter(self):
        if hasattr(self, 'lineEdit_LogFilter') and self.lineEdit_LogFilter.text().strip():
            self.filter_log_content()

    def clear_filter(self):
        self.lineEdit_LogFilter.clear()


@pytest.fixture
def qapp():
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_window_creation(qapp, qtbot):
    window = LogFilterWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Log Filter Test"
    assert window.plainTextEditLog is not None
    assert window.lineEdit_LogFilter is not None


def test_initial_log_entries_populated(qapp, qtbot):
    window = LogFilterWindow()
    qtbot.addWidget(window)

    content = window.plainTextEditLog.toPlainText()
    assert "Initial log entry 1" in content
    assert "Initial log entry 5" in content


def test_filter_positive_match(qapp, qtbot):
    window = LogFilterWindow()
    qtbot.addWidget(window)

    qtbot.keyClicks(window.lineEdit_LogFilter, "Initial")
    qtbot.wait(50)

    content = window.plainTextEditLog.toPlainText()
    assert "Initial log entry 1" in content
    assert "No matching log entries found." not in content


def test_filter_negative_match(qapp, qtbot):
    window = LogFilterWindow()
    qtbot.addWidget(window)

    qtbot.keyClicks(window.lineEdit_LogFilter, "ZZZZNONEXISTENT")
    qtbot.wait(50)

    content = window.plainTextEditLog.toPlainText()
    assert content == "No matching log entries found."


def test_clear_filter_restores_content(qapp, qtbot):
    window = LogFilterWindow()
    qtbot.addWidget(window)

    qtbot.keyClicks(window.lineEdit_LogFilter, "Initial")
    qtbot.wait(50)

    window.clear_filter()
    qtbot.wait(50)

    content = window.plainTextEditLog.toPlainText()
    assert "Initial log entry 1" in content
    assert "Initial log entry 5" in content


def test_add_log_entry_updates_content(qapp, qtbot):
    window = LogFilterWindow()
    qtbot.addWidget(window)

    window.add_log_entry()
    qtbot.wait(50)

    content = window.plainTextEditLog.toPlainText()
    assert "New log entry at" in content


def test_log_list_widget_uses_multiple_columns(qapp, qtbot):
    widget = LogListWidget()
    qtbot.addWidget(widget)

    record = type(
        "Record",
        (),
        {
            "asctime": "2026-06-10 12:00:00,001",
            "levelname": "INFO",
            "pathname": "/Users/tpeulen/dev/chisurf/full_origin.py",
            "module": "module",
            "name": "name",
        },
    )()
    widget.addItem("2026-06-10 12:00:00,001 - INFO - started", record=record)
    widget.addItem("2026-06-10 12:00:00,002 - WARNING - slow fit")

    assert widget.columnCount() == 4
    assert widget.horizontalHeaderItem(0).text() == "Time"
    assert widget.horizontalHeaderItem(1).text() == "Level"
    assert widget.horizontalHeaderItem(3).text() == "Message"
    assert widget.verticalHeader().defaultSectionSize() <= 22
    assert widget.item(0, 0).text() == "12:00:00"
    assert widget.item(0, 0).toolTip() == "2026-06-10 12:00:00,001"
    assert widget.item(0, 2).toolTip() == "/Users/tpeulen/dev/chisurf/full_origin.py"
    assert widget.item(0, 3).toolTip() == "started"
    assert widget.item(0, 1).text() == "INFO"
    assert widget.item(0, 3).text() == "started"
    assert widget.item(1, 1).text() == "WARNING"


def test_log_list_widget_filter_hides_rows(qapp, qtbot):
    widget = LogListWidget()
    qtbot.addWidget(widget)
    window = QWidget()
    window.plainTextEditLog = widget
    window.lineEdit_LogFilter = QLineEdit()
    window.checkBox_filter_hide = QCheckBox()

    widget.addItem("2026-06-10 12:00:00,001 - INFO - alpha message")
    widget.addItem("2026-06-10 12:00:00,002 - INFO - beta message")

    window.lineEdit_LogFilter.setText("alpha")
    window.checkBox_filter_hide.setChecked(True)
    misc_helpers.filter_log_content(window)

    assert widget.isRowHidden(0) is False
    assert widget.isRowHidden(1) is True


def test_log_list_widget_copy_selected_rows(qapp, qtbot):
    widget = LogListWidget()
    qtbot.addWidget(widget)

    widget.addItem("2026-06-10 12:00:00,001 - INFO - alpha")
    widget.addItem("2026-06-10 12:00:00,002 - WARNING - beta")
    widget.selectRow(1)
    widget.copy_selected_items()

    clipboard_text = qapp.clipboard().text()
    assert "2026-06-10 12:00:00,002" in clipboard_text
    assert "WARNING" in clipboard_text
    assert "beta" in clipboard_text
