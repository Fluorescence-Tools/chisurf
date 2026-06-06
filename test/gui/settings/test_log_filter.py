import time
import pytest
from qtpy.QtWidgets import QMainWindow, QPlainTextEdit, QLineEdit, QVBoxLayout, QWidget, QPushButton, QLabel


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
