from chisurf.gui import QtGui, QtWidgets, QtCore


QValidator = QtGui.QValidator


class ProgressWindow(QtWidgets.QDialog):
    def __init__(self, title="Processing Files", message="Loading files...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(message)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, max_value)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()


class CommaSeparatedIntegersValidator(QValidator):

    """
    QValidator to ensure input is a comma-separated list of valid
    integers between 0 and 255.
    """

    def validate(self, input_str, pos):
        if not input_str:
            return QValidator.Intermediate, input_str, pos

        parts = input_str.split(',')
        for part in parts:
            part = part.strip()
            if part == '':
                continue
            if not part.isdigit():
                return QValidator.Intermediate, input_str, pos
            num = int(part)
            if num < 0 or num > 255:
                return QValidator.Invalid, input_str, pos

        if input_str.endswith(','):
            return QValidator.Intermediate, input_str, pos

        return QValidator.Acceptable, input_str, pos

    def fixup(self, input_str):
        input_str = input_str.rstrip(',')
        parts = input_str.split(',')
        valid_parts = []
        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if 0 <= num <= 255:
                    valid_parts.append(str(num))
        return ', '.join(valid_parts)
