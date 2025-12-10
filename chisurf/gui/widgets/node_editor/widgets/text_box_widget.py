from __future__ import annotations

from qtpy import QtCore, QtWidgets


class TextBoxWidget(QtWidgets.QWidget):
    """Simple labeled text box used inside node content areas.

    Inspired by QNodeEditor text entries: a label on the left and a QLineEdit
    on the right, emitting ``valueChanged`` when the text changes. Styling is
    left to the surrounding node via ``apply_node_ui_theme``.

    A minimal "label only" mode is provided so nodes can collapse the editor
    into just a label when an input is wired, mirroring the QNodeEditor UX.
    """

    valueChanged = QtCore.Signal(str)
    editingChanged = QtCore.Signal(bool)

    def __init__(self, label: str = "Text", value: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._label = QtWidgets.QLabel(label, self)
        self._edit = QtWidgets.QLineEdit(self)
        self._edit.setText(value)

        self._stack = QtWidgets.QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setSpacing(0)

        # Page 0: label + editor
        page_edit = QtWidgets.QWidget(self)
        lay = QtWidgets.QHBoxLayout(page_edit)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._label)
        lay.addWidget(self._edit, 1)
        self._stack.addWidget(page_edit)

        # Page 1: label only (used when an input port is connected)
        page_label_only = QtWidgets.QWidget(self)
        lay2 = QtWidgets.QHBoxLayout(page_label_only)
        lay2.setContentsMargins(0, 0, 0, 0)
        lay2.setSpacing(4)
        lay2.addWidget(self._label)
        self._stack.addWidget(page_label_only)

        self._stack.setCurrentIndex(0)

        self._edit.textChanged.connect(self.valueChanged)
        self._edit.editingFinished.connect(self._on_editing_finished)
        self._edit.textEdited.connect(self._on_editing_started)

    # ----- API -----------------------------------------------------------
    def text(self) -> str:
        return self._edit.text()

    def setText(self, text: str) -> None:
        self._edit.setText(text)

    def setLabelText(self, text: str) -> None:
        self._label.setText(text)

    def set_label_only(self, label_only: bool) -> None:
        """Show only the label (no line edit) when ``label_only`` is True."""

        self._stack.setCurrentIndex(1 if label_only else 0)

    # ----- Internal slots -----------------------------------------------
    def _on_editing_started(self, _text: str) -> None:
        self.editingChanged.emit(True)

    def _on_editing_finished(self) -> None:
        self.editingChanged.emit(False)
