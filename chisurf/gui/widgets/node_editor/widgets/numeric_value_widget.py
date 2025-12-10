from __future__ import annotations

from qtpy import QtCore, QtWidgets


class NumericValueWidget(QtWidgets.QWidget):
    """Simple labeled numeric value editor for use inside node content areas.

    This wraps a QLabel + QDoubleSpinBox and emits valueChanged(float) when the
    numeric value changes. Styling is expected to be applied by the surrounding
    node via apply_node_ui_theme.
    """

    valueChanged = QtCore.Signal(float)
    editingChanged = QtCore.Signal(bool)

    def __init__(
        self,
        label: str = "Value",
        value: float = 0.0,
        minimum: float | None = None,
        maximum: float | None = None,
        step: float = 0.1,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._label = QtWidgets.QLabel(label, self)
        self._spin = QtWidgets.QDoubleSpinBox(self)
        self._spin.setDecimals(6)
        self._spin.setKeyboardTracking(False)
        self._spin.setValue(float(value))
        if minimum is not None:
            self._spin.setMinimum(float(minimum))
        if maximum is not None:
            self._spin.setMaximum(float(maximum))
        self._spin.setSingleStep(float(step))

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._label)
        layout.addWidget(self._spin, 1)

        self._spin.valueChanged.connect(self._on_value_changed)
        self._spin.editingFinished.connect(self._on_editing_finished)

    # ----- API -----------------------------------------------------------
    def value(self) -> float:
        return float(self._spin.value())

    def setValue(self, v: float) -> None:
        self._spin.setValue(float(v))

    def setLabelText(self, text: str) -> None:
        self._label.setText(text)

    # ----- Internal slots -----------------------------------------------
    def _on_value_changed(self, v: float) -> None:
        self.valueChanged.emit(float(v))
        self.editingChanged.emit(True)

    def _on_editing_finished(self) -> None:
        self.editingChanged.emit(False)
