from __future__ import annotations
import math
from qtpy import QtWidgets, QtCore, QtGui


class ScientificDoubleSpinBox(QtWidgets.QAbstractSpinBox):
    """Numeric spinbox with scientific-notation display and decimal stepping.

    Reproduces the pg.SpinBox interface used in parameter_widgets:
      value() / setValue() / editingFinished / opts dict
      finite=False allows ±inf; dec=True uses ±1% multiplicative steps.
    """

    def __init__(
        self,
        dec: bool = True,
        decimals: int = 4,
        suffix: str = "",
        finite: bool = True,
        value: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self._dec = dec
        self._decimals = decimals
        self._suffix = suffix
        self._finite = finite
        self._value = float(value)
        self._step_factor = 0.01  # 1% per step in dec mode
        # Backwards-compat dict read by FittingParameterDetailPopup
        self.opts: dict = {"decimals": decimals, "compactHeight": False}

        self._refresh_display()
        self.lineEdit().editingFinished.connect(self._commit_text)

    # ------------------------------------------------------------------ value
    def value(self) -> float:
        return self._value

    def setValue(self, v: float) -> None:
        v = float(v)
        if v == self._value:
            return
        self._value = v
        self._refresh_display()

    # ---------------------------------------------------------- display helpers
    def _format(self, v: float) -> str:
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.{self._decimals}g}"

    def _refresh_display(self) -> None:
        le = self.lineEdit()
        le.blockSignals(True)
        le.setText(self._format(self._value) + self._suffix)
        le.blockSignals(False)

    # ------------------------------------------------------------ input commit
    def _commit_text(self) -> None:
        text = self.lineEdit().text().strip()
        if self._suffix and text.endswith(self._suffix):
            text = text[: -len(self._suffix)].strip()
        try:
            v = float(text)
        except ValueError:
            self._refresh_display()
            return
        if self._value != v:
            self._value = v
            self._refresh_display()
        # editingFinished already emitted by inner QLineEdit — don't re-emit

    # ------------------------------------------------ QAbstractSpinBox contract
    def stepBy(self, steps: int) -> None:
        if self._dec and self._value != 0.0:
            self._value *= (1.0 + self._step_factor) ** steps
        else:
            self._value += float(steps)
        self._refresh_display()
        self.editingFinished.emit()

    def stepEnabled(self) -> QtWidgets.QAbstractSpinBox.StepEnabled:
        return self.StepUpEnabled | self.StepDownEnabled

    def validate(self, text: str, pos: int):
        stripped = text.strip()
        if self._suffix and stripped.endswith(self._suffix):
            stripped = stripped[: -len(self._suffix)].strip()
        if stripped in ("", "-", "+", "inf", "-inf", "+inf"):
            return (QtGui.QValidator.Intermediate, text, pos)
        if stripped and stripped[-1] in ("e", "E"):
            return (QtGui.QValidator.Intermediate, text, pos)
        if len(stripped) >= 2 and stripped[-2:] in ("e-", "e+", "E-", "E+"):
            return (QtGui.QValidator.Intermediate, text, pos)
        try:
            float(stripped)
            return (QtGui.QValidator.Acceptable, text, pos)
        except ValueError:
            return (QtGui.QValidator.Invalid, text, pos)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self.isEnabled() or self.isReadOnly():
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        self.stepBy(steps)
        event.accept()
