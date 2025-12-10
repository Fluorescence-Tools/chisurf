from __future__ import annotations

from typing import Iterable, List, Sequence

from qtpy import QtCore, QtGui, QtWidgets


class Vector1DWidget(QtWidgets.QWidget):
    """Simple 1D vector input: label + line edit with multi-value validation.

    The user enters a list of numeric values (e.g. ``"1, 2, 3"`` or
    ``"1 2 3"``). On successful parsing the widget emits ``valueChanged``
    with the list of floats. No slider is used; this is purely manual input
    with validation.
    """

    valueChanged = QtCore.Signal(object)

    def __init__(
        self,
        text: str,
        values: Sequence[float] | float = 0.0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label_text = str(text)
        self._values: List[float] = []

        self._values = self._normalize_values(values)

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._label = QtWidgets.QLabel(self._label_text)
        self._edit = QtWidgets.QLineEdit(self)
        self._edit.setText(self._format_values())
        self._edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._edit.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._edit.setAutoFillBackground(False)
        self._edit.setStyleSheet(
            "background: transparent; border: 1px solid rgb(40,40,40);"
            "color: rgb(235,235,235); padding: 1px 4px;"
        )
        lay.addWidget(self._label)
        lay.addWidget(self._edit, 1)

        self._edit.editingFinished.connect(self._on_editing_finished)

    # ----- Value handling --------------------------------------------------

    def _normalize_values(self, values: Sequence[float] | float) -> List[float]:
        out: List[float] = []
        try:
            if isinstance(values, (list, tuple)):
                for v in values:
                    out.append(float(v))
            else:
                out.append(float(values))
        except Exception:
            out = []
        # Ensure we always have at least two components so the vector
        # "contains more than one value". Default to three for readability.
        if not out:
            out = [0.0, 0.0, 0.0]
        if len(out) == 1:
            out = [out[0], 0.0, 0.0]
        return out

    def _format_values(self) -> str:
        return ", ".join(f"{v:g}" for v in self._values)

    def value(self) -> List[float]:
        return list(self._values)

    def setValue(self, values: Sequence[float] | float) -> None:
        new_vals = self._normalize_values(values)
        if new_vals == self._values:
            return
        self._values = new_vals
        self._edit.setText(self._format_values())
        # Emit a copy to avoid accidental external mutation.
        self.valueChanged.emit(self.value())

    # ----- Editing ---------------------------------------------------------

    def _on_editing_finished(self) -> None:
        text = self._edit.text().strip()
        if not text:
            # Restore previous value on empty input
            self._edit.setText(self._format_values())
            return
        parts = [p for p in [s.strip() for s in text.replace(";", ",").split(",")] if p]
        if len(parts) == 1 and " " in parts[0]:
            # Also support space-separated values: "1 2 3"
            parts = [p for p in parts[0].split() if p]
        vals: List[float] = []
        try:
            for p in parts:
                vals.append(float(p))
        except Exception:
            # Invalid input: revert to previous values
            self._edit.setText(self._format_values())
            return
        self.setValue(vals)
