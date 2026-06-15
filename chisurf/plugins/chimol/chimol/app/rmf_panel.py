"""RMF inspection dock for Chimol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ..io import RmfNotAvailableError, load_rmf_full


class RmfPlotWidget(QtWidgets.QWidget):
    """Small dependency-free line plot for RMF frame series."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the RMF plot widget."""
        super().__init__(parent)
        self.values = np.asarray([], dtype=float)
        self.current_frame = 0
        self.setMinimumHeight(160)

    def set_data(self, values: object, current_frame: int = 0) -> None:
        """Set plottable values and the active frame index."""
        self.values = np.asarray(values, dtype=float)
        self.current_frame = int(current_frame)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt API
        """Paint the series, axes, and current-frame marker."""
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(255, 255, 255))

        rect = self.rect().adjusted(38, 12, 12, 28)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        finite = self.values[np.isfinite(self.values)]
        if finite.size == 0:
            painter.setPen(QtGui.QColor(120, 120, 120))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No numeric RMF frame series")
            return

        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if y_max == y_min:
            y_max += 1.0
            y_min -= 1.0

        def to_point(index: int, value: float) -> QtCore.QPoint:
            x = rect.left() + (index / max(1, len(self.values) - 1)) * rect.width()
            y = rect.bottom() - ((value - y_min) / (y_max - y_min)) * rect.height()
            return QtCore.QPoint(int(round(x)), int(round(y)))

        painter.setPen(QtGui.QColor(210, 210, 210))
        painter.drawRect(rect)
        painter.setPen(QtGui.QColor(160, 160, 160))
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())

        painter.setPen(QtGui.QColor(40, 90, 180))
        path = QtGui.QPainterPath()
        first = True
        for index, value in enumerate(self.values):
            if not np.isfinite(value):
                first = True
                continue
            point = to_point(index, float(value))
            if first:
                path.moveTo(point)
                first = False
            else:
                path.lineTo(point)
        painter.drawPath(path)

        frame = max(0, min(int(self.current_frame), max(0, len(self.values) - 1)))
        if len(self.values) > 0:
            marker_x = rect.left() + (frame / max(1, len(self.values) - 1)) * rect.width()
            painter.setPen(QtGui.QPen(QtGui.QColor(220, 40, 40), 2))
            painter.drawLine(int(marker_x), rect.top(), int(marker_x), rect.bottom())
            painter.drawText(
                rect.adjusted(0, 0, 0, -4),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                str(frame + 1),
            )


class RmfPanel(QtCore.QObject):
    """Panel for RMF features and frame-score plotting (no outer QDockWidget)."""

    def __init__(self, parent: QtWidgets.QMainWindow, viewer: object) -> None:
        """Create the RMF panel attached to *viewer*."""
        super().__init__(parent)
        self.parent_window = parent
        self.viewer = viewer

        self._widget = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(self._widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QtWidgets.QLabel("RMF frame series")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self.series_combo = QtWidgets.QComboBox(self._widget)
        self.series_combo.currentIndexChanged.connect(self._on_series_changed)
        layout.addWidget(self.series_combo)

        self.plot = RmfPlotWidget(self._widget)
        layout.addWidget(self.plot, stretch=1)

        self.value_label = QtWidgets.QLabel("Current: --")
        layout.addWidget(self.value_label)

        button_row = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Refresh RMF", self._widget)
        self.refresh_button.clicked.connect(self.refresh_active_rmf)
        button_row.addWidget(self.refresh_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

    @property
    def widget(self) -> QtWidgets.QWidget:
        """Return the content widget."""
        return self._widget

    def set_state(self, state: object | None) -> None:
        """Update the panel for the active Chimol object state."""
        series = getattr(state, "rmf_frame_series", {}) if state is not None else {}
        if not isinstance(series, dict):
            series = {}

        current = self.series_combo.currentText()
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        names = [name for name, values in series.items() if _has_numeric_values(values)]
        self.series_combo.addItems(names)
        if current in names:
            self.series_combo.setCurrentText(current)
        self.series_combo.blockSignals(False)
        self._update_plot()

    def _on_series_changed(self, index: int) -> None:
        """Refresh the plot when the selected series changes."""
        del index
        self._update_plot()

    def _update_plot(self) -> None:
        """Draw the selected frame series."""
        state = _active_state(self.viewer)
        series = getattr(state, "rmf_frame_series", {}) if state is not None else {}
        if not isinstance(series, dict):
            series = {}

        name = self.series_combo.currentText()
        values = series.get(name, []) if name else []
        current = getattr(state, "active_frame", None)
        if current is None:
            try:
                current = self.viewer.get_current_frame()
            except Exception:
                current = 0

        self.plot.set_data(values, int(current or 0))
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            self.value_label.setText("Current: --")
            return
        frame = max(0, min(int(current or 0), arr.size - 1))
        self.value_label.setText(
            f"{name}: {arr[frame]:.6g}  min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g}"
        )

    def refresh_active_rmf(self) -> None:
        """Reload the active RMF file from disk."""
        object_id = self.viewer.get_active_object_id()
        if object_id is None:
            return
        store = getattr(self.parent_window, "_object_store", {})
        entry = store.get(object_id, {})
        path = entry.get("path")
        if not path:
            return

        old_frame = 0
        try:
            old_frame = int(self.viewer.get_current_frame())
        except Exception:
            old_frame = 0

        try:
            data = load_rmf_full(Path(path))
        except RmfNotAvailableError as exc:
            QtWidgets.QMessageBox.warning(self._widget, "RMF Not Available", str(exc))
            return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self._widget, "RMF Refresh Failed", f"{exc}")
            return

        try:
            self.viewer.set_rmf_data(
                hierarchy=data["hierarchy"],
                frames=data["frames"],
                radii=data["radii"],
                restraints=data.get("restraints"),
                rmf_provenance=data.get("rmf_provenance"),
                rmf_frame_series=data.get("rmf_frame_series", {}),
                rmf_frame_metadata=data.get("rmf_frame_metadata", {}),
                rmf_resolutions=data.get("rmf_resolutions", set()),
                bond_pairs=data.get("bond_pairs"),
                object_id=object_id,
            )
            state = _active_state(self.viewer)
            n_frames = getattr(getattr(state, "frames", None), "shape", (0,))[0]
            if n_frames > 0:
                self.viewer.set_current_frame(min(old_frame, n_frames - 1))
            self.set_state(state)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self._widget, "RMF Refresh Failed", f"{exc}")


def _active_state(viewer: object) -> object | None:
    """Return the active viewer state, or None if unavailable."""
    try:
        return viewer._get_active_state()
    except Exception:
        return None


def _has_numeric_values(values: object) -> bool:
    """Return True when values contain at least one finite numeric value."""
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return False
    return arr.size > 0 and bool(np.isfinite(arr).any())


__all__ = ["RmfPanel", "RmfPlotWidget"]
