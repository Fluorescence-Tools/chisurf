from __future__ import annotations

from typing import Optional
from qtpy import QtCore, QtWidgets, QtGui

class TimelineDock(QtCore.QObject):
    """Timeline slider and transport controls for movie playback."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        viewer: "MolView",
        cmd: "Cmd",
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.cmd = cmd

        self._dock = QtWidgets.QDockWidget("Timeline", parent)
        self._dock.setObjectName("ChimolTimelineDock")
        self._dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
        self._dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)

        container = QtWidgets.QWidget(parent)
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(spacing)
        container.setFixedHeight(40)

        # Transport controls
        self.btn_stop = QtWidgets.QToolButton(container)
        self.btn_stop.setText("■")
        self.btn_stop.setToolTip("Stop (Jump to first frame)")
        self.btn_stop.clicked.connect(lambda: self.cmd.do("mstop"))

        self.btn_play = QtWidgets.QToolButton(container)
        self.btn_play.setText("▶")
        self.btn_play.setToolTip("Play")
        self.btn_play.clicked.connect(lambda: self.cmd.do("mplay"))

        self.btn_pause = QtWidgets.QToolButton(container)
        self.btn_pause.setText("‖")
        self.btn_pause.setToolTip("Pause")
        self.btn_pause.clicked.connect(lambda: self.cmd.do("mpause"))

        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, container)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.setValue(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self._on_slider_changed)

        # Labels
        self.lbl_frame = QtWidgets.QLabel("1 / 1", container)
        self.lbl_frame.setMinimumWidth(60)
        self.lbl_frame.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.lbl_frame)

        self._dock.setWidget(container)

        # Periodic update from viewer state
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self.refresh_ui)
        self._update_timer.start(100) # 10Hz UI refresh

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget:
        return self._dock

    def _on_slider_changed(self, value: int) -> None:
        if not self.viewer._animation_running:
             self.cmd.do(f"frame {value}")

    def refresh_ui(self) -> None:
        """Sync slider and label with viewer state."""
        curr = self.viewer.get_current_frame() + 1
        total = self.viewer.get_total_frames()

        if self.slider.maximum() != total:
            self.slider.setMaximum(total)
            self.slider.setTickInterval(max(1, total // 10))
        
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(curr)
            self.slider.blockSignals(False)

        self.lbl_frame.setText(f"{curr} / {total}")
