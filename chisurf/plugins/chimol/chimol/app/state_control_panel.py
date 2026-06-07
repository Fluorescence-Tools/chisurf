from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any
from qtpy import QtCore, QtWidgets, QtGui

if TYPE_CHECKING:
    from ..renderer.view import MolView

class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.clicked.emit()
        super().mousePressEvent(ev)

class StateControlDock(QtCore.QObject):
    """PyMOL-inspired control panel for mouse modes, selection info, and playback."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        viewer: MolView,
        cmd: Any,
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.cmd = cmd
        self._mouse_mode_idx = 0
        self._mouse_modes = ["3-Button Viewing", "2-Button Viewing", "Editing"]

        self._dock = QtWidgets.QDockWidget("State", parent)
        self._dock.setObjectName("ChimolStateControlDock")
        self._dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self._dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)

        container = QtWidgets.QWidget(parent)
        container.setStyleSheet("background-color: #111; color: #eee; font-family: monospace;")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # 1. Mouse mode display (Interactive)
        self.lbl_mouse = ClickableLabel(container)
        self.lbl_mouse.setTextFormat(QtCore.Qt.RichText)
        self.lbl_mouse.setCursor(QtCore.Qt.PointingHandCursor)
        self.lbl_mouse.clicked.connect(self._toggle_mouse_mode)
        layout.addWidget(self.lbl_mouse)

        # 2. Selection mode display
        self.lbl_selection = QtWidgets.QLabel(container)
        self.lbl_selection.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.lbl_selection)

        # 3. State display
        self.lbl_state = QtWidgets.QLabel(container)
        self.lbl_state.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(self.lbl_state)
        
        layout.addSpacing(4)

        # 4. Action buttons (Zoom, Orient)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(2)
        
        self.btn_zoom = QtWidgets.QToolButton(container)
        self.btn_zoom.setText("Zoom")
        self.btn_zoom.setStyleSheet("padding: 2px 8px; background-color: #333; border: 1px solid #555;")
        self.btn_zoom.clicked.connect(lambda: self.cmd.do("zoom"))
        
        self.btn_orient = QtWidgets.QToolButton(container)
        self.btn_orient.setText("Orient")
        self.btn_orient.setStyleSheet("padding: 2px 8px; background-color: #333; border: 1px solid #555;")
        self.btn_orient.clicked.connect(lambda: self.cmd.do("orient"))
        
        btn_layout.addWidget(self.btn_zoom)
        btn_layout.addWidget(self.btn_orient)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        # 5. Playback controls (Icon-like text)
        play_layout = QtWidgets.QHBoxLayout()
        play_layout.setSpacing(1)
        
        def _make_play_btn(text: str, cmd_str: str, tip: str):
            btn = QtWidgets.QToolButton(container)
            btn.setText(text)
            btn.setToolTip(tip)
            btn.setFixedWidth(24)
            btn.setStyleSheet("background-color: #222; border: 1px solid #444; color: #f88;")
            btn.clicked.connect(lambda: self.cmd.do(cmd_str))
            return btn

        play_layout.addWidget(_make_play_btn("|<", "frame 1", "Go to first frame"))
        play_layout.addWidget(_make_play_btn("<", "frame -1", "Previous frame"))
        play_layout.addWidget(_make_play_btn(" ■ ", "mstop", "Stop"))
        play_layout.addWidget(_make_play_btn(" ► ", "mplay", "Play"))
        play_layout.addWidget(_make_play_btn(" > ", "frame +1", "Next frame"))
        play_layout.addWidget(_make_play_btn(">|", "frame last", "Go to last frame"))
        
        layout.addLayout(play_layout)
        layout.addStretch(1)

        self._dock.setWidget(container)
        self._refresh_mouse_text()

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.timeout.connect(self.refresh_ui)
        self._update_timer.start(100)

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget:
        return self._dock

    def _toggle_mouse_mode(self) -> None:
        self._mouse_mode_idx = (self._mouse_mode_idx + 1) % len(self._mouse_modes)
        self._refresh_mouse_text()

    def _refresh_mouse_text(self) -> None:
        self.lbl_mouse.setText(self._get_mouse_html())

    def _get_mouse_html(self) -> str:
        mode_str = self._mouse_modes[self._mouse_mode_idx]
        
        html = f"""
        <div style='line-height: 1.2;'>
        <span style='color: #00FF00;'>Mouse Mode {mode_str}</span><br>
        <span style='color: #FF8888;'>&nbsp;Buttons&nbsp;&nbsp;&nbsp;L&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;M&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;R&nbsp;&nbsp;&nbsp;Wheel</span><br>
        """
        
        if "Editing" in mode_str:
            html += """
            <span style='color: #FF8888;'>&nbsp;&nbsp;& Keys</span>&nbsp;<span style='color: #FFF;'>RotA&nbsp;&nbsp;MvAt&nbsp;&nbsp;MvZ&nbsp;&nbsp;Slab</span><br>
            <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;Shft</span>&nbsp;&nbsp;<span style='color: #FFF;'>RotO&nbsp;&nbsp;MvFr&nbsp;&nbsp;Clip&nbsp;MovS</span><br>
            <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;Ctrl</span>&nbsp;&nbsp;<span style='color: #FFF;'>RotT&nbsp;&nbsp;PkAt&nbsp;&nbsp;Pk1&nbsp;&nbsp;MvSZ</span><br>
            """
        elif "2-Button" in mode_str:
             html += """
            <span style='color: #FF8888;'>&nbsp;&nbsp;& Keys</span>&nbsp;<span style='color: #FFF;'>Rota&nbsp;&nbsp;Move&nbsp;&nbsp;Zoom&nbsp;Slab</span><br>
            <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;Shft</span>&nbsp;&nbsp;<span style='color: #FFF;'>+Box&nbsp;&nbsp;-Box&nbsp;&nbsp;Clip&nbsp;MovS</span><br>
            <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;Ctrl</span>&nbsp;&nbsp;<span style='color: #FFF;'>Move&nbsp;&nbsp;PkAt&nbsp;&nbsp;Pk1&nbsp;&nbsp;MvSZ</span><br>
            """
        else: # 3-Button
            html += """
            <span style='color: #FF8888;'>&nbsp;&nbsp;& Keys</span>&nbsp;<span style='color: #FFF;'>Rota&nbsp;&nbsp;Move&nbsp;&nbsp;MovZ&nbsp;Slab</span><br>
            <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;Shft</span>&nbsp;&nbsp;<span style='color: #FFF;'>+Box&nbsp;&nbsp;-Box&nbsp;&nbsp;Clip&nbsp;MovS</span><br>
            <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;Ctrl</span>&nbsp;&nbsp;<span style='color: #FFF;'>Move&nbsp;&nbsp;PkAt&nbsp;&nbsp;Pk1&nbsp;&nbsp;MvSZ</span><br>
            """
            
        html += """
        <span style='color: #8888FF;'>&nbsp;&nbsp;&nbsp;CtSh</span>&nbsp;&nbsp;<span style='color: #FFF;'>Sele&nbsp;&nbsp;Orig&nbsp;&nbsp;Clip&nbsp;MovZ</span><br>
        <span style='color: #8888FF;'>SnglClk</span>&nbsp;&nbsp;<span style='color: #FFF;'>+/-&nbsp;&nbsp;&nbsp;Cent&nbsp;&nbsp;Menu</span><br>
        <span style='color: #8888FF;'>DblClk</span>&nbsp;&nbsp;<span style='color: #FFF;'>Menu&nbsp;&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;PkAt</span>
        </div>
        """
        return html

    def refresh_ui(self) -> None:
        curr = self.viewer.get_current_frame() + 1
        total = self.viewer.get_total_frames()
        
        # Selection info
        sel_mode = getattr(self.viewer, "selection_mode", "Residues")
        self.lbl_selection.setText(f"<span style='color: #00FF00;'>Selecting</span> <span style='color: #00FFFF;'>{sel_mode}</span>")
        
        self.lbl_state.setText(f"<span style='color: #00FF00;'>State</span>&nbsp;&nbsp;&nbsp;&nbsp;<span style='color: #FFF;'>{curr} / {total}</span>")

def _install_dock(window: QtWidgets.QMainWindow, viewer: MolView, cmd: Any, margins: tuple, spacing: int):
    # This will be called from MolViewPluginWindow.__init__
    sd = StateControlDock(window, viewer, cmd, margins=margins, spacing=spacing)
    window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, sd.dock_widget)
    return sd
