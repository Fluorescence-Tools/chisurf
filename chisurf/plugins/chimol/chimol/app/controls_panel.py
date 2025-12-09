"""Toolbar/dock controls for the Chimol viewer."""

from __future__ import annotations

from typing import Any, Optional

from qtpy import QtWidgets, QtCore


class ControlsDock(QtCore.QObject):
    """Toolbar controls embedded inside a dock widget."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
        button_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        super().__init__(parent)
        self._dock = QtWidgets.QDockWidget("Controls", parent)
        self._dock.setObjectName("ChimolControlsDock")
        self._dock.setAllowedAreas(
            QtCore.Qt.TopDockWidgetArea | QtCore.Qt.BottomDockWidgetArea
        )
        overrides = dict(button_overrides or {})

        def _button_cfg(key: str) -> dict[str, Any]:
            cfg = overrides.get(key)
            return cfg if isinstance(cfg, dict) else {}

        def _make_button(
            key: str,
            *,
            text: str,
            tool_tip: str,
            checkable: bool = False,
            checked: bool = False,
            object_name: Optional[str] = None,
        ) -> QtWidgets.QToolButton:
            button = QtWidgets.QToolButton(parent)
            cfg = _button_cfg(key)

            btn_text = cfg.get("text", text)
            if btn_text is not None:
                button.setText(str(btn_text))

            tip = cfg.get("tool_tip", tool_tip)
            if tip is not None:
                button.setToolTip(str(tip))

            is_checkable = cfg.get("checkable", checkable)
            button.setCheckable(bool(is_checkable))
            if button.isCheckable():
                button.setChecked(bool(cfg.get("checked", checked)))
            else:
                button.setChecked(False)

            obj_name = cfg.get("object_name", object_name)
            if obj_name:
                button.setObjectName(str(obj_name))

            setters = cfg.get("setters")
            if isinstance(setters, dict):
                for method_name, value in setters.items():
                    method = getattr(button, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        if isinstance(value, (list, tuple)):
                            method(*value)
                        else:
                            method(value)
                    except Exception:
                        continue

            return button

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)

        self.button_open = _make_button(
            "open",
            text="Open",
            tool_tip="Open structure file",
        )
        toolbar.addWidget(self.button_open)

        display_group = QtWidgets.QHBoxLayout()
        display_group.setSpacing(2)

        self.button_plane = _make_button(
            "plane",
            text="Plane",
            tool_tip="Toggle reference plane",
            checkable=True,
            checked=False,
        )
        self.button_surface = _make_button(
            "surface",
            text="Surf",
            tool_tip="Toggle surface representation",
            checkable=True,
            checked=False,
        )
        self.button_display_cfg = _make_button(
            "display_cfg",
            text="Cfg",
            tool_tip="Open Chimol display configuration JSON in the system editor",
        )
        display_group.addWidget(self.button_plane)
        display_group.addWidget(self.button_surface)
        display_group.addWidget(self.button_display_cfg)

        color_group = QtWidgets.QHBoxLayout()
        color_group.setSpacing(2)
        self.button_color = _make_button(
            "color",
            text="Color AA",
            tool_tip="Color amino acids by residue type",
            checkable=True,
            checked=False,
        )
        self.button_color_ss = _make_button(
            "color_ss",
            text="Color SS",
            tool_tip="Color 3D geometry by secondary structure (helix/strand/coil)",
            checkable=True,
            checked=False,
        )
        self.button_color_sequence = _make_button(
            "color_seq",
            text="Color Seq",
            tool_tip="Color atoms by sequence position (gradient)",
            checkable=True,
            checked=False,
        )
        color_group.addWidget(self.button_color)
        color_group.addWidget(self.button_color_ss)
        color_group.addWidget(self.button_color_sequence)

        rep_group = QtWidgets.QHBoxLayout()
        rep_group.setSpacing(2)
        self.button_rep_cartoon = _make_button(
            "rep_cartoon",
            text="Cartoon",
            tool_tip="Toggle cartoon ribbon (CA tube)",
            checkable=True,
            checked=True,
            object_name="chimolRepCartoon",
        )
        self.button_rep_atoms = _make_button(
            "rep_atoms",
            text="Atoms",
            tool_tip="Toggle atoms/balls representation",
            checkable=True,
            checked=False,
            object_name="chimolRepAtoms",
        )
        self.button_rep_sticks = _make_button(
            "rep_sticks",
            text="Sticks",
            tool_tip="Toggle sticks (bond) representation",
            checkable=True,
            checked=False,
            object_name="chimolRepSticks",
        )
        self.button_rep_trace = _make_button(
            "rep_trace",
            text="Trace",
            tool_tip="Toggle CA trace line",
            checkable=True,
            checked=False,
            object_name="chimolRepTrace",
        )
        self.button_rep_dots = _make_button(
            "rep_dots",
            text="Dots",
            tool_tip="Toggle fast dot cloud",
            checkable=True,
            checked=False,
            object_name="chimolRepDots",
        )
        rep_group.addWidget(self.button_rep_cartoon)
        rep_group.addWidget(self.button_rep_atoms)
        rep_group.addWidget(self.button_rep_sticks)
        rep_group.addWidget(self.button_rep_trace)
        rep_group.addWidget(self.button_rep_dots)

        self.button_info = _make_button(
            "info",
            text="Info",
            tool_tip="Toggle system info panel",
            checkable=True,
            checked=True,
        )

        toolbar.addLayout(display_group)
        toolbar.addSpacing(8)
        toolbar.addLayout(color_group)
        toolbar.addSpacing(8)
        toolbar.addLayout(rep_group)
        toolbar.addWidget(self.button_info)
        toolbar.addStretch(1)

        controls_widget = QtWidgets.QWidget(parent)
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(*margins)
        controls_layout.setSpacing(spacing)
        controls_layout.addLayout(toolbar)
        controls_layout.addStretch(1)

        self._dock.setWidget(controls_widget)

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget:
        return self._dock
