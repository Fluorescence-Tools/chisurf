from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import json
import logging

import chinet as cn

from qtpy import QtCore, QtWidgets, QtGui

from .py_syntax import PythonHighlighter, CodeEditor

from .model import NodeModel, PortSpec
from .node_item import NodeGraphicsItem
from .scene import NodeScene
from .state_tracker import SceneStateTracker
from .timeline_widget import TimelineWidget
from .view import NodeView
from .ui import (
    InlineLabeledSlider,
    NumericValueWidget,
    StyledComboBox,
    TextBoxWidget,
    Vector1DWidget,
    WidgetPalette,
    PtPlotWidget,
    apply_node_ui_theme,
)
from .theme import metric as theme_metric
from .registry import registry, NodeType
from .chinet_eval import evaluate_pt_graph


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NodeEditorWidget(QtWidgets.QWidget):
    graphChanged = QtCore.Signal()

    """High-level node editor widget with an example arithmetic graph.

    Convenience wrapper around NodeScene and NodeView. Includes a side panel
    with JSON load/save for the example.
    """

    def __init__(
        self,
        parent=None,
        *,
        node_width: float = 220.0,
        node_title_height: float = 26.0,
        node_min_body_height: float = 60.0,
        node_radius: float = 8.0,
        scene_kwargs: Optional[Dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Node Editor Example")

        # Store style parameters for nodes
        # Allow theme to override core geometry so the overall node size can
        # be tuned without changing code.
        self.node_width = float(theme_metric("node_default_width", node_width))
        self.node_title_height = float(theme_metric("node_title_height", node_title_height))
        self.node_min_body_height = float(theme_metric("node_min_body_height", node_min_body_height))
        self.node_radius = float(theme_metric("node_corner_radius", node_radius))

        # Available node types that can be created from the context menu.
        # This list controls which node IDs appear in the scene context menu.
        # The actual behaviour and widgets for each type are defined by the
        # node type registry in ``registry.py``.
        self._available_node_types: List[str] = [
            "constant",
            "binary_op",
            "vector",
            "output",
            "controls",
            "text_entry",
            "text_note",
            "pt_constant",
            "pt_transform",
            "pt_output",
            "pt_plot",
        ]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        layout.addWidget(splitter)

        # Left panel: scene view + timeline navigation controls
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)

        # Scene + view
        if scene_kwargs is None:
            scene_kwargs = {}
        self.scene = NodeScene(self, **scene_kwargs)
        # Let the scene delegate node creation requests back to this widget
        self.scene.node_adder = self._on_add_node_requested
        self.view = NodeView(self.scene, self)
        left_layout.addWidget(self.view, stretch=1)

        # Right panel: widget palette + JSON tools for the example graph
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Palette of node types (widgets) loaded from widgets_palette.json
        right_layout.addWidget(QtWidgets.QLabel("Widget palette:"))
        self.widget_palette = WidgetPalette(right_panel)
        self.widget_palette.setMinimumHeight(120)
        self.widget_palette.setMaximumHeight(220)
        right_layout.addWidget(self.widget_palette)

        right_layout.addWidget(QtWidgets.QLabel("Graph JSON (example):"))

        btn_bar = QtWidgets.QHBoxLayout()
        self.btn_load_json = QtWidgets.QToolButton()
        self.btn_load_json.setText("Load JSON")
        self.btn_save_json = QtWidgets.QToolButton()
        self.btn_save_json.setText("Save JSON")
        self.btn_evaluate_chinet = QtWidgets.QToolButton()
        self.btn_evaluate_chinet.setText("Evaluate (chinet)")
        # Wire JSON panel buttons to the helper callbacks so they operate on
        # the live scene.
        self.btn_load_json.clicked.connect(self._on_load_json_clicked)
        self.btn_save_json.clicked.connect(self._on_save_json_clicked)
        self.btn_evaluate_chinet.clicked.connect(self._on_evaluate_chinet_clicked)
        btn_bar.addWidget(self.btn_load_json)
        btn_bar.addWidget(self.btn_save_json)
        btn_bar.addWidget(self.btn_evaluate_chinet)
        btn_bar.addStretch(1)
        right_layout.addLayout(btn_bar)

        self.json_edit = QtWidgets.QPlainTextEdit()
        self.json_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.json_edit.setPlaceholderText("Graph JSON will appear here...")
        right_layout.addWidget(self.json_edit, stretch=1)

        # Add panels to splitter so the user can resize scene vs JSON
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        try:
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
        except Exception:
            pass

        # Connect widget palette to create nodes at the view center
        try:
            self.widget_palette.nodeTypeActivated.connect(self._on_palette_node_type_activated)
        except Exception:
            pass

        # Undo/redo infrastructure
        self.undo_stack = QtWidgets.QUndoStack(self)
        # Centralised scene state tracker used by all graph-level undo/redo.
        self.state_tracker = SceneStateTracker(self, self.scene, self.undo_stack)

        # Timeline: separate widget placed directly below the scene view.
        # It uses its own QGraphicsScene/QGraphicsView so that the timeline
        # geometry is completely decoupled from the node scene, avoiding
        # overlay positioning issues.
        self.timeline_scene = QtWidgets.QGraphicsScene(self)
        self.timeline_view = QtWidgets.QGraphicsView(self.timeline_scene)
        self.timeline_view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.timeline_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.timeline_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.timeline_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.timeline_view.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.timeline_view.setStyleSheet("background: transparent; border: 0px;")
        self.timeline_view.setMinimumHeight(40)
        self.timeline_view.setMaximumHeight(44)

        timeline_bar = QtWidgets.QHBoxLayout()
        timeline_bar.setContentsMargins(4, 0, 4, 4)
        timeline_bar.setSpacing(2)
        timeline_bar.addStretch(1)
        timeline_bar.addWidget(self.timeline_view, stretch=0)
        timeline_bar.addStretch(1)
        left_layout.addLayout(timeline_bar)

        # Keyboard shortcuts for undo/redo: bind to the view so they work when
        # the scene has focus, and keep references to avoid premature GC.
        self._undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Undo, self.view)
        self._undo_shortcut.activated.connect(self._on_undo_shortcut)
        self._redo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Redo, self.view)
        self._redo_shortcut.activated.connect(self._on_redo_shortcut)

        # Graphics-based timeline hosted in its own scene/view below the main
        # editor view. The back/forward buttons are embedded via
        # QGraphicsProxyWidget so everything appears as one integrated control
        # strip.
        self.timeline = TimelineWidget()
        self.timeline.set_undo_stack(self.undo_stack)
        # Hide less relevant operations by default; users can enable them
        # from the context menu if desired.
        self.timeline.set_show_moves(False)
        try:
            self.timeline.set_show_folds(False)
        except Exception:
            pass
        self.btn_timeline_back = QtWidgets.QToolButton()
        self.btn_timeline_back.setText("\u25c0")
        self.btn_timeline_back.setToolTip("Step backward in history")
        self.btn_timeline_back.setAutoRaise(True)
        self.btn_timeline_back.setFixedSize(20, 20)
        self.btn_timeline_forward = QtWidgets.QToolButton()
        self.btn_timeline_forward.setText("\u25b6")
        self.btn_timeline_forward.setToolTip("Step forward in history")
        self.btn_timeline_forward.setAutoRaise(True)
        self.btn_timeline_forward.setFixedSize(20, 20)
        self.timeline.set_navigation_buttons(self.btn_timeline_back, self.btn_timeline_forward)
        # Add the graphics item to the dedicated timeline scene and size the
        # scene rect so the view can keep it centered.
        self.timeline_scene.addItem(self.timeline)
        try:
            br = self.timeline.path().boundingRect()
            self.timeline_scene.setSceneRect(br)
        except Exception:
            pass

        self.btn_timeline_back.clicked.connect(self._on_timeline_back)
        self.btn_timeline_forward.clicked.connect(self._on_timeline_forward)

        # Reactively evaluate chinet PT graphs when the graph changes.
        try:
            self.graphChanged.connect(self._on_graph_changed_reactive_chinet)
        except Exception:
            pass

        # Register example node types
        self._register_example_node_types()

        self._build_example_graph()

    def _register_example_node_types(self):
        """Register the built-in example node types."""
        # Constant node (scalar value source)
        if registry.get("constant") is None:
            registry.register(NodeType(
                id="constant",
                title="Constant",
                inputs=[],
                outputs=[PortSpec("Value", True)],
                category="Math",
                factory=self._create_constant_factory,
                default_config={"label": "Value", "value": 0.0},
            ))

        # Binary operation node
        if registry.get("binary_op") is None:
            registry.register(NodeType(
                id="binary_op",
                title="Operation",
                inputs=[PortSpec("Value 1", False), PortSpec("Value 2", False)],
                outputs=[PortSpec("Output", True)],
                category="Math",
                factory=self._create_binary_op_factory,
                default_config={"op": "Add"},
            ))

        # Output node (sink / display)
        if registry.get("output") is None:
            registry.register(NodeType(
                id="output",
                title="Output",
                inputs=[PortSpec("Value", False)],
                outputs=[],
                category="I/O",
                factory=self._create_output_factory,
                default_config={"text": "-"},
            ))

        # Controls node (UI controls only)
        if registry.get("controls") is None:
            registry.register(NodeType(
                id="controls",
                title="Controls",
                inputs=[],
                outputs=[],
                category="Control",
                factory=self._create_controls_factory,
                default_config={},
            ))

        # 1D vector node: manual numeric input with validation (no slider)
        if registry.get("vector") is None:
            registry.register(NodeType(
                id="vector",
                title="Vector",
                inputs=[],
                outputs=[PortSpec("Value", True)],
                category="Math",
                factory=self._create_vector1d_factory,
                default_config={"label": "Vector", "value": 0.0},
            ))

        # Text entry node: string value with optional hide-on-connect behaviour
        if registry.get("text_entry") is None:
            registry.register(NodeType(
                id="text_entry",
                title="Text entry",
                inputs=[PortSpec("Value", False)],
                outputs=[PortSpec("Value", True)],
                category="I/O",
                factory=self._create_text_entry_factory,
                default_config={
                    "label": "Text",
                    "text": "",
                    # When True, the embedded TextBoxWidget will switch to a
                    # label-only view as soon as the input port is wired.
                    "hide_value_when_connected": True,
                },
            ))

        # Simple text note node (label + line edit, non-evaluating)
        if registry.get("text_note") is None:
            registry.register(NodeType(
                id="text_note",
                title="Text note",
                inputs=[],
                outputs=[],
                category="Annotation",
                factory=self._create_text_note_factory,
                default_config={"label": "Note", "text": ""},
            ))

        # Chinet-related nodes for parameter graphs
        if registry.get("pt_constant") is None:
            registry.register(NodeType(
                id="pt_constant",
                title="PT Constant",
                inputs=[],
                outputs=[PortSpec("Value", True)],
                category="Chinet",
                factory=self._create_pt_constant_factory,
                default_config={"name": "param", "value": 0.0},
            ))

        if registry.get("pt_transform") is None:
            registry.register(NodeType(
                id="pt_transform",
                title="PT Transform",
                inputs=[PortSpec("A", False), PortSpec("B", False)],
                # Allow up to three outputs from the transform; these map to
                # chinet Node output ports like "out_00", "out_01", ... and
                # extra ports simply remain unused when the callback returns
                # fewer values.
                outputs=[
                    PortSpec("out_00", True),
                    PortSpec("out_01", True),
                    PortSpec("out_02", True),
                ],
                category="Chinet",
                factory=self._create_pt_transform_factory,
                default_config={
                    "code": "def f(A=0.0, B=0.0):\n    return {'out_00': A + B, 'out_01': A - B, 'out_02': A * B}",
                },
            ))

        if registry.get("pt_plot") is None:
            registry.register(NodeType(
                id="pt_plot",
                title="PT Plot",
                inputs=[PortSpec("X", False), PortSpec("Y", False)],
                outputs=[],
                category="Chinet",
                factory=self._create_pt_plot_factory,
                default_config={"title": "Damped sine"},
            ))

        if registry.get("pt_output") is None:
            registry.register(NodeType(
                id="pt_output",
                title="PT Output",
                inputs=[PortSpec("Value", False)],
                outputs=[],
                category="Chinet",
                factory=self._create_output_factory,
                default_config={"text": "-"},
            ))

    def _create_constant_factory(
        self,
        config: Dict[str, Any],
        on_value_changed=None,
        on_value_committed=None,
    ):
        """Factory for constant node content.

        ``on_value_changed`` is an optional callback invoked whenever the
        value changes (e.g. during a slider drag). ``on_value_committed`` is
        called once at the end of an edit gesture (mouse release, wheel step,
        or text edit commit).
        """
        label = str(config.get("label", "Value"))
        try:
            value = float(config.get("value", 0.0))
        except Exception:
            value = 0.0
        slider_min = float(config.get("slider_min", 0.0))
        slider_max = float(config.get("slider_max", 1.0 if value == 0 else max(value * 2.0, 1.0)))
        decimals = int(config.get("decimals", 3))
        show_fix = bool(config.get("show_fix", False))
        fixed = bool(config.get("fixed", False))

        editor = self

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")
            lay = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            lay.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            lay.setSpacing(spacing_small)
            slider = InlineLabeledSlider(label, slider_min, slider_max, value, decimals, w)

            # Optional "fix" checkbox to lock the value; hidden unless
            # config["show_fix"] is true.
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(spacing_small)
            row.addWidget(slider, 1)
            fix_cb: QtWidgets.QCheckBox | None = None
            if show_fix:
                fix_cb = QtWidgets.QCheckBox("Fix", w)
                fix_cb.setChecked(fixed)
                row.addWidget(fix_cb)
            lay.addLayout(row)
            spin = QtWidgets.QDoubleSpinBox(w)
            spin.setObjectName("value_spin")
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(decimals)
            spin.setValue(value)
            spin.setVisible(False)
            apply_node_ui_theme(w)
            themed_font = w.font()
            slider.setFont(themed_font)
            spin.setFont(themed_font)

            def sync_from_slider(v: float):
                spin.blockSignals(True)
                spin.setValue(v)
                spin.blockSignals(False)
                try:
                    config["value"] = float(v)
                except Exception:
                    pass
                if on_value_changed is not None:
                    try:
                        on_value_changed(float(v))
                    except Exception:
                        pass

            def sync_from_spin(v: float):
                slider.blockSignals(True)
                slider.setValue(v)
                slider.blockSignals(False)
                try:
                    config["value"] = float(v)
                except Exception:
                    pass
                if on_value_changed is not None:
                    try:
                        on_value_changed(float(v))
                    except Exception:
                        pass

            slider.valueChanged.connect(sync_from_slider)
            spin.valueChanged.connect(sync_from_spin)

            if show_fix and fix_cb is not None:
                def _on_fix_toggled(checked: bool) -> None:
                    config["fixed"] = bool(checked)
                    slider.setEnabled(not checked)
                    spin.setEnabled(not checked)

                fix_cb.toggled.connect(_on_fix_toggled)
                # Apply initial enabled/disabled state
                _on_fix_toggled(fix_cb.isChecked())

            # Make slider edits undoable as a single graph change per
            # interaction (drag, wheel, or text edit) by snapshotting the
            # scene before/after.
            def _on_slider_edit_start() -> None:
                editor.begin_undo("Change constant")

            def _on_slider_edit_finish(v: float) -> None:
                editor.commit_undo()
                if on_value_committed is not None:
                    try:
                        on_value_committed(float(v))
                    except Exception:
                        pass

            try:
                slider.editingStarted.connect(_on_slider_edit_start)
                slider.editingFinished.connect(_on_slider_edit_finish)
            except Exception:
                pass
            return w

        return factory

    def _create_text_entry_factory(self, config: Dict[str, Any]):
        """Factory for a text entry node using TextBoxWidget.

        This is similar to ``text_note`` but with ports and
        ``hide_value_when_connected`` semantics handled by the port items.
        """

        label = str(config.get("label", "Text"))
        text = str(config.get("text", ""))

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")

            v = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            v.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            v.setSpacing(spacing_small)

            editor = TextBoxWidget(label, text, w)
            v.addWidget(editor)

            apply_node_ui_theme(w)

            def _on_text_changed(value: str) -> None:
                config["text"] = value

            try:
                editor.valueChanged.connect(_on_text_changed)
            except Exception:
                pass

            return w

        return factory

    def _create_vector1d_factory(self, config: Dict[str, Any]):
        """Factory for a 1D vector input node (manual text input with validation)."""

        label = str(config.get("label", "Vector"))
        raw_val = config.get("value", 0.0)
        show_fix = bool(config.get("show_fix", False))
        fixed = bool(config.get("fixed", False))

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")
            lay = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            lay.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            lay.setSpacing(spacing_small)

            vec = Vector1DWidget(label, raw_val, w)

            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(spacing_small)
            row.addWidget(vec, 1)
            fix_cb: QtWidgets.QCheckBox | None = None
            if show_fix:
                fix_cb = QtWidgets.QCheckBox("Fix", w)
                fix_cb.setChecked(fixed)
                row.addWidget(fix_cb)
            lay.addLayout(row)

            def _on_value_changed(v) -> None:
                # v is expected to be a sequence[float]; store it as-is in
                # the node config so it can round-trip via JSON.
                try:
                    config["value"] = list(v)
                except Exception:
                    config["value"] = v

            vec.valueChanged.connect(_on_value_changed)

            if show_fix and fix_cb is not None:
                def _on_fix_toggled(checked: bool) -> None:
                    config["fixed"] = bool(checked)
                    vec.setEnabled(not checked)

                fix_cb.toggled.connect(_on_fix_toggled)
                _on_fix_toggled(fix_cb.isChecked())
            apply_node_ui_theme(w)
            return w

        return factory

    def _create_text_note_factory(self, config: Dict[str, Any]):
        """Factory for a simple text note node using TextBoxWidget."""

        label = str(config.get("label", "Note"))
        text = str(config.get("text", ""))

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")

            v = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            v.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            v.setSpacing(spacing_small)

            lbl = QtWidgets.QLabel(label, w)
            edit = QtWidgets.QPlainTextEdit(w)
            edit.setPlainText(text)
            edit.setTabChangesFocus(True)
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
            v.addWidget(lbl)
            v.addWidget(edit, 1)

            apply_node_ui_theme(w)
            edit.setStyleSheet(
                "background: transparent; border: 1px solid rgb(40,40,40);"
                "color: rgb(235,235,235); padding: 2px;"
            )

            def _on_text_changed() -> None:
                config["text"] = str(edit.toPlainText())

            try:
                edit.textChanged.connect(_on_text_changed)
            except Exception:
                pass

            return w

        return factory

    def _create_pt_constant_factory(self, config: Dict[str, Any]):
        """Factory for chinet-backed constant parameter node content."""

        # Reuse the Constant node's slider-based layout directly; PT Constant
        # behaves like a regular Constant but is tagged with a different
        # node_type ("pt_constant") so evaluation logic can treat it
        # specially. No separate name field is shown; the parameter name can
        # be placed in the node title instead.

        if "label" not in config:
            config["label"] = "Value"

        # Reuse Constant factory but hook chinet evaluation both while the
        # value is changing (for live updates) and once when the edit is
        # committed, so reactive chinet graphs stay in sync.
        def _on_changed(_v: float) -> None:
            try:
                self.evaluate_chinet_graph()
            except Exception:
                pass

        def _on_committed(_v: float) -> None:
            try:
                self.evaluate_chinet_graph()
            except Exception:
                pass

        return self._create_constant_factory(
            config,
            on_value_changed=_on_changed,
            on_value_committed=_on_committed,
        )

    def _create_pt_plot_factory(self, config: Dict[str, Any]):
        """Factory for a PT plot node that renders X/Y arrays via PtPlotWidget."""

        title = str(config.get("title", "Damped sine"))

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")

            v = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            v.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            v.setSpacing(spacing_small)

            plot = PtPlotWidget(title, w)
            v.addWidget(plot, 1)

            apply_node_ui_theme(w)

            return w

        return factory

    def _create_pt_transform_factory(self, config: Dict[str, Any]):
        """Factory for chinet-backed transform node content (Python callback)."""

        code = str(config.get("code", "def f(A=0.0, B=0.0):\n    return A + B"))

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")
            v = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            v.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            v.setSpacing(spacing_small)

            lbl = QtWidgets.QLabel("Function (Python):", w)
            edit = CodeEditor(w)
            edit.setPlainText(code)
            font = edit.font()
            # Use a monospaced font for better code readability
            try:
                font.setStyleHint(QtGui.QFont.Monospace)
            except Exception:
                pass
            try:
                font.setFamily("Consolas")
            except Exception:
                font.setFamily("Courier New")
            edit.setFont(font)
            edit.setFrameStyle(QtWidgets.QFrame.NoFrame)
            edit.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            edit.setAutoFillBackground(False)
            edit.setStyleSheet(
                "background: transparent; border: 1px solid rgb(40,40,40);"
                "color: rgb(235,235,235); padding: 2px;"
            )
            # Lightweight Python syntax highlighting using the local
            # PythonHighlighter. Keep a reference on the editor widget so the
            # highlighter is not garbage-collected.
            try:
                edit._syntax_highlighter = PythonHighlighter(edit.document())  # type: ignore[attr-defined]
            except Exception:
                try:
                    edit._syntax_highlighter = None  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Keep the code editor compact by default so nodes are more space
            # efficient, while still allowing the user to resize the node
            # vertically if more room is needed.
            edit.setMinimumHeight(48)
            edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

            v.addWidget(lbl)
            v.addWidget(edit)

            apply_node_ui_theme(w)
            # Re-apply the monospaced font in case the theme adjusted fonts
            # on the parent widget.
            edit.setFont(font)

            def _on_text_changed() -> None:
                config["code"] = str(edit.toPlainText())

            edit.textChanged.connect(_on_text_changed)

            return w

        return factory

    def _create_binary_op_factory(self, config: Dict[str, Any]):
        """Factory for binary operation node content."""
        default_op = str(config.get("op", "Add"))
        operations = ["Add", "Subtract", "Multiply", "Divide"]

        editor = self

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")
            lay = QtWidgets.QHBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_margin_v_top", 2))
            m_v_bottom = int(theme_metric("node_layout_margin_v_bottom", 2))
            spacing_default = int(theme_metric("node_layout_spacing_default", 4))
            lay.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            lay.setSpacing(spacing_default)
            lab = QtWidgets.QLabel("Op:")
            combo = StyledComboBox()
            combo.addItems(operations)
            idx = combo.findText(default_op)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            lay.addWidget(lab)
            lay.addWidget(combo, 1)
            apply_node_ui_theme(w)

            # Treat a combobox selection change as a single undoable
            # operation by snapshotting the scene before/after the popup.
            def _on_combo_edit_start() -> None:
                editor.begin_undo("Change operation")

            def _on_combo_edit_finish(_text: str) -> None:
                editor.commit_undo()

            try:
                combo.editingStarted.connect(_on_combo_edit_start)
                combo.editingFinished.connect(_on_combo_edit_finish)
            except Exception:
                pass

            return w

        return factory

    def _create_output_factory(self, config: Dict[str, Any]):
        """Factory for output node content."""
        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")
            lay = QtWidgets.QHBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_v_top_output", 2))
            m_v_bottom = int(theme_metric("node_layout_v_bottom_output", 2))
            spacing_default = int(theme_metric("node_layout_spacing_default", 4))
            lay.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            lay.setSpacing(spacing_default)
            label = QtWidgets.QLabel("Result")
            value_label = QtWidgets.QLabel(str(config.get("text", "-")))
            value_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            lay.addWidget(label)
            lay.addWidget(value_label)
            apply_node_ui_theme(w)
            return w

        return factory

    def _create_controls_factory(self, config: Dict[str, Any]):
        """Factory for controls node content."""
        editor = self

        def factory() -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            w.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            w.setAutoFillBackground(False)
            w.setStyleSheet("background: transparent;")
            v = QtWidgets.QVBoxLayout(w)
            m_h = int(theme_metric("node_layout_margin_h", 4))
            m_v_top = int(theme_metric("node_layout_v_top_controls", 2))
            m_v_bottom = int(theme_metric("node_layout_v_bottom_controls", 2))
            spacing_small = int(theme_metric("node_layout_spacing_small", 2))
            spacing_default = int(theme_metric("node_layout_spacing_default", 4))
            v.setContentsMargins(m_h, m_v_top, m_h, m_v_bottom)
            v.setSpacing(spacing_small)
            r1 = QtWidgets.QHBoxLayout()
            r1.setSpacing(spacing_default)
            r1.addWidget(QtWidgets.QLabel("Mode:"))
            cb = StyledComboBox()
            cb.addItems(["A", "B", "C"])
            r1.addWidget(cb, 1)
            v.addLayout(r1)
            r2 = QtWidgets.QHBoxLayout()
            r2.setSpacing(spacing_default)
            btn_apply = QtWidgets.QPushButton("Apply")
            btn_reset = QtWidgets.QPushButton("Reset")
            btn_apply.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn_reset.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            r2.addWidget(btn_apply)
            r2.addWidget(btn_reset)
            v.addLayout(r2)
            v.addWidget(QtWidgets.QCheckBox("Enable option"))
            v.addWidget(QtWidgets.QRadioButton("Choice 1"))
            v.addWidget(QtWidgets.QRadioButton("Choice 2"))

            # Simple numeric control (demonstrates NumericValueWidget)
            try:
                current = float(config.get("control_value", 1.0))
            except Exception:
                current = 1.0
            num = NumericValueWidget("Control value", current, parent=w)
            v.addWidget(num)

            def _on_value_changed(val: float) -> None:
                try:
                    config["control_value"] = float(val)
                except Exception:
                    pass

            def _on_editing_changed(active: bool) -> None:
                # Treat a drag/edit gesture as a single undoable change.
                if active:
                    editor.begin_undo("Change control value")
                else:
                    editor.commit_undo()

            try:
                num.valueChanged.connect(_on_value_changed)
                num.editingChanged.connect(_on_editing_changed)
            except Exception:
                pass
            apply_node_ui_theme(w)
            return w

        return factory

    # ----- Configuration --------------------------------------------------
    @property
    def available_node_types(self) -> List[str]:
        """List of node type identifiers that can be added from the scene context menu.

        This is read by ``NodeScene`` when building the "Add node" submenu. The
        identifiers are the same strings used by ``_create_content_factory_for_json_node``
        and the example JSON graph (e.g. ``"constant"``, ``"binary_op"``).
        """

        return list(self._available_node_types)

    @available_node_types.setter
    def available_node_types(self, types: Sequence[str] | Iterable[str]) -> None:
        self._available_node_types = [str(t) for t in types]

    def _create_content_factory_for_json_node(self, node_type: str, config: Dict[str, Any]):
        """Return a content factory callable for a JSON-loaded node.

        This keeps editor-specific behaviour (like chinet evaluation hooks)
        bound to the correct NodeEditorWidget instance for each scene.
        """

        if node_type == "constant":
            return self._create_constant_factory(config)
        if node_type == "binary_op":
            return self._create_binary_op_factory(config)
        if node_type == "vector":
            return self._create_vector1d_factory(config)
        if node_type == "output":
            return self._create_output_factory(config)
        if node_type == "controls":
            return self._create_controls_factory(config)
        if node_type == "text_entry":
            return self._create_text_entry_factory(config)
        if node_type == "text_note":
            return self._create_text_note_factory(config)
        if node_type == "pt_constant":
            return self._create_pt_constant_factory(config)
        if node_type == "pt_transform":
            return self._create_pt_transform_factory(config)
        if node_type == "pt_output":
            return self._create_output_factory(config)
        if node_type == "pt_plot":
            return self._create_pt_plot_factory(config)
        return None

    # ----- Convenience helpers -------------------------------------------
    def create_node_item(self, model: NodeModel) -> NodeGraphicsItem:
        """Create a styled NodeGraphicsItem for a given model."""
        t = getattr(model, "node_type", None)
        type_key = str(t) if t is not None else ""
        type_key = type_key.replace(" ", "_")

        width = self.node_width
        title_h = self.node_title_height
        min_body = self.node_min_body_height

        if type_key:
            try:
                width = float(theme_metric(f"node_width_{type_key}", width))
                title_h = float(theme_metric(f"node_title_height_{type_key}", title_h))
                min_body = float(theme_metric(f"node_min_body_height_{type_key}", min_body))
            except Exception:
                pass

        return NodeGraphicsItem(
            model,
            width=width,
            title_height=title_h,
            min_body_height=min_body,
            radius=self.node_radius,
        )

    def clear_graph(self):
        self.scene.clear()
        self.scene.edges = []

    def is_directed_acyclic(self) -> Optional[bool]:
        return self.scene.is_directed_acyclic()

    # ----- chinet evaluation helpers -------------------------------------

    def evaluate_chinet_graph(self) -> None:
        """Evaluate the current graph using chinet for PT nodes.

        This operates on nodes of types ``pt_constant``, ``pt_transform`` and
        ``pt_output`` without affecting the existing example nodes.
        """

        values = evaluate_pt_graph(self.scene)
        if values is None:
            return

        # Store last evaluation result for potential callers/debugging
        try:
            self._last_chinet_values = values
        except Exception:
            pass

    def _on_graph_changed_reactive_chinet(self) -> None:
        """Slot used to keep PT examples reactive.

        This runs a lightweight chinet evaluation whenever the graph changes
        (after undo/redo or widget interactions that commit an undo step).
        Non-PT graphs are ignored cheaply.
        """

        # Fast check: only bother when there is at least one PT node.
        try:
            from .node_item import NodeGraphicsItem as _NGI  # local import to avoid cycles

            has_pt = False
            sc = self.scene
            for it in sc.items():
                if not isinstance(it, _NGI):
                    continue
                t = getattr(getattr(it, "model", None), "node_type", "")
                if t in ("pt_constant", "pt_transform", "pt_output"):
                    has_pt = True
                    break
            if not has_pt:
                return
        except Exception:
            # Best-effort only; if detection fails, fall back to manual button.
            return

        try:
            self.evaluate_chinet_graph()
        except Exception:
            # Errors are already logged inside evaluate_chinet_graph / evaluate_pt_graph
            pass

    # ----- Node creation from scene --------------------------------------
    def _default_ports_for_type(self, node_type: str) -> tuple[List[PortSpec], List[PortSpec]]:
        """Return default input/output PortSpec lists for a node type.

        These defaults mirror the example JSON graph and keep the typing
        decisions local to this widget, so the scene stays presentation-only.
        """

        if node_type == "constant":
            return [], [PortSpec(name="Value", is_output=True)]
        if node_type == "binary_op":
            inputs = [
                PortSpec(name="Value 1", is_output=False),
                PortSpec(name="Value 2", is_output=False),
            ]
            outputs = [PortSpec(name="Output", is_output=True)]
            return inputs, outputs
        if node_type == "vector":
            return [], [PortSpec(name="Value", is_output=True)]
        if node_type == "output":
            return [PortSpec(name="Value", is_output=False)], []
        if node_type == "controls":
            return [], []
        # Fallback: generic node with no ports
        return [], []

    def _default_title_for_type(self, node_type: str) -> str:
        if node_type == "constant":
            return "Constant"
        if node_type == "binary_op":
            return "Operation"
        if node_type == "vector":
            return "Vector"
        if node_type == "output":
            return "Output"
        if node_type == "controls":
            return "Controls"
        return str(node_type).title() or "Node"

    def _on_add_node_requested(self, node_type: str, scene_pos: QtCore.QPointF) -> None:
        """Callback used by ``NodeScene`` when the user chooses "Add node"."""
        node_type_obj = registry.get(node_type)
        if node_type_obj is None:
            return

        cfg = dict(node_type_obj.default_config)  # Copy
        factory = None
        try:
            factory = self._create_content_factory_for_json_node(node_type, cfg)
        except Exception:
            factory = None
        if factory is None and node_type_obj.factory:
            try:
                factory = node_type_obj.factory(cfg)
            except Exception:
                factory = None
        logger.debug("on_add_node_requested type=%s pos=(%.2f, %.2f)", node_type, scene_pos.x(), scene_pos.y())
        self.begin_undo(f"Add {node_type_obj.title}")
        try:
            model = NodeModel(
                title=node_type_obj.title,
                inputs=node_type_obj.inputs,
                outputs=node_type_obj.outputs,
                node_type=node_type,
                config=cfg,
                content_factory=factory,
            )

            item = self.create_node_item(model)
            self.scene.addItem(item)
            item.setPos(scene_pos)
        finally:
            self.commit_undo()

    def _on_palette_node_type_activated(self, node_type: str) -> None:
        """Create a node of the given type near the view center.

        Used by the WidgetPalette selection widget on the right-hand side.
        """

        try:
            view = self.view
        except Exception:
            view = None
        if view is not None:
            rect = view.viewport().rect()
            scene_pos = view.mapToScene(rect.center())
        else:
            scene_pos = QtCore.QPointF(0.0, 0.0)
        self._on_add_node_requested(str(node_type), scene_pos)

    # ----- JSON IO -------------------------------------------------------
    def load_graph_from_json(self, json_str: str):
        """Load graph from JSON string using the scene's from_dict method."""
        self.clear_graph()
        try:
            data = json.loads(json_str)
        except Exception as exc:
            logger.error("Error parsing JSON graph: %s", exc)
            return
        try:
            self.scene.from_dict(data)
        except Exception as exc:
            logger.error("Error loading graph from data: %s", exc)
            return
        # Keep PT examples reactive by running an initial evaluation after load.
        try:
            self.evaluate_chinet_graph()
        except Exception:
            pass

        # Restore view state (center/zoom) if present. This is stored in an
        # optional top-level ``"_view"`` block so older JSON without it
        # continues to work and the validation logic ignores it.
        try:
            view_state = data.get("_view", {})
            if isinstance(view_state, dict) and self.view is not None:
                center = view_state.get("center")
                scale = view_state.get("scale")
                if (
                    isinstance(center, (list, tuple))
                    and len(center) == 2
                ):
                    try:
                        cx = float(center[0])
                        cy = float(center[1])
                        self.view.centerOn(cx, cy)
                    except Exception:
                        pass
                if isinstance(scale, (int, float)) and scale > 0:
                    try:
                        self.view.resetTransform()
                        self.view.scale(float(scale), float(scale))
                    except Exception:
                        pass
        except Exception:
            # View state is best-effort only; never fail the load because of it.
            pass

    def to_json(self) -> str:
        """Serialize current scene to JSON string."""
        data = self.scene.to_dict()

        # Persist the current view center & zoom in an optional ``"_view"``
        # block so that reloading via NodeEditorWidget restores the camera
        # position without affecting the validated graph schema.
        try:
            if self.view is not None:
                rect = self.view.viewport().rect()
                center_scene = self.view.mapToScene(rect.center())
                transform = self.view.transform()
                scale_x = float(transform.m11())
                data["_view"] = {
                    "center": [float(center_scene.x()), float(center_scene.y())],
                    "scale": scale_x,
                }
        except Exception:
            pass

        return json.dumps(data, indent=2)

    def load_graph_from_file(self, filepath: str):
        """Load graph from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_str = f.read()
            self.load_graph_from_json(json_str)
        except Exception as exc:
            logger.error("Error loading graph from file '%s': %s", filepath, exc)

    def save_graph_to_file(self, filepath: str):
        """Save current graph to a JSON file."""
        try:
            json_str = self.to_json()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        except Exception as exc:
            logger.error("Error saving graph to file '%s': %s", filepath, exc)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        try:
            timeline = getattr(self, "timeline", None)
            if timeline is not None:
                # Detach the timeline from the undo stack so no further
                # signals are delivered while objects are being destroyed.
                if hasattr(timeline, "set_undo_stack"):
                    try:
                        timeline.set_undo_stack(None)
                    except Exception:
                        pass

                # Also remove the graphics item from its dedicated scene so Qt
                # will not attempt to paint it after the Python wrapper is
                # gone.
                try:
                    scene = getattr(self, "timeline_scene", None)
                    if scene is not None:
                        scene.removeItem(timeline)
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(event)

    # ----- UI callbacks for example JSON panel ---------------------------
    def _on_load_json_clicked(self):
        text = self.json_edit.toPlainText()
        if not text.strip():
            return
        self.load_graph_from_json(text)

    def _on_save_json_clicked(self):
        text = self.to_json()
        self.json_edit.setPlainText(text)

    def _on_evaluate_chinet_clicked(self):
        """Slot for the "Evaluate (chinet)" tool button.

        This delegates to ``evaluate_chinet_graph`` which operates only on
        the PT-specific node types and leaves the original example nodes
        untouched. Any errors are logged via the module logger.
        """

        try:
            self.evaluate_chinet_graph()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Error during chinet evaluation from UI: %s", exc)

    # ----- Help dialog ----------------------------------------------------
    def show_help_dialog(self):
        """Show a simple help dialog that renders the README.md content.

        Returns the created QDialog instance so callers/tests can inspect or
        close it programmatically.
        """

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Node Editor Help")
        dlg.setModal(False)

        layout = QtWidgets.QVBoxLayout(dlg)
        text_edit = QtWidgets.QTextEdit(dlg)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        # Load README.md from the same folder as this module
        content = ""
        try:
            import pathlib

            readme_path = pathlib.Path(__file__).with_name("README.md")
            content = readme_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - very unlikely
            content = f"Unable to load help content (README.md):\n{exc}"

        text_edit.setPlainText(content)
        dlg.resize(800, 600)
        dlg.show()
        return dlg

    def _build_example_graph(self):
        try:
            import pathlib

            root = pathlib.Path(__file__).resolve().parent
            example_path = root.joinpath("examples", "example_graph.json")
            example_json = example_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to load example graph JSON: %s", exc)
            return

        self.load_graph_from_json(example_json)
        try:
            self.json_edit.setPlainText(example_json.strip())
        except Exception:
            pass

    # ----- Undo/redo helpers -----------------------------------------------
    def _on_undo_shortcut(self) -> None:
        logger.debug("undo shortcut activated")
        self.undo()

    def _on_redo_shortcut(self) -> None:
        logger.debug("redo shortcut activated")
        self.redo()

    def _on_timeline_back(self) -> None:
        """Jump to the previous *visible* history entry in the timeline."""

        if self.timeline is None or self.undo_stack is None:
            return
        current = self.undo_stack.index()
        prev_idx = self.timeline.previous_visible_index(current)
        if prev_idx is None:
            return
        while self.undo_stack.index() > prev_idx and self.undo_stack.canUndo():
            self.undo_stack.undo()

    def _on_timeline_forward(self) -> None:
        """Jump to the next *visible* history entry in the timeline."""

        if self.timeline is None or self.undo_stack is None:
            return
        current = self.undo_stack.index()
        next_idx = self.timeline.next_visible_index(current)
        if next_idx is None:
            return
        while self.undo_stack.index() < next_idx and self.undo_stack.canRedo():
            self.undo_stack.redo()
    def begin_undo(self, text: str) -> None:
        """Mark the beginning of an undoable graph change."""
        self.state_tracker.begin_action(text)

    def commit_undo(self) -> None:
        """Commit the pending undo snapshot as a QUndoCommand, if changed."""
        self.state_tracker.commit_action()
        try:
            self.graphChanged.emit()
        except Exception:
            pass

    def undo(self) -> None:
        """Undo last graph change, if any."""

        try:
            logger.debug(
                "undo invoked: index=%d can_undo=%s", self.undo_stack.index(), self.undo_stack.canUndo()
            )
            self.undo_stack.undo()
            try:
                self.graphChanged.emit()
            except Exception:
                pass
        except Exception as exc:
            logger.error("Error during undo: %s", exc)

    def redo(self) -> None:
        """Redo last undone graph change, if any."""

        try:
            logger.debug(
                "redo invoked: index=%d can_redo=%s", self.undo_stack.index(), self.undo_stack.canRedo()
            )
            self.undo_stack.redo()
            try:
                self.graphChanged.emit()
            except Exception:
                pass
        except Exception as exc:
            logger.error("Error during redo: %s", exc)


__all__ = ["NodeEditorWidget"]
