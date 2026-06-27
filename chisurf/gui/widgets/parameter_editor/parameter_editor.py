"""Qt-native settings editor — a filterable ``Parameter | Value`` tree.

``ParameterEditor`` edits an arbitrary (possibly nested) settings ``dict``,
optionally backed by a JSON file. Its data layer is a
:class:`chisurf.core.dataspec.SettingsView` (the AutoForm-compatible declaration
of the dict), while its presentation is a two-column ``QTreeWidget`` that mirrors
the familiar parameter-tree look: nested dicts are expandable groups, scalar
leaves get a typed inline editor, every editable row has a reset-to-default
button, and a filter box at the top hides non-matching parameters. The backing
dict is mutated in place, so non-editable values (callables, lists, colours)
round-trip untouched.
"""

from __future__ import annotations

import json
import pathlib

from qtpy import QtWidgets

import chisurf.core.fio as io
import chisurf.core.settings
import chisurf.gui.widgets
from chisurf import typing
from chisurf.core.dataspec import ChoiceSection, PanelSection, SettingsView, ToggleSection, ValueSection
from chisurf.gui.widgets.fitting.scientific_spinbox import ScientificDoubleSpinBox

__all__ = ["ParameterEditor"]


def _normalize_none(obj):
    """Recursively map the string ``"None"`` to Python ``None``.

    The editor stores/loads JSON where ``null`` may round-trip as the literal
    string ``"None"`` (and some settings author it that way). Consumers expect
    real ``None`` — e.g. pyqtgraph reads ``symbol=None`` as "no symbol" but
    raises on ``symbol="None"``. Dicts and lists are copied; every other value
    (numbers, bools, callables, …) is returned by reference.
    """
    if isinstance(obj, str):
        return None if obj == "None" else obj
    if isinstance(obj, dict):
        return {k: _normalize_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_none(v) for v in obj]
    return obj


class ParameterEditor(QtWidgets.QWidget):
    """Edit a settings ``dict`` (optionally a JSON file) as a filterable tree.

    Parameters
    ----------
    target : dict, optional
        The settings dict to edit. Defaults to the global ChiSurf settings.
    json_file : pathlib.Path or str, optional
        File the settings are loaded from / saved to. When it points at an
        existing file a ``Save`` button is shown.
    windows_title : str, optional
        Window title.
    callback : callable, optional
        Invoked with no arguments whenever a value changes.
    """

    def __init__(
        self,
        target: typing.Dict = None,
        json_file: pathlib.Path = None,
        windows_title: str = None,
        callback: typing.Callable = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if json_file is None:
            json_file = chisurf.gui.widgets.get_filename()
        if target is None:
            target = chisurf.core.settings.cs_settings
        if windows_title is None:
            windows_title = "Configuration: %s" % json_file

        self.callback = callback
        self._json_file = json_file
        self._target = target
        self._dict = dict()
        self._view = None
        self._tree = None
        self._save_btn = None
        self._building = False

        if json_file and pathlib.Path(json_file).is_file():
            self.json_file = json_file  # loads ``_dict`` from the file
        else:
            self._dict = target

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self.setWindowTitle(windows_title)

        # Filter box: an editable top row to hide non-matching parameters.
        self._filter = QtWidgets.QLineEdit()
        self._filter.setClearButtonEnabled(True)
        self._filter.setPlaceholderText("Filter parameters…")
        self._filter.textChanged.connect(self._apply_filter)
        self._layout.addWidget(self._filter)

        self._build()

    # -- rendering ----------------------------------------------------------
    def _build(self) -> None:
        # Drop any previous tree / save button (keep the filter box).
        for w in (self._tree, self._save_btn):
            if w is not None:
                self._layout.removeWidget(w)
                w.setParent(None)
        self._save_btn = None

        self._view = SettingsView(self._dict, on_change=self._on_change)

        tree = QtWidgets.QTreeWidget()
        from chisurf.gui.widgets.general import table_font, table_row_height, table_header_height
        tree.setFont(table_font())
        row_h = table_row_height()
        font_sz = table_font().pointSize()
        tree.setStyleSheet(f"""
            QTreeView::item {{
                height: {row_h}px;
            }}
            QTreeView QLineEdit, QTreeView QComboBox, QTreeView QSpinBox, QTreeView QDoubleSpinBox, QTreeView QAbstractSpinBox {{
                font-size: {font_sz}px;
                padding: 0px 2px;
                height: {row_h - 2}px;
                margin: 0px;
            }}
            QTreeView QCheckBox {{
                margin: 0px;
            }}
        """)

        tree.setColumnCount(3)
        tree.setHeaderLabels(["Parameter", "Value", ""])
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        header = tree.header()
        header.setFont(table_font())
        header.setMinimumHeight(table_header_height())
        header.setMaximumHeight(table_header_height())
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        tree.setColumnWidth(0, 160)
        self._tree = tree

        self._add_sections(self._view.view_spec().sections, tree.invisibleRootItem())
        tree.expandAll()
        self._layout.addWidget(tree, 1)

        if self._json_file and pathlib.Path(self._json_file).is_file():
            self._save_btn = QtWidgets.QPushButton("Save")
            self._save_btn.clicked.connect(self.save)
            self._layout.addWidget(self._save_btn)

        if self._filter.text():
            self._apply_filter(self._filter.text())

    def _add_sections(self, sections, parent_item) -> None:
        for section in sections:
            if isinstance(section, PanelSection):
                group = QtWidgets.QTreeWidgetItem(parent_item, [section.title or ""])
                font = group.font(0)
                font.setBold(True)
                group.setFont(0, font)
                self._add_sections(section.sections, group)
            elif isinstance(section, (ValueSection, ToggleSection, ChoiceSection)):
                self._add_leaf(section, parent_item)

    def _add_leaf(self, section, parent_item) -> None:
        label = section.label or section.attr or ""
        item = QtWidgets.QTreeWidgetItem(parent_item, [label])

        group = getattr(self._view, section.target)
        current = getattr(group, section.attr)
        read_only = bool(getattr(section, "read_only", False))

        if isinstance(section, ChoiceSection):
            options = list(section.options)
            if not options and section.options_source:
                from chisurf.gui.autoform.sections.builtin import _resolve_options_source
                options = _resolve_options_source(section.options_source)
            _labels = list(getattr(section, "labels", ()))
            _has_labels = bool(_labels) and len(_labels) == len(options)
            self._add_choice_leaf(
                item, group, section.attr, options, _labels if _has_labels else None, read_only
            )
            return

        # A list value is a choice/enumeration → render it as a combobox (the
        # old parameter-tree's ListParameter). Selecting moves the chosen item
        # to the front so the choice is reflected at index 0 while the full list
        # is preserved (so it round-trips and is reversible via reset).
        if isinstance(current, (list, tuple)):
            self._add_list_leaf(item, group, section.attr, list(current))
            return

        kind = "bool" if isinstance(section, ToggleSection) else section.kind

        def commit(value, _g=group, _a=section.attr):
            setattr(_g, _a, value)  # mutates the live dict + fires the callback

        editor, setter = self._make_field(kind, read_only, current, commit)
        self._tree.setItemWidget(item, 1, editor)

        if not read_only:
            reset = QtWidgets.QToolButton()
            reset.setText("↩")
            reset.setAutoRaise(True)
            reset.setToolTip(f"Reset to {current!r}")
            reset.setStyleSheet("QToolButton { color: #d4a017; border: none; }")

            def do_reset(_=None, _e=editor, _s=setter, _d=current, _c=commit):
                _e.blockSignals(True)
                _s(_d)
                _e.blockSignals(False)
                _c(_d)

            reset.clicked.connect(do_reset)
            self._tree.setItemWidget(item, 2, reset)

    def _add_choice_leaf(self, item, group, attr, options, labels=None, read_only=False) -> None:
        combo = QtWidgets.QComboBox()
        for i, opt in enumerate(options):
            display = labels[i] if labels is not None else str(opt)
            combo.addItem(display)

        current = getattr(group, attr)
        if current is not None:
            for i, opt in enumerate(options):
                if str(opt) == str(current):
                    combo.setCurrentIndex(i)
                    break

        if read_only:
            combo.setEnabled(False)

        def on_choice(idx, _g=group, _a=attr, _opts=options):
            if 0 <= idx < len(_opts):
                setattr(_g, _a, _opts[idx])

        combo.currentIndexChanged.connect(on_choice)
        self._tree.setItemWidget(item, 1, combo)

        if not read_only:
            reset = QtWidgets.QToolButton()
            reset.setText("↩")
            reset.setAutoRaise(True)
            reset.setToolTip(f"Reset to {current!r}")
            reset.setStyleSheet("QToolButton { color: #d4a017; border: none; }")

            def do_reset(_=None, _c=combo, _d=current, _opts=options, _g=group, _a=attr):
                _c.blockSignals(True)
                for idx, opt in enumerate(_opts):
                    if str(opt) == str(_d):
                        _c.setCurrentIndex(idx)
                        break
                _c.blockSignals(False)
                setattr(_g, _a, _d)

            reset.clicked.connect(do_reset)
            self._tree.setItemWidget(item, 2, reset)

    def _add_list_leaf(self, item, group, attr, items) -> None:
        combo = QtWidgets.QComboBox()
        for v in items:
            combo.addItem(str(v))
        combo.setCurrentIndex(0)

        def on_choice(idx, _g=group, _a=attr, _items=items):
            if idx <= 0:
                return
            setattr(_g, _a, [_items[idx]] + _items[:idx] + _items[idx + 1 :])

        combo.currentIndexChanged.connect(on_choice)
        self._tree.setItemWidget(item, 1, combo)

        reset = QtWidgets.QToolButton()
        reset.setText("↩")
        reset.setAutoRaise(True)
        reset.setToolTip("Reset order")
        reset.setStyleSheet("QToolButton { color: #d4a017; border: none; }")

        def do_reset(_=None, _g=group, _a=attr, _items=list(items), _c=combo):
            _c.blockSignals(True)
            _c.setCurrentIndex(0)
            _c.blockSignals(False)
            setattr(_g, _a, list(_items))

        reset.clicked.connect(do_reset)
        self._tree.setItemWidget(item, 2, reset)

    @staticmethod
    def _make_field(kind, read_only, current, commit):
        """Build (editor_widget, setter) for one scalar leaf."""
        if kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(current))
            if not read_only:
                w.toggled.connect(lambda v: commit(bool(v)))
            return w, lambda val: w.setChecked(bool(val))

        if kind == "int":
            w = QtWidgets.QSpinBox()
            w.setRange(-2_147_483_648, 2_147_483_647)
            w.setValue(int(current))
            if not read_only:
                w.valueChanged.connect(lambda v: commit(int(v)))
            return w, lambda val: w.setValue(int(val))

        if kind == "float":
            w = ScientificDoubleSpinBox(value=float(current))
            if not read_only:
                w.sigValueChanged.connect(lambda sb: commit(float(sb.value())))
            return w, lambda val: w.setValue(float(val))

        # "str" — also the fallback for read-only callables / lists / None.
        w = QtWidgets.QLineEdit()
        w.setText("" if current is None else str(current))
        if read_only:
            w.setReadOnly(True)
        else:
            w.editingFinished.connect(lambda: commit(w.text()))
        return w, lambda val: w.setText("" if val is None else str(val))

    # -- filtering ----------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        if self._tree is None:
            return
        needle = text.strip().lower()
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            self._filter_item(root.child(i), needle)

    def _filter_item(self, item, needle: str) -> bool:
        """Hide *item* unless it (or a descendant) matches; returns visibility."""
        child_match = False
        for i in range(item.childCount()):
            child_match = self._filter_item(item.child(i), needle) or child_match
        self_match = (needle in item.text(0).lower()) if needle else True
        visible = self_match or child_match
        item.setHidden(not visible)
        if needle and child_match:
            item.setExpanded(True)
        return visible

    def _on_change(self) -> None:
        if self.callback is not None:
            try:
                self.callback()
            except Exception:
                pass

    def update(self) -> None:
        """Rebuild the editor from the current backing dict."""
        if self._building:
            return
        self._building = True
        try:
            self._build()
        finally:
            self._building = False

    # -- data ---------------------------------------------------------------
    @property
    def dict(self) -> dict:
        """Current settings as a normalized snapshot.

        The string ``"None"`` is mapped back to Python ``None`` (recursively),
        matching the behaviour consumers relied on (e.g. a pyqtgraph
        ``symbol="None"`` means *no symbol*). Non-container values — including
        callables — are passed through by reference, so the dict round-trips.
        """
        return _normalize_none(self._dict)

    def save(self, *args, filename: str = None) -> None:
        if filename is None:
            filename = self._json_file
        with open(filename, "w+") as fp:
            json.dump(self._dict, fp, indent=4)
        self._target = self._dict

    @property
    def json_file(self) -> str:
        return self._json_file

    @json_file.setter
    def json_file(self, v: str) -> None:
        with io.open_maybe_zipped(v, mode="r") as fp:
            self._dict = json.load(fp)
        self._json_file = v


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    win = ParameterEditor()
    win.show()
    sys.exit(app.exec_())
