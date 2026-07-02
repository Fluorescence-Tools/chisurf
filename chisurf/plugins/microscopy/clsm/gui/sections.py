"""Custom AutoForm sections for the CLSM tool.

Only one genuinely bespoke piece remains as a registered ``custom`` section:

- ``clsm_roi_list`` — the ROI list with add/apply/remove/save/load.

Everything else is now authored declaratively in ``clsm.view.json`` using
AutoForm primitives: the File panel uses ``value`` (incl. ``kind="file"``),
``choice`` (with model-backed options and add/remove buttons) and ``button_row``
sections; the image uses AutoForm's reusable brush+colormap ``image`` section
(:class:`~chisurf.gui.autoform.sections.builtin.ImageMapWidget`).

The factory takes ``(model, target, **options)`` where *model* is the
:class:`ClsmViewModel`; it subscribes to the model's observer hook and refreshes
on the relevant events. It is imported (and thus registered) by ``gui.tool``.
"""

from __future__ import annotations

import pathlib

from qtpy import QtWidgets

import chisurf as cs
from chisurf.gui.autoform.sections.registry import register_section

log = cs.logging


def _tool_button(text: str, tooltip: str = "", slot=None) -> QtWidgets.QToolButton:
    """Build a configured ``QToolButton`` in one call."""
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    if tooltip:
        btn.setToolTip(tooltip)
    if slot is not None:
        btn.clicked.connect(slot)
    return btn


def _working_dir(model) -> str:
    if model.filename:
        return str(pathlib.Path(model.filename).parent)
    try:
        return str(cs.working_path)
    except Exception:
        return str(pathlib.Path.home())


# The interactive image canvas is provided by AutoForm's reusable ``image``
# section (``ImageMapWidget``), configured with brush/draw + colormap options in
# ``clsm.view.json`` — no bespoke canvas widget is needed here.


# ── ROI list ────────────────────────────────────────────────────────────────
@register_section("clsm_roi_list")
def clsm_roi_list(model, target=None, **options):
    """List of saved ROIs with add / apply / remove / save / load."""
    return _RoiList(model)


class _RoiList(QtWidgets.QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        name_row = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("ROI name")
        add_btn = _tool_button("+", "Save the current selection as an ROI", self._on_add)
        name_row.addWidget(self.name_edit)
        name_row.addWidget(add_btn)
        layout.addLayout(name_row)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemClicked.connect(self._on_apply)
        layout.addWidget(self.list_widget)

        btn_row = QtWidgets.QHBoxLayout()
        for text, tip, slot in (
            ("Remove", "Remove the selected ROI", self._on_remove),
            ("Save", "Save the selected ROI to an image file", self._on_save),
            ("Load", "Load an ROI from an image file", self._on_load),
        ):
            btn_row.addWidget(_tool_button(text, tip, slot))
        layout.addLayout(btn_row)

        model.add_observer(self._on_event)

    def _current_name(self):
        item = self.list_widget.currentItem()
        return item.text() if item else ""

    def _on_add(self):
        name = self.name_edit.text().strip() or f"roi_{len(self._model.rois) + 1}"
        self._model.add_roi(name)

    def _on_apply(self, item):
        self._model.apply_roi(item.text())

    def _on_remove(self):
        name = self._current_name()
        if name:
            self._model.remove_roi(name)

    def _on_save(self):
        name = self._current_name()
        if not name:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROI", _working_dir(self._model), "PNG (*.png);;TIFF (*.tif)"
        )
        if fn:
            self._model.save_roi(name, fn)

    def _on_load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROI", _working_dir(self._model), "Images (*.tif *.png)"
        )
        if fn:
            self._model.load_roi(fn)

    def _on_event(self, event):
        if event == "roi":
            self.list_widget.clear()
            self.list_widget.addItems(list(self._model.rois))
