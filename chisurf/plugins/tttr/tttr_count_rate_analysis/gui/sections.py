"""Custom AutoForm sections for the Count Rate Analysis tool.

Three bespoke Qt widgets are registered here: the detector channel-definition
page (``count_rate_channels``), the drag-drop file list with the action
tool-buttons (``count_rate_files``) and the per-channel results table
(``count_rate_results``). The count-rate-vs-file plot is a declarative ``plot``
section in ``count_rate.view.json``. The widgets own only Qt concerns and drive
the Qt-free :class:`~..view_model.CountRateViewModel`. Imported (registered) by
``gui.tool``.
"""

from __future__ import annotations

import logging

from qtpy import QtCore, QtWidgets

from chisurf.gui.autoform.sections.registry import register_section

logger = logging.getLogger(__name__)


def _tool_button(text: str, tooltip: str, slot) -> QtWidgets.QToolButton:
    """Build a configured ``QToolButton`` (emoji label + tooltip) in one call."""
    btn = QtWidgets.QToolButton()
    btn.setText(text)
    btn.setToolTip(tooltip)
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    btn.clicked.connect(slot)
    return btn


# ---------------------------------------------------------------------------
# count_rate_channels — detector channel definition page
# ---------------------------------------------------------------------------


@register_section("count_rate_channels")
def count_rate_channels(model, target=None, **options):
    """AutoForm factory for the detector channel-definition page."""
    from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

    page = DetectorWizardPage(
        show_edit_json=False,
        show_save=False,
        show_setups_file=True,
        show_setup_selection=True,
        show_help=True,
        show_tttr_reading=True,
        show_tables=True,
        show_add_inputs=True,
    )
    # Inject the channels source into the Qt-free model.
    model.channels_provider = page.channels
    return page


# ---------------------------------------------------------------------------
# count_rate_files — drag-drop file list + actions
# ---------------------------------------------------------------------------


@register_section("count_rate_run")
def count_rate_run(model, target=None, **options):
    """AutoForm factory for the Calculate / Save action bar.

    The file list is the general ``path_list`` section (target ``files``); this
    bar only computes and exports.
    """
    return _RunSection(model)


class _RunSection(QtWidgets.QWidget):
    """Calculate + Save action bar for the count-rate analysis."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        bar = QtWidgets.QHBoxLayout(self)
        bar.setContentsMargins(2, 2, 2, 2)
        bar.addStretch(1)
        bar.addWidget(
            _tool_button("📈 Calculate", "Compute count rates for all files.", self._calculate)
        )
        bar.addWidget(_tool_button("💾 Save", "Save the results table as text.", self._save))

    def _calculate(self) -> None:
        reason = self._model.can_compute()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "Cannot calculate", reason)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self._model.compute()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _save(self) -> None:
        reason = self._model.can_save()
        if reason is not None:
            QtWidgets.QMessageBox.warning(self, "No data", reason)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Table as Text File", "", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            self._model.save_table(path)
            QtWidgets.QMessageBox.information(self, "Success", f"Table saved to {path}")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save file: {exc}")


# ---------------------------------------------------------------------------
# count_rate_results — per-channel results table
# ---------------------------------------------------------------------------


@register_section("count_rate_results")
def count_rate_results(model, target=None, **options):
    """AutoForm factory for the per-channel results table."""
    return _ResultsSection(model)


class _ResultsSection(QtWidgets.QWidget):
    """Read-only per-channel count-rate results table."""

    _HEADERS = ["Channel", "Mean (kHz)", "Std (kHz)", "#Photons", "Time (s)"]

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._table)

        self._model.add_observer(self._on_model_event)

    def _on_model_event(self, event: str) -> None:
        if event in ("computed", "files"):
            self._refresh()

    def _refresh(self) -> None:
        rows = self._model.results_rows()
        self._table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row["channel"],
                f"{row['mean_khz']:.2f}",
                f"{row['std_khz']:.2f}",
                f"{row['photons']:.0f}",
                f"{row['time_s']:.3f}",
            ]
            for c, val in enumerate(values):
                self._table.setItem(r, c, QtWidgets.QTableWidgetItem(val))


__all__ = ["count_rate_channels", "count_rate_run", "count_rate_results"]
