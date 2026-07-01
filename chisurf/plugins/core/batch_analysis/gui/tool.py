"""Batch-Analysis wizard host.

The layout and flow live entirely in ``batch.view.json`` and are rendered by
:class:`~chisurf.gui.autoform.AutoForm` over the Qt-free
:class:`~chisurf.plugins.core.batch_analysis.gui.view_model.BatchViewModel`. This
module is the thin window that hosts that form. The historical name
``BatchProcessingWizard`` is preserved so the plugin manifest and launch paths
keep working. Mirrors the boarding wizard host.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from .view_model import BatchViewModel

logger = logging.getLogger(__name__)


class BatchAnalysisWidget(QtWidgets.QWidget):
    """Embeddable batch-analysis assistant: ``AutoForm`` over ``batch.view.json``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = BatchViewModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)

        self.model.add_observer(self._on_model_event)

    def _on_model_event(self, event: str) -> None:
        try:
            self.auto_form.refresh_plots()
        except Exception:
            logger.warning("batch: refresh failed", exc_info=True)


class BatchProcessingWizard(QtWidgets.QDialog):
    """Batch-analysis wizard rendered from ``batch.view.json``."""

    name = "Batch-Analysis"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Analysis")
        self.setMinimumSize(760, 560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.assistant = BatchAnalysisWidget(self)
        layout.addWidget(self.assistant)

    @property
    def model(self):
        """The backing :class:`BatchViewModel` (kept for callers/tests)."""
        return self.assistant.model


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = BatchProcessingWizard()
    win.show()
    app.exec_()
