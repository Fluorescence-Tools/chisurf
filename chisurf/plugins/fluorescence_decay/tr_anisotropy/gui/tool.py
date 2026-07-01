"""Anisotropy wizard host.

The layout and flow live in ``anisotropy.view.json`` and are rendered by
:class:`~chisurf.gui.autoform.AutoForm` over the
:class:`~chisurf.plugins.fluorescence_decay.tr_anisotropy.gui.view_model.AnisotropyViewModel`.
This module is the thin window that hosts the form. ``ChisurfWizard`` is kept as
an alias so the plugin manifest and legacy launch paths keep working.
"""

from __future__ import annotations

import logging

from qtpy import QtWidgets

from chisurf.gui.autoform import AutoForm

from .view_model import AnisotropyViewModel

logger = logging.getLogger(__name__)

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:  # pragma: no cover

    def persist_plugin_state(_n):
        """No-op fallback when the state-persistence helper is unavailable."""
        return lambda c: c


class AnisotropyAssistantWidget(QtWidgets.QWidget):
    """Embeddable anisotropy assistant: ``AutoForm`` over ``anisotropy.view.json``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = AnisotropyViewModel()

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
            logger.warning("anisotropy: refresh failed", exc_info=True)


@persist_plugin_state("tr_anisotropy")
class AnisotropyWizard(QtWidgets.QDialog):
    """Time-resolved anisotropy wizard rendered from ``anisotropy.view.json``."""

    name = "Anisotropy-Wizard"

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent)
        self.setWindowTitle("TCSPC — Anisotropy")
        self.setMinimumSize(820, 560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.assistant = AnisotropyAssistantWidget(self)
        layout.addWidget(self.assistant)

    @property
    def model(self):
        """The backing :class:`AnisotropyViewModel` (kept for callers/tests)."""
        return self.assistant.model


#: Backwards-compatible alias for the historical class name.
ChisurfWizard = AnisotropyWizard


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = AnisotropyWizard()
    win.show()
    app.exec_()
