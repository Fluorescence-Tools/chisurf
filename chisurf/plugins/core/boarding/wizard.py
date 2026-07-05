"""First-run onboarding wizard host.

The layout and flow now live entirely in ``boarding.view.json`` and are rendered
by :class:`~chisurf.gui.autoform.AutoForm` over the Qt-free
:class:`~chisurf.plugins.core.boarding.view_model.BoardingViewModel`. This module
is only the thin window that hosts that form — the class and ``show_onboarding``
names are preserved so the startup launch path and the plugin manifest keep
working. Mirrors the ALEX Creator tool host (``AlexPTUCreator``).
"""

import logging

from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf.gui.autoform import AutoForm

from .view_model import BoardingViewModel

logger = logging.getLogger(__name__)


class BoardingAssistantWidget(QtWidgets.QWidget):
    """Embeddable onboarding assistant: ``AutoForm`` over ``boarding.view.json``.

    A plain widget (no window chrome) so it can be dropped into the wizard dialog
    *or* hosted as a panel in the unified Settings tool. Mirrors the ALEX Creator
    tool host (``AlexPTUCreator``).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = BoardingViewModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.auto_form = AutoForm(self.model)
        layout.addWidget(self.auto_form)

        self.model.add_observer(self._on_model_event)

    def _on_model_event(self, event: str) -> None:
        # Re-read the info panels (status/deps/repair) and update ✓ marks / Next gate.
        try:
            self.auto_form.refresh_plots()
        except Exception:
            logger.warning("boarding: refresh failed", exc_info=True)


class WelcomeToChiSurfWizard(QtWidgets.QDialog):
    """Directed onboarding wizard rendered from ``boarding.view.json``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to ChiSurf")
        self.setMinimumSize(900, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.assistant = BoardingAssistantWidget(self)
        layout.addWidget(self.assistant)

    @property
    def model(self):
        """The backing :class:`BoardingViewModel` (kept for callers/tests)."""
        return self.assistant.model


def show_onboarding(parent=None):
    """Create, show, and retain the onboarding wizard."""
    wiz = WelcomeToChiSurfWizard(parent=parent)
    wiz.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
    try:
        wiz.setWindowModality(QtCore.Qt.ApplicationModal)
    except Exception:
        pass
    wiz.show()
    try:
        wiz.raise_()
        wiz.activateWindow()
    except Exception:
        pass
    cs.__init_chisurf_wizard__ = wiz
    return wiz


def _main():
    """Run the onboarding wizard when executed as a legacy plugin script."""
    try:
        parent = getattr(cs, "cs", None)
    except Exception:
        parent = None
    show_onboarding(parent=parent)


if __name__ in {"__main__", "plugin"}:
    _main()
