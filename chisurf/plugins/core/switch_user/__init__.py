"""Core plugin for switching the active MFDB user."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "Setup:Switch User"


def show_switch_user(parent: QtWidgets.QWidget | None = None) -> None:
    """Open the ChiSurf login dialog to switch the active MFDB user.

    Parameters
    ----------
    parent : QtWidgets.QWidget, optional
        Parent widget for the modal dialog.
    """
    from chisurf.gui import LoginDialog

    dialog = LoginDialog(parent=parent)
    dialog.exec()


class SwitchUserWidget(QtWidgets.QWidget):
    """Transient plugin widget that opens the MFDB login dialog."""

    def showEvent(self, event) -> None:
        """Open the switch-user dialog when the plugin is shown.

        Parameters
        ----------
        event : QtCore.QShowEvent
            Qt show event.
        """
        super().showEvent(event)
        show_switch_user(parent=self)
        self.close()


if __name__ == "plugin":
    show_switch_user(globals().get("window"))


__all__ = ["SwitchUserWidget", "name", "show_switch_user"]
