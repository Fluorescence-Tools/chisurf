from __future__ import annotations

import abc

from qtpy import QtWidgets, QtCore

import chisurf.fitting
import chisurf.gui
import chisurf.gui.widgets

from chisurf.gui.widgets import View


class Plot(View):

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            parent=None,
            plot_controller: QtWidgets.QWidget = None,
            **kwargs
    ):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.parent = parent
        self.fit = fit
        if plot_controller is None:
            self.plot_controller = QtWidgets.QWidget()
        else:
            self.plot_controller = plot_controller
        self.widgets = list()

    def update(self, *args, **kwargs) -> None:
        try:
            # Always propagate to QWidget.update for repaint scheduling
            super().update(*args, **kwargs)
        except Exception:
            pass
        # If subclass did not override update but provides update_all, call it
        try:
            update_all = getattr(self, 'update_all', None)
            if callable(update_all) and type(self).update is Plot.update:
                update_all(*args, **kwargs)
        except Exception:
            pass

    def showEvent(self, event):
        try:
            super().showEvent(event)
        except Exception:
            pass
        # Defer the refresh to when the widget is fully shown
        try:
            QtCore.QTimer.singleShot(0, self._refresh_on_show)
        except Exception:
            # As a fallback, call directly
            self._refresh_on_show()

    def _refresh_on_show(self):
        """Ensure plots refresh their data when the widget becomes visible."""
        try:
            # Calling self.update will trigger subclass-specific refresh if available
            self.update()
        except Exception:
            try:
                update_all = getattr(self, 'update_all', None)
                if callable(update_all):
                    update_all()
            except Exception:
                pass

    def close(self):
        QtWidgets.QWidget.close(self)
        if isinstance(self.plot_controller, QtWidgets.QWidget):
            self.plot_controller.close()
