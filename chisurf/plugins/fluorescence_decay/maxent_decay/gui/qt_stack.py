from __future__ import annotations

pg = None
chisurf = None
QtWidgets = None
QtCore = None
ExperimentalDataSelector = None


def ensure_qt_stack():
    global pg, chisurf, QtWidgets, QtCore, ExperimentalDataSelector

    if QtWidgets is None or QtCore is None:
        from qtpy import QtWidgets as _QtWidgets, QtCore as _QtCore  # type: ignore

        QtWidgets = _QtWidgets
        QtCore = _QtCore

    if pg is None:
        import pyqtgraph as _pg  # type: ignore

        pg = _pg

    if chisurf is None:
        import chisurf as _chisurf  # type: ignore

        chisurf = _chisurf

    if ExperimentalDataSelector is None:
        from chisurf.gui.widgets.experiments import (
            ExperimentalDataSelector as _ExperimentalDataSelector,
        )  # type: ignore

        ExperimentalDataSelector = _ExperimentalDataSelector

    return pg, QtWidgets, QtCore, chisurf, ExperimentalDataSelector
