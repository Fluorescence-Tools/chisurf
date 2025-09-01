from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments

from chisurf.models.tcspc.nusiance import Generic


class GenericWidget(QtWidgets.QGroupBox, Generic):

    def change_bg_curve(self, background_index: int = None):
        if isinstance(background_index, int):
            self.background_select.selected_curve_index = background_index
        self._background_curve = self.background_select.selected_dataset

        self.lineEdit.setText(self.background_select.curve_name)
        self.fit.model.update()

    def onUnloadBackground(self):
        """Unload the background curve and reset it to default (None)
        """
        chisurf.run("cs.current_fit.model.generic.unload_background_curve()")
        self.lineEdit.setText("")
        chisurf.run("cs.current_fit.model.update()")

    def update(self):
        super().update()
        self.lineedit_nphBg.setText("%i" % self.n_ph_bg)
        self.lineedit_nphFl.setText("%i" % self.n_ph_fl)

    def __init__(
            self,
            hide_generic: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if hide_generic:
            self.hide()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.setTitle("Generic")

        # Generic parameters
        sc_w = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._sc,
            label_text='Sc',
        )
        bg_w = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bg,
            label_text='Bg'
        )
        tmeas_bg_w = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._tmeas_bg,
            label_text='t<sub>Bg</sub>',
            callback=self.update,
            hide_bounds = True
        )
        tmeas_exp_w = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._tmeas_exp,
            label_text='t<sub>Meas</sub>',
            callback=self.update,
            hide_bounds=True
        )

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(sc_w, 1, 0)
        layout.addWidget(bg_w, 1, 1)
        layout.addWidget(tmeas_bg_w, 2, 0)
        layout.addWidget(tmeas_exp_w, 2, 1)
        self.layout.addLayout(layout)

        ly = QtWidgets.QHBoxLayout()
        layout.addLayout(ly, 0, 0, 1, 2)
        ly.addWidget(QtWidgets.QLabel('Background file:'))
        self.lineEdit = QtWidgets.QLineEdit()
        ly.addWidget(self.lineEdit)

        open_bg = QtWidgets.QPushButton()
        open_bg.setText('...')
        ly.addWidget(open_bg)

        unload_bg = QtWidgets.QPushButton()
        unload_bg.setText('X')
        unload_bg.setToolTip('Unload background file')
        ly.addWidget(unload_bg)
        unload_bg.clicked.connect(self.onUnloadBackground)

        self.background_select = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            parent=None,
            change_event=self.change_bg_curve,
            fit=self.fit,
            experiment=self.fit.data.experiment.__class__
        )
        open_bg.clicked.connect(self.background_select.show)

        a = QtWidgets.QHBoxLayout()
        a.addWidget(QtWidgets.QLabel('nPh(Bg)'))
        self.lineedit_nphBg = QtWidgets.QLineEdit()
        a.addWidget(self.lineedit_nphBg)
        layout.addLayout(a, 3, 0, 1, 1)

        a = QtWidgets.QHBoxLayout()
        a.addWidget(QtWidgets.QLabel('nPh(Fl)'))
        self.lineedit_nphFl = QtWidgets.QLineEdit()
        a.addWidget(self.lineedit_nphFl)
        layout.addLayout(a, 3, 1, 1, 1)