from __future__ import annotations

import pathlib

import chisurf

from chisurf import typing
from chisurf.gui import QtWidgets, QtCore, QtGui, uic

import chisurf.gui.decorators
import chisurf.math
import chisurf.fitting
import chisurf.fitting.parameter
import chisurf.decorators
import chisurf.experiments
import chisurf.models
import chisurf.models.parse.widget
import chisurf.models.tcspc.fret as fret
import chisurf.gui.widgets
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments
import chisurf.plots

from chisurf.models.model import ModelWidget
from chisurf.models.tcspc.anisotropy import Anisotropy
from chisurf.models.tcspc.lifetime import Lifetime, LifetimeModel, LifetimeMixtureModel
from chisurf.models.tcspc.mix_model import LifetimeMixModel
from chisurf.models.tcspc.nusiance import Convolve, Corrections, Generic
from chisurf.models.parse.tcspc.tcspc_parse import ParseDecayModel
from chisurf.models.tcspc.pddem import PDDEM, PDDEMModel


class ConvolveWidget(Convolve, QtWidgets.QWidget):

    @property
    def fwhm(self) -> float:
        return self.irf.fwhm

    @fwhm.setter
    def fwhm(self, v: float):
        self.lineEdit_2.setText("%.3f" % v)

    @property
    def gui_mode(self):
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"

    @chisurf.gui.decorators.init_with_ui("tcspc_convolve.ui")
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            hide_curve_convolution: bool = True,
            *args,
            **kwargs
    ):
        if hide_curve_convolution:
            self.radioButton_3.setVisible(not hide_curve_convolution)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._dt, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._n0, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._start, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._stop, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._lb, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._ts, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._iw, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._ik, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            fitting_parameter=self._rep,
            layout=self.horizontalLayout_3,
            label_text='r[MHz]'
        )

        self.irf_select = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            parent=None,
            change_event=self.change_irf,
            fit=self.fit,
            experiment=self.fit.data.experiment.__class__
        )

        self.actionSelect_IRF.triggered.connect(self.irf_select.show)
        self.radioButton_3.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton_2.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton.clicked.connect(self.onConvolutionModeChanged)
        self.checkBox.clicked.connect(self.onConvolutionModeChanged)

    def onConvolutionModeChanged(self):
        chisurf.run(
            "\n".join(
                [
                    f"for f in cs.current_fit:",
                    f"   f.model.convolve.mode = '{self.gui_mode}'",
                    f"cs.current_fit.model.convolve.do_convolution = {self.checkBox.isChecked()}",
                    f"cs.current_fit.update()"
                ]
            )
        )

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        name = self.irf_select.curve_name
        chisurf.run(f"chisurf.macros.model_tcspc.change_irf({idx}, '{name}')")
        self.fwhm = self.irf.fwhm



class CorrectionsWidget(Corrections, QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspcCorrections.ui")
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            hide_corrections: bool = False,
            threshold: float = 0.9,
            reverse: bool = False,
            enabled: bool = False,
            **kwargs
    ):
        self.groupBox.setChecked(False)
        self.comboBox.addItems(chisurf.math.signal.window_function_types)
        if hide_corrections:
            self.hide()

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._dead_time,
            layout=self.horizontalLayout_2,
            label_text='t<sub>dead</sub>[ns]'
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._window_length,
            layout=self.horizontalLayout_2,
            label_text='Lin.win.'
        )

        self.lin_select = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            parent=None,
            change_event=self.onChangeLin,
            fit=fit,
            experiment=fit.data.experiment.__class__
        )

        self.actionSelect_lintable.triggered.connect(self.lin_select.show)

        self.checkBox_3.toggled.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.correct_pile_up = %s\n" %
                self.checkBox_3.isChecked()
            )
        )

        self.checkBox_2.toggled.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.reverse = %s" %
                self.checkBox_2.isChecked()
            )
        )

        self.checkBox.toggled.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.correct_dnl = %s" %
                self.checkBox.isChecked()
            )
        )

        self.comboBox.currentIndexChanged.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.window_function = '%s'" %
                self.comboBox.currentText()
            )
        )

    def onChangeLin(self):
        idx = self.lin_select.selected_curve_index
        lin_name = self.lin_select.curve_name
        chisurf.run(
            "chisurf.macros.model_tcspc.set_linearization(%s, '%s')" %
            (idx, lin_name)
        )


class GenericWidget(QtWidgets.QGroupBox, Generic):

    def change_bg_curve(self, background_index: int = None):
        if isinstance(background_index, int):
            self.background_select.selected_curve_index = background_index
        self._background_curve = self.background_select.selected_dataset

        self.lineEdit.setText(self.background_select.curve_name)
        self.fit.model.update()

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


class AnisotropyWidget(Anisotropy, QtWidgets.QGroupBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setTitle("Rotational-times")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.setLayout(self.lh)
        self.rot_vis = False
        self._rho_widgets = list()
        self._b_widgets = list()

        self.radioButtonVM = QtWidgets.QRadioButton("VM")
        self.radioButtonVM.setToolTip(
            "Excitation: Vertical\nDetection: Magic-Angle"
        )
        self.radioButtonVM.setChecked(True)
        self.radioButtonVM.clicked.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.anisotropy.polarization_type = 'vm'"
            )
        )
        self.radioButtonVM.clicked.connect(self.hide_roation_parameters)

        self.radioButtonVV = QtWidgets.QRadioButton("VV")
        self.radioButtonVV.setToolTip(
            "Excitation: Vertical\nDetection: Vertical"
        )
        self.radioButtonVV.clicked.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.anisotropy.polarization_type = 'vv'"
            )
        )

        self.radioButtonVH = QtWidgets.QRadioButton("VH")
        self.radioButtonVH.setToolTip(
            "Excitation: Vertical\nDetection: Horizontal"
        )
        self.radioButtonVH.clicked.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.anisotropy.polarization_type = 'vh'"
            )
        )

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        add_rho = QtWidgets.QPushButton()
        add_rho.setText("add")
        layout.addWidget(add_rho)
        add_rho.clicked.connect(self.onAddRotation)

        remove_rho = QtWidgets.QPushButton()
        remove_rho.setText("del")
        layout.addWidget(remove_rho)
        remove_rho.clicked.connect(self.onRemoveRotation)

        spacerItem = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        layout.addItem(spacerItem)

        layout.addWidget(self.radioButtonVM)
        layout.addWidget(self.radioButtonVV)
        layout.addWidget(self.radioButtonVH)

        self.lh.addLayout(layout)

        self.gb = QtWidgets.QGroupBox()
        self.lh.addWidget(self.gb)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)
        self.gb.setLayout(self.lh)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._r0,
            label_text='r<sub>0</sub>',
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._g,
            label_text='g',
            layout=layout
        )
        self.lh.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._l1,
            label_text='l<sub>1</sub>',
            layout=layout,
            decimals=4
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._l2,
            label_text='l<sub>2</sub>',
            layout=layout,
            decimals=4
        )
        self.lh.addLayout(layout)

        self.lh.addLayout(layout)
        self.add_rotation()
        self.hide_roation_parameters()

    def hide_roation_parameters(self):
        self.rot_vis = not self.rot_vis
        if self.rot_vis:
            self.gb.show()
        else:
            self.gb.hide()

    def onAddRotation(self):
        chisurf.run(
            "\n".join(
                [
                    "for f in cs.current_fit:",
                    "   f.model.anisotropy.add_rotation()",
                    "cs.current_fit.update()"
                ]
            )
        )

    def onRemoveRotation(self):
        chisurf.run(
            "\n".join(
                [
                    "for f in cs.current_fit:",
                    "   f.model.anisotropy.remove_rotation()",
                    "cs.current_fit.update()"
                ]
            )
        )

    def add_rotation(self, **kwargs):
        super().add_rotation(**kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.lh.addLayout(layout)
        self._b_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._bs[-1],
                decimals=4,
                layout=layout
            )
        )
        self._rho_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._rhos[-1],
                decimals=4,
                layout=layout
            )
        )

    def remove_rotation(self):
        self._rhos.pop()
        self._bs.pop()
        self._rho_widgets.pop().close()
        self._b_widgets.pop().close()


class PDDEMWidget(QtWidgets.QWidget, PDDEM):

    @chisurf.gui.decorators.init_with_ui("pddem.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._fAB,
            layout=layout,
            label_text='A>B'
        )
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._fBA,
            layout=layout,
            label_text='B>A'
        )
        self.verticalLayout_3.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._pA,
            layout=layout
        )
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._pB,
            layout=layout
        )
        self.verticalLayout_3.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._pxA,
            layout=layout,
            label_text='Ex<sub>A</sub>'
        )
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._pxB,
            layout=layout,
            label_text='Ex<sub>B</sub>'
        )
        self.verticalLayout_3.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._pmA,
            layout=layout,
            label_text='Em<sub>A</sub>'
        )
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._pmB,
            layout=layout,
            label_text='Em<sub>B</sub>'
        )
        self.verticalLayout_3.addLayout(layout)


class PDDEMModelWidget(ModelWidget, PDDEMModel):

    plot_classes = [
        (
            chisurf.plots.LinePlot, {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'time',
                'y_label': 'counts',
                'reference_curve': False
            }
         ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.DistributionPlot, {}),
        (chisurf.plots.ParameterScanPlot, {})
    ]

    def __init__(self, fit, **kwargs):
        super().__init__(
            fit,
            icon=QtGui.QIcon(":/icons/icons/TCSPC.ico"),
            **kwargs
        )

        self.convolve = ConvolveWidget(
            name='convolve',
            fit=fit,
            model=self,
            dt=fit.data.dx,
            hide_curve_convolution=True,
            **kwargs
        )

        self.corrections = CorrectionsWidget(fit=fit, model=self, **kwargs)
        self.generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        self.anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        self.pddem = PDDEMWidget(parent=self, model=self, short='P')
        self.gaussians = GaussianWidget(
            donors=None,
            model=self.model,
            short='G',
            no_donly=True,
            name='gaussians'
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(self.convolve)
        self.layout.addWidget(self.generic)
        self.layout.addWidget(self.pddem)

        self.fa = LifetimeWidget(
            title='Lifetimes-A',
            model=self.model,
            short='A',
            name='fa'
        )
        self.fb = LifetimeWidget(
            title='Lifetimes-B',
            model=self.model,
            short='B',
            name='fb'
        )
        self.donor = self.fa

        self.layout.addWidget(self.fa)
        self.layout.addWidget(self.fb)

        self.layout.addWidget(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
                self.fret_parameters
            )
        )

        self.layout.addWidget(self.gaussians)
        self.layout.addWidget(self.anisotropy)
        self.layout.addWidget(self.corrections)


class LifetimeWidget(Lifetime, QtWidgets.QWidget):

    def update(self, *__args):
        Lifetime.update(self)
        QtWidgets.QWidget.update(self, *__args)
        for w, v in zip(self._amp_widgets, self.amplitudes):
            w.setValue(v)

    @property
    def parameter_widgets(self):
        return self._amp_widgets + self._lifetime_widgets

    def read_values(self, target):

        def linkcall():
            fit_idx = self._amp_widgets[0].fitting_parameter.fit_idx
            for key in self.parameter_dict:
                p = target.parameters_all_dict[key]
                chisurf.run(f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{key}'].value = {p.value}")
                chisurf.run(f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{key}'].controller.finalize()")
            chisurf.run("cs.current_fit.update()")

        return linkcall

    def read_menu(self):
        menu = self.readFrom_menu
        menu.clear()
        for f in chisurf.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.read_values(a))
                menu.addMenu(submenu)

    def link_values(self, target):
        def linkcall():
            self._link = target
            chisurf.run("cs.current_fit.update()")
            self.gb.setChecked(False)
        return linkcall

    def onLinkToggeled(self, checked):
        if checked:
            self._link = None
            chisurf.run("cs.current_fit.update()")

    def link_menu(self):
        menu = self.linkFrom_menu
        menu.clear()
        for f in chisurf.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.link_values(a))
                menu.addMenu(submenu)

    def __init__(self, title: str = '', **kwargs):
        super().__init__(**kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setCheckable(True)
        self.gb.setChecked(True)
        self.gb.toggled.connect(self.onLinkToggeled)
        self.gb.setTitle(title)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
        self._amp_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()
        self._lifetime_widgets: typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()

        lh = QtWidgets.QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(0)

        addLifetime = QtWidgets.QPushButton()
        addLifetime.setText("add")
        addLifetime.clicked.connect(self.onAddLifetime)
        lh.addWidget(addLifetime)

        removeLifetime = QtWidgets.QPushButton()
        removeLifetime.setText("del")
        removeLifetime.clicked.connect(self.onRemoveLifetime)
        lh.addWidget(removeLifetime)

        readFrom = QtWidgets.QToolButton()
        readFrom.setText("read")
        self.readFrom_menu = QtWidgets.QMenu(self.readFrom)
        self.readFrom_menu.aboutToShow.connect(self.read_menu)
        readFrom.setMenu(self.readFrom_menu)
        readFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(readFrom)
        self.readFrom = readFrom

        linkFrom = QtWidgets.QToolButton()
        linkFrom.setText("link")
        self.linkFrom_menu = QtWidgets.QMenu(self.linkFrom)
        self.linkFrom_menu.aboutToShow.connect(self.link_menu)
        linkFrom.setMenu(self.linkFrom_menu)
        linkFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(linkFrom)
        self.linkFrom = linkFrom

        normalize_amplitude = QtWidgets.QCheckBox("Norm.")
        normalize_amplitude.setChecked(True)
        normalize_amplitude.setToolTip("Normalize amplitudes to unity.\nThe sum of all amplitudes equals one.")
        normalize_amplitude.clicked.connect(self.onNormalizeAmplitudes)
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtWidgets.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip("Take absolute value of amplitudes\nNo negative amplitudes")
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
        self.absolute_amplitude = absolute_amplitude

        lh.addWidget(absolute_amplitude)
        lh.addWidget(normalize_amplitude)
        self.lh.addLayout(lh)

        self.append()

    def onNormalizeAmplitudes(self):
        chisurf.run(f"chisurf.macros.model_tcspc.normalize_lifetime_amplitudes({self.normalize_amplitude.isChecked()})")

    def onAbsoluteAmplitudes(self):
        chisurf.run(f"chisurf.macros.model_tcspc.absolute_amplitudes({self.absolute_amplitude.isChecked()})")

    def onAddLifetime(self):
        chisurf.run(f"chisurf.macros.model_tcspc.add_lifetime('{self.name}')")

    def onRemoveLifetime(self):
        chisurf.run(f"chisurf.macros.model_tcspc.remove_lifetime('{self.name}')")

    def append(self, *args, **kwargs):
        Lifetime.append(self, *args, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._amp_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=layout
            )
        )

        self._lifetime_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._lifetimes[-1],
                layout=layout
            )
        )

        self.lh.addLayout(layout)

    def pop(self):
        self._amplitudes.pop()
        self._lifetimes.pop()
        self._amp_widgets.pop().close()
        self._lifetime_widgets.pop().close()


class LifetimeModelWidgetBase(ModelWidget, LifetimeModel):

    plot_classes = [
        (
            chisurf.plots.LinePlot, {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y'
            }
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.DistributionPlot, {}),
        (chisurf.plots.ResidualPlot, {})
    ]

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            icon: QtGui.QIcon = None,
            hide_nuisances: bool = False,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super().__init__(fit=fit, icon=icon)

        corrections = CorrectionsWidget(
            fit=fit,
            **kwargs
        )
        generic = GenericWidget(
            fit=fit,
            parent=self,
            **kwargs
        )
        convolve = ConvolveWidget(
            name='convolve',
            fit=fit,
            hide_curve_convolution=False,
            **kwargs
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        ## add widgets
        if not hide_nuisances:
            layout.addWidget(convolve)
            layout.addWidget(generic)
            layout.addWidget(corrections)

        if hide_nuisances:
            corrections.hide()

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.generic = generic
        self.corrections = corrections
        self.convolve = convolve


class LifetimeModelWidget(LifetimeModelWidgetBase):

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        lifetimes: chisurf.fitting.parameter.FittingParameterGroup = None,
        **kwargs
     ):
        super().__init__(fit=fit, **kwargs)
        if lifetimes is None:
            lifetimes = LifetimeWidget(
                name='lifetimes',
                parent=self,
                title='Lifetimes',
                short='L',
                fit=fit
            )
        self.lifetimes = lifetimes
        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            **kwargs
        )
        self.anisotropy = anisotropy
        self.layout.addWidget(self.lifetimes)
        self.layout.addWidget(anisotropy)

    def finalize(self):
        super().finalize()
        self.lifetimes.update()



class LifetimeMixtureModelWidget(LifetimeMixtureModel, LifetimeModelWidgetBase):

    def __init__(self, fit: chisurf.fitting.FitGroup, **kwargs):
        super().__init__(fit=fit, **kwargs)

        hl = QtWidgets.QHBoxLayout()
        self.layout.addLayout(hl)
        self.cb = QtWidgets.QComboBox(None)
        hl.addWidget(self.cb)

        self.update_button = QtWidgets.QToolButton(None)
        self.update_button.setText("update")
        hl.addWidget(self.update_button)
        self.update_button.clicked.connect(self.onUpdataFitList)

        label = QtWidgets.QLabel('Name')
        self.name_box = QtWidgets.QLineEdit()
        self.name_box.setPlaceholderText("Define name...")
        hl.addWidget(label)
        hl.addWidget(self.name_box)

        self.add_button = QtWidgets.QToolButton(None)
        self.all_fits = QtWidgets.QCheckBox()
        self.all_fits.setChecked(False)
        self.add_button.setText("add")
        self.add_button.clicked.connect(lambda: self.onAddFit(all_fits=self.all_fits.isChecked()))
        hl.addWidget(self.add_button)
        self.all_fits.setText('all')
        hl.addWidget(self.all_fits)

        self.fit_list = QtWidgets.QListWidget()
        self.fit_list.doubleClicked.connect(self.onRemoveFit)
        self.layout.addWidget(self.fit_list)

        self.layout_fractions = QtWidgets.QGridLayout()
        self.layout.addLayout(self.layout_fractions)

    def onRemoveFit(self):
        idx = self.fit_list.currentRow()
        if idx != -1:
            self.fit_list.takeItem(idx)
            self.pop_model(idx)
        else:
            print("Please select an item to remove.")
        self.onUpdateParameterUI()

    def onUpdataFitList(self):
        self.cb.clear()
        names = [f.name for f in self.lifetime_fits]
        self.cb.addItems(names)

    def onAddFit(self, all_fits: bool = False):
        if not all_fits:
            idxs = [self.cb.currentIndex()]
        else:
            idxs = range(0, len(self.lifetime_fits))
        for idx in idxs:
            i = self.fit_list.count() + 1
            f = self.lifetime_fits[idx]
            if len(self.name_box.text()) == 0:
                name = f"x_{i}"
            else:
                name = self.name_box.text()
            self.fit_list.addItem(f'{i}: {f.name}')
            self.append_model(f.model, name)
        self.onUpdateParameterUI()

    def onUpdateParameterUI(self):
        n_columns, row = 2, 1
        layout = self.layout_fractions
        chisurf.gui.widgets.clear_layout(layout)
        for i, x in enumerate(self._fractions):
            pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                label_text=f'x<sub>{i + 1}</sub>',
                fitting_parameter=x
            )
            column = i % n_columns
            if column == 0:
                row += 1
            layout.addWidget(pw, row, column)



class GaussianWidget(fret.Gaussians, QtWidgets.QWidget):

    def __init__(
            self,
            donors,
            model=None,
            **kwargs
    ):
        super().__init__(
            donors=donors,
            model=model,
            **kwargs
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("Gaussian distances")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)

        self._gb = list()

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        addGaussian = QtWidgets.QPushButton()
        addGaussian.setText("add")
        layout.addWidget(addGaussian)

        removeGaussian = QtWidgets.QPushButton()
        removeGaussian.setText("del")
        layout.addWidget(removeGaussian)
        self.lh.addLayout(layout)

        self.lh.addLayout(self.grid_layout)

        addGaussian.clicked.connect(self.onAddGaussian)
        removeGaussian.clicked.connect(self.onRemoveGaussian)

        # add some initial distance
        self.append(1.0, 50.0, 6.0, 0.0)

    def onAddGaussian(self):
        chisurf.run(
            f"for f in cs.current_fit:\n"\
            f"   f.model.{self.name}.append()\n"\
            f"   f.model.update()"
        )

    def onRemoveGaussian(self):
        chisurf.run(
            f"for f in cs.current_fit:\n"\
            f"   f.model.{self.name}.pop()\n"\
            f"   f.model.update()"
        )

    def append(self, *args, **kwargs):
        super().append(50.0,6.0,1.0,)

        gb = QtWidgets.QGroupBox()
        n_gauss = len(self)
        gb.setTitle('G%i' % n_gauss)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianMeans[-1],
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianSigma[-1],
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianShape[-1],
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianAmplitudes[-1],
            layout=layout
        )

        gb.setLayout(layout)
        row = (n_gauss - 1) // 2 + 1
        col = (n_gauss - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    def pop(self) -> None:
        super().pop()
        self._gb.pop().close()


class DiscreteDistanceWidget(fret.DiscreteDistance, QtWidgets.QWidget):

    def __init__(
            self,
            donors,
            model: chisurf.models.Model = None,
            **kwargs
    ):
        super().__init__(
            donors=donors,
            model=model,
            **kwargs
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("FRET-rates")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)

        self._gb = list()

        self.grid_layout = QtWidgets.QGridLayout()

        l = QtWidgets.QHBoxLayout()
        addFRETrate = QtWidgets.QPushButton()
        addFRETrate.setText("add")
        l.addWidget(addFRETrate)

        removeFRETrate = QtWidgets.QPushButton()
        removeFRETrate.setText("del")
        l.addWidget(removeFRETrate)
        self.lh.addLayout(l)

        self.lh.addLayout(self.grid_layout)

        addFRETrate.clicked.connect(self.onAddFRETrate)
        removeFRETrate.clicked.connect(self.onRemoveFRETrate)

        # add some initial distance
        self.append(1.0, 50.0, False)

    def onAddFRETrate(self):
        chisurf.run(
            f"for f in cs.current_fit:\n"\
            f"   f.model.{self.name}.append()\n"\
            f"   f.model.update()"
        )

    def onRemoveFRETrate(self):
        chisurf.run(
            f"for f in cs.current_fit:\n"\
            f"   f.model.{self.name}.pop()\n"\
            f"   f.model.update()"
        )

    def append(self, *args, **kwargs):
        super().append(50., 1.0)

        gb = QtWidgets.QGroupBox()
        n_rates = len(self)
        gb.setTitle('%i' % (n_rates + 1))

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._distances[-1],
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._amplitudes[-1],
            layout=layout
        )

        gb.setLayout(layout)
        row = (n_rates - 1) // 2 + 1
        col = (n_rates - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    def pop(self):
        super().pop()
        self._gb.pop().close()


class GaussianModelWidget(fret.GaussianModel, LifetimeModelWidgetBase):

    plot_classes = [
        (
            chisurf.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
         ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {}),
        (chisurf.plots.DistributionPlot, {})
    ]
    
    def finalize(self):
        super().finalize()
        self.donor.update()

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
        self.donor = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)',
            name='donor'
        )
        gaussians = GaussianWidget(
            donors=self.donor,
            parent=self,
            model=self,
            short='G',
            **kwargs
        )
        fret.GaussianModel.__init__(
            self,
            fit=fit,
            lifetimes=self.donor,
            gaussians=gaussians
        )

        LifetimeModelWidgetBase.__init__(
            self,
            fit=fit,
            **kwargs
        )

        self.layout.addWidget(self.donor)
        self.layout.addWidget(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
                self.fret_parameters
            )
        )
        self.layout.addWidget(gaussians)

        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            **kwargs
        )
        self.anisotropy = anisotropy
        self.layout.addWidget(self.anisotropy)


class FRETrateModelWidget(fret.FRETrateModel, LifetimeModelWidgetBase):

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
        self.donor = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)'
        )
        self.fret_rates = DiscreteDistanceWidget(
            donors=self.donor,
            parent=self,
            model=self,
            short='G',
            **kwargs
        )
        super().__init__(
            fit=fit,
            fret_rates=self.fret_rates,
            donor=self.donor
        )

        self.layout.addWidget(self.donor)
        # self.layout_parameter.addWidget(self.fret_parameters.to_widget())
        self.layout.addWidget(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
                self.fret_parameters
            )
        )
        self.layout.addWidget(self.fret_rates)


class WormLikeChainModelWidget(fret.WormLikeChainModel, LifetimeModelWidgetBase):

    plot_classes = [
        (
            chisurf.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
         ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {}),
        (chisurf.plots.DistributionPlot, {})
    ]

    @property
    def use_dye_linker(self) -> bool:
        return bool(self._use_dye_linker.isChecked())

    @use_dye_linker.setter
    def use_dye_linker(self, v: bool):
        self._use_dye_linker.setChecked(v)

    def __init__(self, fit: chisurf.fitting.fit.Fit, **kwargs):
        self.donor = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)'
        )

        super().__init__(fit, **kwargs)

        layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(layout)

        self._use_dye_linker = QtWidgets.QCheckBox()
        self._use_dye_linker.setText('Use linker')
        layout.addWidget(self._use_dye_linker)

        pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._sigma_linker)
        layout.addWidget(pw)

        pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters)
        layout.addWidget(pw)

        self.layout.addWidget(self.donor)

        pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._chain_length)
        self.layout.addWidget(pw)

        pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._persistence_length,
            layout=layout
        )
        self.layout.addWidget(pw)

        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        #uic.loadUi(pathlib.Path(__file__).parent / "load_distance_distibution.ui", self)
        #self.actionOpen_distirbution.triggered.connect(self.load_distance_distribution)

    # def load_distance_distribution(self, **kwargs):
    #     #print "load_distance_distribution"
    #     verbose = kwargs.get('verbose', self.verbose)
    #     #filename = kwargs.get('filename', str(QtGui.QFileDialog.getOpenFileName(self, 'Open File')))
    #     filename = chisurf.gui.widgets.get_filename(
    #         description='Open distance distribution',
    #         file_type='CSV-files (*.csv)'
    #     )
    #     self.lineEdit.setText(filename)
    #     csv = chisurf.fio.ascii.Csv(filename)
    #     ar = csv.data.T
    #     if verbose:
    #         print("Opening distribution")
    #         print("Filename: %s" % filename)
    #         print("Shape: %s" % ar.shape)
    #     self.rda = ar[0]
    #     self.prda = ar[1]
    #     self.update_model()


class ParseDecayModelWidget(ParseDecayModel, ModelWidget):

    plot_classes = [
        (
            chisurf.plots.LinePlot,
            {
                'd_scalex': 'lin',
                'd_scaley': 'log',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y',
                'plot_irf': True
            }
         ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (chisurf.plots.ResidualPlot, {})
    ]

    def get_curves(self, copy_curves: bool = False) -> typing.Dict[str, chisurf.curve.Curve]:
        d = super().get_curves(copy_curves)
        d['IRF'] = self.convolve.irf
        return d

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            icon: QtGui.QIcon = None,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super(ModelWidget, self).__init__(fit=fit, icon=icon)
        super(ParseDecayModel, self).__init__(fit=fit, icon=icon)

        self.convolve = chisurf.models.tcspc.widgets.ConvolveWidget(
            fit=fit,
            model=self,
            show_convolution_mode=False,
            dt=fit.data.dx,
            **kwargs
        )
        generic = chisurf.models.tcspc.widgets.GenericWidget(
            fit=fit,
            parent=self,
            model=self,
            **kwargs
        )
        fn = pathlib.Path(__file__).parent / 'tcspc.models.json'
        pw = chisurf.models.parse.widget.ParseFormulaWidget(
            model=self,
            model_file=fn
        )
        corrections = chisurf.models.tcspc.widgets.CorrectionsWidget(
            fit=fit,
            model=self,
            **kwargs
        )

        self.fit = fit
        super().__init__(
            fit=fit,
            parse=pw,
            icon=icon,
            convolve=self.convolve,
            generic=generic,
            corrections=corrections
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.convolve)
        layout.addWidget(generic)
        layout.addWidget(corrections)
        layout.addWidget(pw)
        self.setLayout(layout)


class LifetimeMixModelWidget(LifetimeModelWidgetBase, LifetimeMixModel):

    plot_classes = [
        (chisurf.plots.LinePlot, {
            'd_scalex': 'lin',
            'd_scaley': 'log',
            'r_scalex': 'lin',
            'r_scaley': 'lin',
            'x_label': 'x',
            'y_label': 'y',
            'plot_irf': True}
         ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.DistributionPlot, {}),
        (chisurf.plots.ParameterScanPlot, {})
    ]

    @property
    def current_model_idx(self) -> int:
        return int(self._current_model.value())

    @current_model_idx.setter
    def current_model_idx(self, v: int):
        self._current_model.setValue(v)

    @property
    def amplitude(self) -> typing.List[float]:
        layout = self.model_layout
        re = list()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(
                    item,
                    chisurf.gui.widgets.fitting.widgets.FittingParameterWidget
            ):
                re.append(item)
        return re

    @property
    def selected_fit(self):
        i = self.model_selector.currentIndex()
        return self.model_types[i]

    def __init__(self, fit, **kwargs):
        super().__init__(self, fit, **kwargs)
        # LifetimeModelWidgetBase.__init__(self, fit, **kwargs)
        # LifetimeMixModel.__init__(self, fit, **kwargs)

        layout = QtWidgets.QHBoxLayout()

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems([m.name for m in self.model_types])
        layout.addWidget(self.model_selector)

        self.add_button = QtWidgets.QPushButton('Add')
        layout.addWidget(self.add_button)

        self.clear_button = QtWidgets.QPushButton('Clear')
        layout.addWidget(self.clear_button)
        layout.addLayout(layout)

        self.add_button.clicked.connect(self.add_model)
        self.clear_button.clicked.connect(self.clear_models)

        self.model_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.model_layout)

        self._current_model = QtWidgets.QSpinBox()
        self._current_model.setMinimum(0)
        self._current_model.setMaximum(0)
        self.layout_parameter.addWidget(self._current_model)
        self.layout_parameter.addLayout(layout)

    def add_model(self, fit: chisurf.fitting.fit.FitGroup = None):
        layout = QtWidgets.QHBoxLayout()

        if fit is None:
            model = self.selected_fit.model
        else:
            model = fit.model

        xi = len(self) + 1
        fraction_name = f"x({xi})"
        label_text = f"x<sub>{xi}</sub>"
        fraction = chisurf.gui.widgets.fitting.widgets.FittingParameterWidget(
            name=fraction_name,
            value=1.0,
            model=self,
            ub=1.0,
            lb=0.0,
            label_text=label_text,
            layout=layout
        )
        layout.addWidget(fraction)
        model_label = QtWidgets.QLabel(fit.name)
        layout.addWidget(model_label)

        self.model_layout.addLayout(layout)
        self.append(model, fraction)

    def clear_models(self):
        LifetimeMixModel.clear_models(self)
        chisurf.gui.widgets.clear_layout(self.model_layout)

