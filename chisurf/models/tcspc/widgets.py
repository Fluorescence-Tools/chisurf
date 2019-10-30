"""

"""
from __future__ import annotations

from qtpy import QtWidgets, uic, QtCore, QtGui

import os

import chisurf
import chisurf.fitting
import chisurf.settings as mfm
import chisurf.decorators
import chisurf.math
import chisurf.models
from chisurf import plots
from chisurf.fitting.parameter import FittingParameter
from chisurf.fitting.widgets import FittingControllerWidget
from chisurf.models import parse
from chisurf.models.model import ModelWidget
from chisurf.models.tcspc.anisotropy import Anisotropy
from chisurf.models.tcspc.fret import Gaussians, DiscreteDistance, \
    GaussianModel, FRETrateModel, \
    WormLikeChainModel, SingleDistanceModel
from chisurf.models.tcspc.lifetime import Lifetime, LifetimeModel
from chisurf.models.tcspc.mix_model import LifetimeMixModel
from chisurf.models.tcspc.nusiance import Convolve, Corrections, Generic
from chisurf.models.parse.tcspc.tcspc_parse import ParseDecayModel
from chisurf.models.tcspc.pddem import PDDEM, PDDEMModel
from chisurf.widgets import clear_layout
from chisurf.experiments.widgets import ExperimentalDataSelector


class ConvolveWidget(Convolve, QtWidgets.QWidget):
    """

    """

    @chisurf.decorators.init_with_ui(ui_filename="convolveWidget.ui")
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            hide_curve_convolution: bool = True,
            *args,
            **kwargs
    ):
        """

        :param fit:
        :param hide_curve_convolution:
        :param kwargs:
        """
        if hide_curve_convolution:
            self.radioButton_3.setVisible(not hide_curve_convolution)
        layout = QtWidgets.QHBoxLayout()
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._dt,
            layout=layout,
            fixed=True,
            hide_bounds=True
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._n0,
            layout=layout,
            fixed=True,
            hide_bounds=True
        )
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._start,
            layout=layout
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._stop,
            layout=layout
        )
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._lb,
            layout=layout
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._ts,
            layout=layout
        )
        self.verticalLayout_2.addLayout(layout)

        chisurf.fitting.widgets.make_fitting_parameter_widget(
            fitting_parameter=self._rep,
            layout=self.horizontalLayout_3,
            text='r[MHz]'
        )

        self.irf_select = ExperimentalDataSelector(
            parent=None,
            change_event=self.change_irf,
            fit=self.fit,
            setup=chisurf.experiments.tcspc.TCSPCReader
        )

        self.actionSelect_IRF.triggered.connect(self.irf_select.show)
        self.radioButton_3.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton_2.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton.clicked.connect(self.onConvolutionModeChanged)
        self.groupBox.toggled.connect(self.onConvolutionModeChanged)

    def onConvolutionModeChanged(self):
        chisurf.run(
            "\n".join(
                [
                    "for f in cs.current_fit:\n"
                    "   f.model.convolve.reading_routine = '%s'\n" % self.gui_mode,
                    "cs.current_fit.model.convolve.do_convolution = %s" %
                    self.groupBox.isChecked(),
                    "cs.current_fit.update()"
                ]
            )
        )

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        name = self.irf_select.curve_name
        chisurf.run(
            "chisurf.macros.tcspc.change_irf(%s, '%s')" % (idx, name)
        )
        self.fwhm = self._irf.fwhm

    @property
    def fwhm(
            self
    ) -> float:
        return self._irf.fwhm

    @fwhm.setter
    def fwhm(
            self,
            v: float
    ):
        self.lineEdit_2.setText("%.3f" % v)

    @property
    def gui_mode(self):
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"


class CorrectionsWidget(
    Corrections,
    QtWidgets.QWidget
):

    @chisurf.decorators.init_with_ui(
        ui_filename="tcspcCorrections.ui"
    )
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            hide_corrections: bool = False,
            threshold = 0.9,
            reverse = False,
            enabled = False,
            **kwargs
    ):
        """

        :param fit:
        :param hide_corrections:
        :param kwargs:
        """
        self.groupBox.setChecked(False)
        self.comboBox.addItems(chisurf.math.signal.window_function_types)
        if hide_corrections:
            self.hide()

        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._dead_time,
            layout=self.horizontalLayout_2,
            text='t<sub>dead</sub>[ns]'
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._window_length,
            layout=self.horizontalLayout_2,
            text='t<sub>dead</sub>[ns]'
        )

        self.lin_select = ExperimentalDataSelector(
            parent=None,
            change_event=self.onChangeLin,
            fit=fit,
            setup=chisurf.experiments.tcspc.tcspc.TCSPCReader
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
            "chisurf.macros.tcspc.set_linearization(%s, '%s')" %
            (idx, lin_name)
        )


class GenericWidget(
    QtWidgets.QWidget,
    Generic
):
    """

    """

    def change_bg_curve(
            self,
            background_index: int = None
    ):
        if isinstance(background_index, int):
            self.background_select.selected_curve_index = background_index
        self._background_curve = self.background_select.selected_dataset

        self.lineEdit.setText(self.background_select.curve_name)
        self.fit.model.update()

    def update_widget(self):
        self.lineedit_nphBg.setText("%i" % self.n_ph_bg)
        self.lineedit_nphFl.setText("%i" % self.n_ph_fl)

    def __init__(
            self,
            hide_generic: bool = False,
            **kwargs
    ):
        """

        :param hide_generic:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.parent = kwargs.get('parent', None)

        if hide_generic:
            self.hide()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        gb = QtWidgets.QGroupBox()
        gb.setTitle("Generic")
        self.layout.addWidget(gb)

        gbl = QtWidgets.QVBoxLayout()

        gb.setLayout(gbl)
        # Generic parameters
        l = QtWidgets.QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        gbl.addLayout(l)
        sc_w = chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._sc,
            text='Sc'
        )
        bg_w = chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._bg,
            text='Bg'
        )
        tmeas_bg_w = chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._tmeas_bg,
            text='t<sub>Bg</sub>'
        )
        tmeas_exp_w = chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._tmeas_exp,
            text='t<sub>Meas</sub>'
        )

        l.addWidget(sc_w, 1, 0)
        l.addWidget(bg_w, 1, 1)
        l.addWidget(tmeas_bg_w, 2, 0)
        l.addWidget(tmeas_exp_w, 2, 1)

        ly = QtWidgets.QHBoxLayout()
        l.addLayout(ly, 0, 0, 1, 2)
        ly.addWidget(QtWidgets.QLabel('Background file:'))
        self.lineEdit = QtWidgets.QLineEdit()
        ly.addWidget(self.lineEdit)

        open_bg = QtWidgets.QPushButton()
        open_bg.setText('...')
        ly.addWidget(open_bg)

        self.background_select = ExperimentalDataSelector(
            parent=None,
            change_event=self.change_bg_curve,
            fit=self.fit
        )
        open_bg.clicked.connect(self.background_select.show)

        a = QtWidgets.QHBoxLayout()
        a.addWidget(QtWidgets.QLabel('nPh(Bg)'))
        self.lineedit_nphBg = QtWidgets.QLineEdit()
        a.addWidget(self.lineedit_nphBg)
        l.addLayout(a, 3, 0, 1, 1)

        a = QtWidgets.QHBoxLayout()
        a.addWidget(QtWidgets.QLabel('nPh(Fl)'))
        self.lineedit_nphFl = QtWidgets.QLineEdit()
        a.addWidget(self.lineedit_nphFl)
        l.addLayout(a, 3, 1, 1, 1)


class AnisotropyWidget(
    Anisotropy,
    QtWidgets.QWidget
):
    """

    """

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setTitle("Rotational-times")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
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

        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._r0,
            text='r0',
            layout=layout,
            fixed=True
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._g,
            text='g',
            layout=layout,
            fixed=True
        )
        self.lh.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._l1,
            text='l1',
            layout=layout,
            fixed=True,
            decimals=4
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._l2,
            text='l2',
            layout=layout,
            fixed=True,
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
        Anisotropy.add_rotation(self, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.lh.addLayout(layout)
        self._rho_widgets.append(
            chisurf.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._rhos[-1],
                decimals=2,
                layout=layout
            )
        )
        self._b_widgets.append(
            chisurf.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._bs[-1],
                decimals=2,
                layout=layout
            )
        )

    def remove_rotation(self):
        self._rhos.pop()
        self._bs.pop()
        self._rho_widgets.pop().close()
        self._b_widgets.pop().close()


class PDDEMWidget(QtWidgets.QWidget, PDDEM):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        PDDEM.__init__(self, **kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "pddem.ui"
            ),
            self
        )

        layout = QtWidgets.QHBoxLayout()
        self._fAB = self._fAB.make_widget(layout=layout, text='A>B')
        self._fBA = self._fBA.make_widget(layout=layout, text='B>A')
        self.verticalLayout_3.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        self._pA = self._pA.make_widget(layout=layout)
        self._pB = self._pB.make_widget(layout=layout)
        self.verticalLayout_3.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        self._pxA = self._pxA.make_widget(layout=layout, text='Ex<sub>A</sub>')
        self._pxB = self._pxB.make_widget(layout=layout, text='Ex<sub>B</sub>')
        self.verticalLayout_3.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        self._pmA = self._pmA.make_widget(layout=layout, text='Em<sub>A</sub>')
        self._pmB = self._pmB.make_widget(layout=layout, text='Em<sub>B</sub>')
        self.verticalLayout_3.addLayout(layout)


class PDDEMModelWidget(ModelWidget, PDDEMModel):

    plot_classes = [
        (plots.LinePlot, {
            'd_scalex': 'lin',
            'd_scaley': 'log',
            'r_scalex': 'lin',
            'r_scaley': 'lin',
            'x_label': 'x',
            'y_label': 'y',
            'plot_irf': True}
         ),
        (plots.FitInfo, {}),
        (plots.DistributionPlot, {}),
        (plots.ParameterScanPlot, {})
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

        self.layout.addWidget(self.fa)
        self.layout.addWidget(self.fb)
        self.layout.addWidget(self.gaussians)
        self.layout.addWidget(self.anisotropy)
        self.layout.addWidget(self.corrections)


class LifetimeWidget(Lifetime, QtWidgets.QWidget):

    def update(self, *__args):
        QtWidgets.QWidget.update(self, *__args)
        Lifetime.update(self)

    def read_values(self, target):

        def linkcall():
            for key in self.parameter_dict:
                v = target.parameters_all_dict[key].value
                chisurf.run(
                    "cs.current_fit.model.parameters_all_dict['%s'].value = %s" %
                    (key, v)
                )
            chisurf.run("cs.current_fit.update()")
        return linkcall

    def read_menu(self):
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.read_values(a))
                menu.addMenu(submenu)
        self.readFrom.setMenu(menu)

    def link_values(self, target):
        def linkcall():
            self._link = target
            self.setEnabled(False)
        return linkcall

    def link_menu(self):
        menu = QtWidgets.QMenu()
        for f in chisurf.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.link_values(a))
                menu.addMenu(submenu)
        self.linkFrom.setMenu(menu)

    def __init__(
            self,
            title: str = '',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setTitle(title)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
        self._amp_widgets = list()
        self._lifetime_widgets = list()

        lh = QtWidgets.QHBoxLayout()
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(0)

        addDonor = QtWidgets.QPushButton()
        addDonor.setText("add")
        addDonor.clicked.connect(self.onAddLifetime)
        lh.addWidget(addDonor)

        removeDonor = QtWidgets.QPushButton()
        removeDonor.setText("del")
        removeDonor.clicked.connect(self.onRemoveLifetime)
        lh.addWidget(removeDonor)

        readFrom = QtWidgets.QToolButton()
        readFrom.setText("read")
        readFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        readFrom.clicked.connect(self.read_menu)
        lh.addWidget(readFrom)
        self.readFrom = readFrom

        linkFrom = QtWidgets.QToolButton()
        linkFrom.setText("link")
        linkFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        linkFrom.clicked.connect(self.link_menu)
        lh.addWidget(linkFrom)
        self.linkFrom = linkFrom

        normalize_amplitude = QtWidgets.QCheckBox("Norm.")
        normalize_amplitude.setChecked(True)
        normalize_amplitude.setToolTip(
            "Normalize amplitudes to unity.\nThe sum of all amplitudes equals one."
        )
        normalize_amplitude.clicked.connect(self.onNormalizeAmplitudes)
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtWidgets.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip(
            "Take absolute value of amplitudes\nNo negative amplitudes"
        )
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
        self.absolute_amplitude = absolute_amplitude

        lh.addWidget(absolute_amplitude)
        lh.addWidget(normalize_amplitude)
        self.lh.addLayout(lh)

        self.append()

    def onNormalizeAmplitudes(self):
        chisurf.run(
            "chisurf.macros.tcspc.normalize_lifetime_amplitudes(%s)",
            self.normalize_amplitude.isChecked()
        )

    def onAbsoluteAmplitudes(self):
        chisurf.run(
            "chisurf.macros.tcspc.absolute_amplitudes(%s)",
            self.absolute_amplitude.isChecked()
        )

    def onAddLifetime(self):
        chisurf.run("chisurf.macros.tcspc.add_lifetime('%s')" % self.name)

    def onRemoveLifetime(self):
        chisurf.run("chisurf.macros.tcspc.remove_lifetime('%s')" % self.name)

    def append(self, *args, **kwargs):
        Lifetime.append(self, *args, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        #amplitude = self._amplitudes[-1].make_widget(layout=layout)
        #self._amp_widgets.append(amplitude)

        self._amp_widgets.append(
            chisurf.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=layout
            )
        )

        self._lifetime_widgets.append(
            chisurf.fitting.widgets.make_fitting_parameter_widget(
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

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            icon: QtGui.QIcon = None,
            hide_nuisances: bool = False,
            **kwargs
    ):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        super().__init__(
            fit=fit,
            icon=icon
        )

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

        LifetimeModel.__init__(
            self,
            fit
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

        self.generic = generic
        self.corrections = corrections
        self.convolve = convolve


class LifetimeModelWidget(LifetimeModelWidgetBase):

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            **kwargs
    ):
        super().__init__(
            fit=fit,
            **kwargs
        )
        self.lifetimes = LifetimeWidget(
            name='lifetimes',
            parent=self,
            title='Lifetimes',
            short='L',
            fit=fit
        )
        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            **kwargs
        )
        self.anisotropy = anisotropy
        self.layout.addWidget(self.lifetimes)
        self.layout.addWidget(anisotropy)


class GaussianWidget(Gaussians, QtWidgets.QWidget):

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
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("Gaussian distances")
        self.lh = QtWidgets.QVBoxLayout()
        self.gb.setLayout(self.lh)

        self._gb = list()

        self.grid_layout = QtWidgets.QGridLayout()

        layout = QtWidgets.QHBoxLayout()
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
            "for f in cs.current_fit:\n" \
            "   f.model.%s.append()\n" \
            "   f.model.update()" % self.name
        )

    def onRemoveGaussian(self):
        chisurf.run(
            "for f in cs.current_fit:\n" \
            "   f.model.%s.pop()\n" \
            "   f.model.update()" % self.name
        )

    def append(self, *args, **kwargs):
        super().append(
            50.0,
            6.0,
            1.0,
        )
        gb = QtWidgets.QGroupBox()
        n_gauss = len(self)
        gb.setTitle('G%i' % n_gauss)
        layout = QtWidgets.QVBoxLayout()

        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianMeans[-1],
            layout=layout
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianSigma[-1],
            layout=layout
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianShape[-1],
            layout=layout
        )
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._gaussianAmplitudes[-1],
            layout=layout
        )

        gb.setLayout(layout)
        row = (n_gauss - 1) // 2 + 1
        col = (n_gauss - 1) % 2
        self.grid_layout.addWidget(
            gb,
            row,
            col
        )
        self._gb.append(gb)

    def pop(self) -> None:
        #self._gaussianMeans.pop().close()
        #self._gaussianSigma.pop().close()
        #self._gaussianAmplitudes.pop().close()
        #self._gaussianShape.pop().close()
        self._gb.pop().close()


class DiscreteDistanceWidget(
    DiscreteDistance,
    QtWidgets.QWidget
):

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
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("FRET-rates")
        self.lh = QtWidgets.QVBoxLayout()
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
        t = """
for f in cs.current_fit:
    f.model.%s.append()
            """ % self.name
        chisurf.run(t)

    def onRemoveFRETrate(self):
        t = """
for f in cs.current_fit:
    f.model.%s.pop()
            """ % self.name
        chisurf.run(t)

    def append(
            self,
            x=None,
            distance=None,
            update=True
    ):
        x = 1.0 if x is None else x
        m = 50.0 if distance is None else distance
        gb = QtWidgets.QGroupBox()
        n_rates = len(self)
        gb.setTitle('G%i' % (n_rates + 1))
        layout = QtWidgets.QVBoxLayout()
        pm = FittingParameter(
            name='R(%s,%i)' % (self.short, n_rates + 1),
            value=m,
            model=self.model,
            decimals=1,
            bounds_on=False,
            lb=chisurf.settings.fret['rda_min'],
            ub=chisurf.settings.fret['rda_max'],
            text='R',
            update_function=self.update
        )
        px = FittingParameter(
            name='x(%s,%i)' % (self.short, n_rates + 1),
            value=x,
            model=self.model,
            decimals=3,
            bounds_on=False,
            text='x',
            update_function=self.update
        )
        m = chisurf.fitting.widgets.make_fitting_parameter_widget(
            pm,
            layout=layout
        )
        x = chisurf.fitting.widgets.make_fitting_parameter_widget(
            px,
            layout=layout
        )

        gb.setLayout(layout)
        row = n_rates / 2
        col = n_rates % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)
        self._distances.append(m)
        self._amplitudes.append(x)
        chisurf.run("cs.current_fit.update()")

    def pop(self):
        self._distances.pop().close()
        self._amplitudes.pop().close()
        self._gb.pop().close()
        chisurf.run("cs.current_fit.update()")


class GaussianModelWidget(
    GaussianModel,
    LifetimeModelWidgetBase
):

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            **kwargs
    ):
        donors = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)'
        )
        gaussians = GaussianWidget(
            donors=donors,
            parent=self,
            model=self,
            short='G',
            **kwargs
        )
        GaussianModel.__init__(
            self,
            fit=fit,
            lifetimes=donors,
            gaussians=gaussians
        )

        LifetimeModelWidgetBase.__init__(
            self,
            fit=fit,
            **kwargs
        )
        self.lifetimes = donors

        self.layout_parameter.addWidget(donors)

        self.layout_parameter.addWidget(
            chisurf.fitting.widgets.make_fitting_parameter_group_widget(
                self.fret_parameters
            )
        )

        self.layout_parameter.addWidget(gaussians)


class FRETrateModelWidget(
    FRETrateModel,
    LifetimeModelWidgetBase
):

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            **kwargs
    ):
        donors = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)'
        )
        fret_rates = DiscreteDistanceWidget(
            donors=donors,
            parent=self,
            model=self,
            short='G',
            **kwargs
        )
        FRETrateModel.__init__(
            self,
            fit=fit,
            lifetimes=donors,
            fret_rates=fret_rates
        )
        LifetimeModelWidgetBase.__init__(
            self,
            fit=fit,
            **kwargs
        )
        self.lifetimes = donors

        self.layout_parameter.addWidget(donors)
        # self.layout_parameter.addWidget(self.fret_parameters.to_widget())
        self.layout_parameter.addWidget(
            chisurf.fitting.widgets.make_fitting_parameter_group_widget(
                self.fret_parameters
            )
        )
        self.layout_parameter.addWidget(fret_rates)


class WormLikeChainModelWidget(
    WormLikeChainModel,
    LifetimeModelWidgetBase
):

    @property
    def use_dye_linker(
            self
    ) -> bool:
        return bool(self._use_dye_linker.isChecked())

    @use_dye_linker.setter
    def use_dye_linker(
            self,
            v: bool
    ):
        self._use_dye_linker.setChecked(v)

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            **kwargs
    ):
        donors = LifetimeWidget(
            parent=self,
            model=self,
            title='Donor(0)',
            name='donors'
        )
        WormLikeChainModel.__init__(
            self,
            fit=fit,
            lifetimes=donors,
            **kwargs
        )

        LifetimeModelWidgetBase.__init__(
            self,
            fit,
            **kwargs
        )
        self.lifetimes = donors

        layout = QtWidgets.QHBoxLayout()
        self._use_dye_linker = QtWidgets.QCheckBox()
        self._use_dye_linker.setText('Use linker')
        layout.addWidget(self._use_dye_linker)

        #self._sigma_linker = self._sigma_linker.make_widget(layout=layout)
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._sigma_linker,
            layout=layout
        )

        self.layout_parameter.addWidget(self.fret_parameters.to_widget())
        self.layout_parameter.addWidget(donors)
        self.layout_parameter.addLayout(layout)

        #self._chain_length = self._chain_length.make_widget(layout=self.layout_parameter)
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._chain_length,
            layout=layout
        )
        #self._persistence_length = self._persistence_length.make_widget(layout=self.layout_parameter)
        chisurf.fitting.widgets.make_fitting_parameter_widget(
            self._persistence_length,
            layout=layout
        )

        self.convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        self.donors = LifetimeWidget(parent=self, model=self, title='Donor(0)')
        self.generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        self.fitting_widget = QtWidgets.QLabel() if kwargs.get('disable_fit', False) else FittingControllerWidget(fit=fit, **kwargs)
        self.corrections = CorrectionsWidget(fit, model=self, **kwargs)

        ModelWidget.__init__(
            self,
            fit=fit,
            icon=QtGui.QIcon(":/icons/icons/TCSPC.png"),
            **kwargs
        )

        SingleDistanceModel.__init__(
            self,
            fit=fit,
            convolve=self.convolve,
            corrections=self.corrections,
            generic=self.generic,
            lifetimes=self.donors,
            anisotropy=self.anisotropy
        )

        self._donly = self._donly.make_widget()

        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "load_distance_distibution.ui"
            ),
            self
        )

        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.actionOpen_distirbution.triggered.connect(self.load_distance_distribution)

        self.verticalLayout.addWidget(self.fitting_widget)
        self.verticalLayout.addWidget(self.convolve)
        self.verticalLayout.addWidget(self.generic)
        self.verticalLayout.addWidget(self._donly)
        self.verticalLayout.addWidget(self.donors)
        self.verticalLayout.addWidget(self.anisotropy)
        self.verticalLayout.addWidget(self.corrections)
        self.verticalLayout.addWidget(self.errors)

    def load_distance_distribution(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        #print "load_distance_distribution"
        verbose = kwargs.get('verbose', self.verbose)
        #filename = kwargs.get('filename', str(QtGui.QFileDialog.getOpenFileName(self, 'Open File')))
        filename = chisurf.widgets.get_filename('Open distance distribution', 'CSV-files (*.csv)')
        self.lineEdit.setText(filename)
        csv = chisurf.fio.ascii.Csv(filename)
        ar = csv.data.T
        if verbose:
            print("Opening distribution")
            print("Filename: %s" % filename)
            print("Shape: %s" % ar.shape)
        self.rda = ar[0]
        self.prda = ar[1]
        self.update_model()


class ParseDecayModelWidget(ParseDecayModel, ModelWidget):

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            **kwargs
    ):
        ModelWidget.__init__(self, icon=QtGui.QIcon(":/icons/icons/TCSPC.ico"))

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

        fn = os.path.join(mfm.package_directory, 'settings/tcspc.models.json')
        pw = parse.ParseFormulaWidget(
            fit=fit,
            model=self,
            model_file=fn
        )
        corrections = chisurf.models.tcspc.widgets.CorrectionsWidget(fit, model=self, **kwargs)

        self.fit = fit
        ParseDecayModel.__init__(self, fit=fit, parse=pw, convolve=self.convolve,
                                 generic=generic, corrections=corrections)
        fitting_widget = FittingControllerWidget(fit=fit, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(fitting_widget)
        layout.addWidget(self.convolve)
        layout.addWidget(generic)
        layout.addWidget(pw)
        layout.addWidget(corrections)
        self.setLayout(layout)


class LifetimeMixModelWidget(LifetimeModelWidgetBase, LifetimeMixModel):

    plot_classes = [
        (plots.LinePlot, {
            'd_scalex': 'lin',
            'd_scaley': 'log',
            'r_scalex': 'lin',
            'r_scaley': 'lin',
            'x_label': 'x',
            'y_label': 'y',
            'plot_irf': True
        }
         )
        , (plots.FitInfo, {})
    ]

    @property
    def current_model_idx(self):
        return int(self._current_model.value())

    @current_model_idx.setter
    def current_model_idx(self, v):
        self._current_model.setValue(v)

    @property
    def amplitude(self):
        layout = self.model_layout
        re = list()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if isinstance(
                    item,
                    chisurf.fitting.widgets.FittingParameterWidget
            ):
                re.append(item)
        return re

    @property
    def selected_fit(self):
        i = self.model_selector.currentIndex()

        return self.model_types[i]

    def __init__(self, fit, **kwargs):
        LifetimeModelWidgetBase.__init__(self, fit, **kwargs)
        LifetimeMixModel.__init__(self, fit, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)

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

    def add_model(
            self,
            fit: chisurf.fitting.fit.FitGroup = None
    ):
        layout = QtWidgets.QHBoxLayout()

        if fit is None:
            model = self.selected_fit.model
        else:
            model = fit.model

        fraction_name = "x(%s)" % (len(self) + 1)
        fraction = chisurf.fitting.widgets.FittingParameterWidget(
            name=fraction_name,
            value=1.0,
            model=self,
            ub=1.0,
            lb=0.0,
            layout=layout
        )
        layout.addWidget(fraction)
        model_label = QtWidgets.QLabel(fit.name)
        layout.addWidget(model_label)

        self.model_layout.addLayout(layout)
        self.append(model, fraction)

    def clear_models(self):
        LifetimeMixModel.clear_models(self)
        clear_layout(self.model_layout)