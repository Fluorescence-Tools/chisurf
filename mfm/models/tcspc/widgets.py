from PyQt5 import QtWidgets, uic, QtCore

import mfm
from mfm.models.tcspc.anisotropy import Anisotropy
from mfm.models.tcspc.nusiance import Convolve, Corrections, Generic
from mfm.widgets import CurveSelector


class ConvolveWidget(Convolve, QtWidgets.QWidget):

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit,
            hide_curve_convolution: bool = True,
            **kwargs
    ):
        """

        :param fit:
        :param hide_curve_convolution:
        :param kwargs:
        """
        Convolve.__init__(self, fit, **kwargs)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('mfm/ui/fitting/models/tcspc/convolveWidget.ui', self)

        if hide_curve_convolution:
            self.radioButton_3.setVisible(not hide_curve_convolution)

        l = QtWidgets.QHBoxLayout()
        mfm.fitting.widgets.make_fitting_parameter_widget(self._dt, layout=l, fixed=True, hide_bounds=True)
        mfm.fitting.widgets.make_fitting_parameter_widget(self._n0, layout=l, fixed=True, hide_bounds=True)
        self.verticalLayout_2.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        mfm.fitting.widgets.make_fitting_parameter_widget(self._start, layout=l)
        mfm.fitting.widgets.make_fitting_parameter_widget(self._stop, layout=l)
        self.verticalLayout_2.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        mfm.fitting.widgets.make_fitting_parameter_widget(self._lb, layout=l)
        mfm.fitting.widgets.make_fitting_parameter_widget(self._ts, layout=l)
        self.verticalLayout_2.addLayout(l)

        self._rep.make_widget(layout=self.horizontalLayout_3, text='r[MHz]')

        self.irf_select = CurveSelector(parent=None,
                                        change_event=self.change_irf,
                                        fit=self.fit,
                                        setup=mfm.experiments.tcspc.TCSPCReader)
        self.actionSelect_IRF.triggered.connect(self.irf_select.show)

        self.radioButton_3.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton_2.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton.clicked.connect(self.onConvolutionModeChanged)
        self.groupBox.toggled.connect(self.onDoConvolutionChanged)

    def onConvolutionModeChanged(self):
        t = "for f in cs.current_fit:\n" \
            "   f.models.convolve.mode = '%s'\n" % self.gui_mode
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onDoConvolutionChanged(self):
        mfm.run("cs.current_fit.models.convolve.do_convolution = %s" % self.groupBox.isChecked())

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        name = self.irf_select.curve_name
        mfm.run("mfm.cmd.change_irf(%s, '%s')" % (idx, name))
        self.fwhm = self._irf.fwhm

    @property
    def fwhm(self) -> float:
        return self._irf.fwhm

    @fwhm.setter
    def fwhm(
            self,
            v: float
    ):
        self._fwhm = v
        self.lineEdit_2.setText("%.3f" % v)

    @property
    def gui_mode(self):
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"


class CorrectionsWidget(Corrections, QtWidgets.QWidget):

    def __init__(self, fit, **kwargs):
        super(CorrectionsWidget, self).__init__(fit=fit, threshold=0.9, reverse=False, enabled=False)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi("mfm/ui/fitting/models/tcspc/tcspcCorrections.ui", self)
        self.groupBox.setChecked(False)
        self.comboBox.addItems(mfm.math.signal.window_function_types)
        if kwargs.get('hide_corrections', False):
            self.hide()

        mfm.fitting.widgets.make_fitting_parameter_widget(self._dead_time, layout=self.horizontalLayout_2, text='t<sub>dead</sub>[ns]')
        mfm.fitting.widgets.make_fitting_parameter_widget(self._window_length, layout=self.horizontalLayout_2, text='t<sub>dead</sub>[ns]')

        self.lin_select = CurveSelector(parent=None,
                                        change_event=self.onChangeLin,
                                        fit=self.fit,
                                        setup=mfm.experiments.tcspc.TCSPCReader)

        self.actionSelect_lintable.triggered.connect(self.lin_select.show)

        self.checkBox_3.toggled.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.correct_pile_up = %s\n" % self.checkBox_3.isChecked())
        )

        self.checkBox_2.toggled.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.reverse = %s" % self.checkBox_2.isChecked())
        )

        self.checkBox.toggled.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.correct_dnl = %s" % self.checkBox.isChecked())
        )

        self.comboBox.currentIndexChanged.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.window_function = '%s'" % self.comboBox.currentText())
        )

    def onChangeLin(self):
        idx = self.lin_select.selected_curve_index
        t = "lin_table = cs.current_fit.models.corrections.lin_select.datasets[%s]\n" \
            "for f in cs.current_fit[cs.current_fit._selected_fit:]:\n" \
            "   f.models.corrections.lintable = mfm.curve.DataCurve(x=lin_table.x, y=lin_table.y)\n" \
            "   f.models.corrections.correct_dnl = True\n" % idx
        mfm.run(t)

        lin_name = self.lin_select.curve_name
        t = "for f in cs.current_fit[cs.current_fit._selected_fit:]:\n" \
            "   f.models.corrections.lineEdit.setText('%s')\n" \
            "   f.models.corrections.checkBox.setChecked(True)\n" % lin_name
        mfm.run(t)

        mfm.run("cs.current_fit.update()")


class GenericWidget(QtWidgets.QWidget, Generic):

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

    def __init__(self, **kwargs):
        super(GenericWidget, self).__init__(**kwargs)

        self.parent = kwargs.get('parent', None)
        if kwargs.get('hide_generic', False):
            self.hide()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        gb = QtWidgets.QGroupBox()
        gb.setTitle("Generic")
        self.layout.addWidget(gb)

        gbl = QtWidgets.QVBoxLayout()

        gb.setLayout(gbl)
        # Generic parameters
        l = QtWidgets.QGridLayout()
        gbl.addLayout(l)

        sc_w = self._sc.make_widget(update_function=self.update_widget, text='Sc')
        bg_w = self._bg.make_widget(update_function=self.update_widget, text='Bg')
        tmeas_bg_w = self._tmeas_bg.make_widget(update_function=self.update_widget, text='t<sub>Bg</sub>')
        tmeas_exp_w = self._tmeas_exp.make_widget(update_function=self.update_widget, text='t<sub>Meas</sub>')

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

        self.background_select = CurveSelector(parent=None, change_event=self.change_bg_curve, fit=self.fit)
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


class AnisotropyWidget(Anisotropy, QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        Anisotropy.__init__(self, **kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setTitle("Rotational-times")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
        self.rot_vis = False
        self._rho_widgets = list()
        self._b_widgets = list()

        self.radioButtonVM = QtWidgets.QRadioButton("VM")
        self.radioButtonVM.setToolTip("Excitation: Vertical\nDetection: Magic-Angle")
        self.radioButtonVM.setChecked(True)
        self.radioButtonVM.clicked.connect(lambda: mfm.run("cs.current_fit.models.anisotropy.polarization_type = 'vm'"))
        self.radioButtonVM.clicked.connect(self.hide_roation_parameters)

        self.radioButtonVV = QtWidgets.QRadioButton("VV")
        self.radioButtonVV.setToolTip("Excitation: Vertical\nDetection: Vertical")
        self.radioButtonVV.clicked.connect(lambda: mfm.run("cs.current_fit.models.anisotropy.polarization_type = 'vv'"))

        self.radioButtonVH = QtWidgets.QRadioButton("VH")
        self.radioButtonVH.setToolTip("Excitation: Vertical\nDetection: Horizontal")
        self.radioButtonVH.clicked.connect(lambda: mfm.run("cs.current_fit.models.anisotropy.polarization_type = 'vh'"))

        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)

        add_rho = QtWidgets.QPushButton()
        add_rho.setText("add")
        l.addWidget(add_rho)
        add_rho.clicked.connect(self.onAddRotation)

        remove_rho = QtWidgets.QPushButton()
        remove_rho.setText("del")
        l.addWidget(remove_rho)
        remove_rho.clicked.connect(self.onRemoveRotation)

        spacerItem = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        l.addItem(spacerItem)

        l.addWidget(self.radioButtonVM)
        l.addWidget(self.radioButtonVV)
        l.addWidget(self.radioButtonVH)

        self.lh.addLayout(l)

        self.gb = QtWidgets.QGroupBox()
        self.lh.addWidget(self.gb)
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)

        l = QtWidgets.QHBoxLayout()
        mfm.fitting.widgets.make_fitting_parameter_widget(self._r0, text='r0', layout=l, fixed=True)
        mfm.fitting.widgets.make_fitting_parameter_widget(self._g, text='r0', layout=l, fixed=True)
        self.lh.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        mfm.fitting.widgets.make_fitting_parameter_widget(self._l1, text='r0', layout=l, fixed=True, decimals=4)
        mfm.fitting.widgets.make_fitting_parameter_widget(self._l2, text='r0', layout=l, fixed=True, decimals=4)
        self.lh.addLayout(l)

        self.lh.addLayout(l)
        self.add_rotation()
        self.hide_roation_parameters()

    def hide_roation_parameters(self):
        self.rot_vis = not self.rot_vis
        if self.rot_vis:
            self.gb.show()
        else:
            self.gb.hide()

    def onAddRotation(self):
        t = "for f in cs.current_fit:\n" \
            "   f.models.anisotropy.add_rotation()"
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onRemoveRotation(self):
        t = "for f in cs.current_fit:\n" \
            "   f.models.anisotropy.remove_rotation()"
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def add_rotation(self, **kwargs):
        Anisotropy.add_rotation(self, **kwargs)
        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        self.lh.addLayout(l)
        rho = self._rhos[-1].make_widget(layout=l, decimals=2)
        x = self._bs[-1].make_widget(layout=l, decimals=2)
        self._rho_widgets.append(rho)
        self._b_widgets.append(x)

    def remove_rotation(self):
        self._rhos.pop()
        self._bs.pop()
        self._rho_widgets.pop().close()
        self._b_widgets.pop().close()