from __future__ import annotations

import chisurf
from chisurf import typing
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.general
import chisurf.math.datatools
import chisurf.plots
import chisurf.fitting.parameter
import chisurf.fitting.fit

from chisurf.models.model import ModelWidget
from chisurf.models.tcspc.lifetime import Lifetime, LifetimeModel, LifetimeMixtureModel

# These will be imported from the new module structure
from chisurf.models.tcspc.widgets.convolve import ConvolveWidget
from chisurf.models.tcspc.widgets.corrections import CorrectionsWidget
from chisurf.models.tcspc.widgets.generic import GenericWidget
from chisurf.models.tcspc.widgets.anisotropy import AnisotropyWidget


class LifetimeWidget(Lifetime, QtWidgets.QWidget):

    def update(self, *__args):
        Lifetime.update(self)
        QtWidgets.QWidget.update(self, *__args)
        for w, v in zip(self._amp_widgets, self.amplitudes):
            w.setValue(v)
        for w, v in zip(self._lifetime_widgets, self.lifetimes):
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
        self.readFrom = readFrom
        self.readFrom_menu = QtWidgets.QMenu(self.readFrom)
        self.readFrom_menu.aboutToShow.connect(self.read_menu)
        readFrom.setMenu(self.readFrom_menu)
        readFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(readFrom)

        linkFrom = QtWidgets.QToolButton()
        linkFrom.setText("link")
        self.linkFrom = linkFrom
        self.linkFrom_menu = QtWidgets.QMenu(self.linkFrom)
        self.linkFrom_menu.aboutToShow.connect(self.link_menu)
        linkFrom.setMenu(self.linkFrom_menu)
        linkFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        lh.addWidget(linkFrom)

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

    def __setstate__(self, state):
        n_lifetime = (len(state.keys()) - 2) // 2
        for _ in range(n_lifetime):
            self.onAddLifetime()
        super().__setstate__(state)

    def onNormalizeAmplitudes(self):
        chisurf.run(f"chisurf.macros.model.normalize_amplitudes('{self.name}', {self.normalize_amplitude.isChecked()})")

    def onAbsoluteAmplitudes(self):
        chisurf.run(f"chisurf.macros.model.absolute_amplitudes('{self.name}', {self.absolute_amplitude.isChecked()})")

    def onAddLifetime(self):
        chisurf.run(f"chisurf.macros.model.add_component('{self.name}')")

    def onRemoveLifetime(self):
        chisurf.run(f"chisurf.macros.model.remove_component('{self.name}')")

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
            chisurf.plots.LinePlot,
            {
                'scale_x': 'lin',
                'd_scaley': 'log',
                'r_scaley': 'lin',
                'x_label': 'x',
                'y_label': 'y'
            }
        ),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
        (
            chisurf.plots.DistributionPlot,
            {
                'distribution_options': {
                    'Lifetime': {
                        'attribute': 'lifetime_spectrum',
                        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
                        'accessor_kwargs': {'sort': True},
                        'curve_options': {
                            'stepMode': False,
                            'connect': False,
                            'symbol': "o"
                        }
                    }
                }
            }
        ),
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
        generic = GenericWidget(fit=fit, parent=self, **kwargs)
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
    """
    A widget for displaying and manipulating fluorescence lifetime models.

    This widget extends LifetimeModelWidgetBase by adding specific components
    for working with fluorescence lifetime data, including lifetime parameters
    and anisotropy settings. It provides a graphical interface for configuring
    and visualizing fluorescence lifetime models used in time-correlated single
    photon counting (TCSPC) experiments.
    """

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
        (
            chisurf.plots.DistributionPlot,
            {
                'distribution_options': {
                    'Lifetime': {
                        'attribute': 'lifetime_spectrum',
                        'accessor': chisurf.math.datatools.interleaved_to_two_columns,
                        'accessor_kwargs': {'sort': True},
                        'curve_options': {
                            'stepMode': False,
                            'connect': False,
                            'symbol': "o"
                        }
                    }
                }
            }
        )
    ]

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
        chisurf.gui.widgets.general.clear_layout(layout)
        layout.addWidget(QtWidgets.QLabel("Fraction"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Model"), 0, 1)
        for i, (name, fraction) in enumerate(zip(self.model_names, self.fractions)):
            layout.addWidget(
                chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                    fraction,
                    label_text=''
                ),
                row, 0
            )
            layout.addWidget(QtWidgets.QLabel(name), row, 1)
            row += 1
