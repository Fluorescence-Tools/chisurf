import typing


import chisurf.gui.widgets.fitting
import chisurf.plots

from chisurf.models.model import ModelWidget
from chisurf.gui import QtWidgets, QtGui, QtCore
from chisurf.models.pda.nusiance import Background
from chisurf.models.pda.simple import ProbCh0, PdaSimpleModel


class BackgroundWidget(QtWidgets.QGroupBox, Background):

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
        bg0 = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bg0,
            label_text='Bg0',
        )
        bg1 = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._bg1,
            label_text='Bg1'
        )

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(bg0, 1, 0)
        layout.addWidget(bg1, 1, 1)
        self.layout.addLayout(layout)



class ProbCh0Widget(ProbCh0, QtWidgets.QWidget):

    def update(self, *__args):
        ProbCh0.update(self)
        QtWidgets.QWidget.update(self, *__args)
        for w, v in zip(self._amp_widgets, self.amplitudes):
            w.setValue(v)

    @property
    def parameter_widgets(self):
        return self._amp_widgets + self._pch0_widgets

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
                    if isinstance(a, self.__class__):
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
                    if isinstance(a, self.__class__):
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
        self._pch0_widgets : typing.List[chisurf.gui.widgets.fitting.widgets.FittingParameterWidget] = list()

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
        chisurf.run(f"chisurf.macros.model.normalize_amplitudes('{self.name}', {self.normalize_amplitude.isChecked()})")

    def onAbsoluteAmplitudes(self):
        chisurf.run(f"chisurf.macros.model.absolute_amplitudes('{self.name}', {self.absolute_amplitude.isChecked()})")

    def onAddLifetime(self):
        chisurf.run(f"chisurf.macros.model.add_component('{self.name}')")

    def onRemoveLifetime(self):
        chisurf.run(f"chisurf.macros.model.remove_component('{self.name}')")

    def append(self, *args, **kwargs):
        ProbCh0.append(self, *args, **kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._amp_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._amplitudes[-1],
                layout=layout
            )
        )

        self._pch0_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                self._pch0[-1],
                layout=layout
            )
        )

        self.lh.addLayout(layout)

    def pop(self):
        self._amplitudes.pop()
        self._pch0.pop()
        self._amp_widgets.pop().close()
        self._pch0_widgets.pop().close()


def get_distribution(fit, kw_hist):
    pda = fit.model.pda
    histogram_function = kw_hist.pop('histogram_function', lambda ch1, ch2: ch1 / max(1, ch2))
    pda.histogram_function = histogram_function
    s1s2_experimental = fit.data.pda['s1s2']
    model_x, model_y = pda.get_1dhistogram(
        s1s2=pda.get_S1S2_matrix().flatten(),
        **kw_hist
    )
    data_x, data_y = pda.get_1dhistogram(
        s1s2=s1s2_experimental.flatten(),
        **kw_hist
    )
    return [data_y, data_x], [model_y, model_x]


class PdaSimpleModelWidget(ModelWidget, PdaSimpleModel):

    plot_classes = [
        (
            chisurf.plots.DistributionPlot,
            {
                'distribution_options': {
                    'S1/(S0+S1)': {
                        'attribute': 'fit',
                        'accessor': get_distribution,
                        'accessor_kwargs': {
                            'kw_hist': {
                                "x_max": 1.0,
                                "x_min": 0.0,
                                "log_x": False,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch2 / max(1, ch2 + ch1),
                            }
                        },
                        'curve_options': {
                            'stepMode': 'right',
                            'connect': 'all',
                            'multi_curve': True,
                            'symbol': ['None', 'None'],
                            'pen': ['r', 'b']
                        }
                    },
                    'S0/S1': {
                        'attribute': 'fit',
                        'accessor': get_distribution,
                        'accessor_kwargs': {
                            'kw_hist': {
                                "x_max": 500.0,
                                "x_min": 0.01,
                                "log_x": True,
                                "n_bins": 81,
                                "n_min": 10,
                                "histogram_function": lambda ch1, ch2: ch1 / max(1, ch2),
                            }
                        },
                        'curve_options': {
                            'stepMode': 'right',
                            'connect': 'all',
                            'multi_curve': True,
                            'symbol': ['None', 'None'],
                            'pen': ['r', 'b']
                        }
                    }
                }
            }
        ),
        (chisurf.plots.ResidualPlot, {}),
        (chisurf.plots.FitInfo, {}),
        (chisurf.plots.ParameterScanPlot, {}),
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

        background = BackgroundWidget(fit=fit, **kwargs)
        pch0 = ProbCh0Widget(fit=fit, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        ## add widgets
        if not hide_nuisances:
            layout.addWidget(background)
        layout.addWidget(pch0)

        self.setLayout(layout)
        self.layout = layout
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.background = background
        self.pch0 = pch0

