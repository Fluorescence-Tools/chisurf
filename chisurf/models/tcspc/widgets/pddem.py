from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.decorators
import chisurf.gui.widgets.fitting

from chisurf.models.tcspc.pddem import PDDEM, PDDEMModel
from chisurf.models.model import ModelWidget

# These will be imported from the new module structure
from chisurf.models.tcspc.widgets.convolve import ConvolveWidget
from chisurf.models.tcspc.widgets.corrections import CorrectionsWidget
from chisurf.models.tcspc.widgets.generic import GenericWidget
from chisurf.models.tcspc.widgets.anisotropy import AnisotropyWidget
from chisurf.models.tcspc.widgets.gaussian import GaussianWidget
from chisurf.models.tcspc.widgets.lifetime import LifetimeWidget

# Import plot_cls_dist_default from the original module
from chisurf.models.tcspc.widgets import plot_cls_dist_default


class PDDEMWidget(PDDEM, QtWidgets.QWidget):

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

    plot_classes = plot_cls_dist_default

    def __init__(self, fit, **kwargs):
        # First call super().__init__ to initialize PDDEMModel
        super().__init__(
            fit,
            icon=QtGui.QIcon(":/icons/icons/TCSPC.ico"),
            **kwargs
        )

        # Store references to the model's Lifetime objects
        model_fa = self.fa
        model_fb = self.fb
        model_donor = self.donor

        # Create LifetimeWidget objects to replace the Lifetime objects
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
        # Set donor to fa for the widget (PDDEMModel sets it to fb)
        self.donor = self.fa

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