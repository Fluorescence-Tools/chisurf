from __future__ import annotations

import chisurf
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting

from chisurf.core.models.tcspc.pddem import PDDEM, PDDEMModel
from chisurf.gui.widgets.models.model_widget import ModelWidget

# These will be imported from the new module structure
from chisurf.gui.widgets.models.tcspc.convolve import ConvolveWidget
from chisurf.gui.widgets.models.tcspc.corrections import CorrectionsWidget
from chisurf.gui.widgets.models.tcspc.generic import GenericWidget
from chisurf.gui.widgets.models.tcspc.anisotropy import AnisotropyWidget
from chisurf.gui.widgets.models.tcspc.gaussian import GaussianWidget
from chisurf.gui.widgets.models.tcspc.lifetime import LifetimeWidget

# Import plot_cls_dist_default from the original module
from chisurf.gui.widgets.models.tcspc import plot_cls_dist_default
from chisurf.gui.widgets.models.tcspc import kappa2_helpers


class PDDEMWidget(PDDEM, QtWidgets.QWidget):

    # TODO: needs docstring
    def __init__(
            self,
            *args,
            **kwargs
    ):
        """Initialize the instance."""
        PDDEM.__init__(self, *args, **kwargs)
        parent = kwargs.get("parent")
        if isinstance(parent, QtWidgets.QWidget):
            QtWidgets.QWidget.__init__(self, parent)
        else:
            QtWidgets.QWidget.__init__(self)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setTitle("PDDEM")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.gridLayout.addLayout(self.verticalLayout_3, 5, 1, 1, 1)

        self.verticalLayout_2.addWidget(self.groupBox)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.addLayout(self.verticalLayout)

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

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._alpha_B,
            layout=layout,
            label_text='&alpha;<sub>A&rarr;B</sub>'
        )
        chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
            fitting_parameter=self._alpha_A,
            layout=layout,
            label_text='&alpha;<sub>B&rarr;A</sub>'
        )
        self.verticalLayout_3.addLayout(layout)


class PDDEMModelWidget(ModelWidget, PDDEMModel):

    plot_classes = plot_cls_dist_default

    # TODO: needs docstring
    def __init__(self, fit, **kwargs):
        """Initialize the instance."""
        anisotropy = AnisotropyWidget(model=self, short='rL', fit=fit, **kwargs)
        kwargs['anisotropy'] = anisotropy
        
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
        self.anisotropy = anisotropy
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

        self._fret_parameters_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters
        )
        self.layout.addWidget(self._fret_parameters_widget)
        kappa2_helpers.setup_kappa2_controls(self, self.layout)

        self.layout.addWidget(self.gaussians)
        self.layout.addWidget(self.anisotropy)
        self.layout.addWidget(self.corrections)

        try:
            self._install_code_badge()
        except Exception:
            pass

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.core.settings
            if not chisurf.core.settings.is_dev_mode():
                return
            if hasattr(self, '_chisurf_code_badge_installed'):
                return
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import resolve_object_source
            resolver = lambda: resolve_object_source(self)
            install_code_badge(self, resolver, corner='top-right', margin=4)
            self._chisurf_code_badge_installed = True
        except Exception:
            pass
