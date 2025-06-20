from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
import chisurf.fitting.fit

import chisurf.models.tcspc.fret as fret

from chisurf.models.tcspc.widgets.lifetime import LifetimeWidget, LifetimeModelWidgetBase
from chisurf.models.tcspc.widgets.anisotropy import AnisotropyWidget

# Import plot_cls_dist_default from the original module
from chisurf.models.tcspc.widgets import plot_cls_dist_default


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

        # Add checkbox for is_distance_between_gaussians
        checkbox_layout = QtWidgets.QHBoxLayout()
        checkbox_layout.setSpacing(0)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)

        self.distance_checkbox = QtWidgets.QCheckBox("Enable distance between gaussians")
        self.distance_checkbox.setChecked(self.is_distance_between_gaussians)
        self.distance_checkbox.toggled.connect(
            lambda checked: self.update_distance_between_gaussians(checked)
        )

        checkbox_layout.addWidget(self.distance_checkbox)
        self.lh.addLayout(checkbox_layout)

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
        gb.setTitle(f'G{n_gauss}')

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

    def update_distance_between_gaussians(self, checked: bool):
        """
        Update the is_distance_between_gaussians property and update the model

        Parameters
        ----------
        checked : bool
            Whether the checkbox is checked
        """
        setattr(self, 'is_distance_between_gaussians', checked)
        # Update the model
        chisurf.run(
            f"for f in cs.current_fit:\n"
            f"   f.model.update()"
        )

    def pop(self) -> None:
        super().pop()
        self._gb.pop().close()


class GaussianModelWidget(fret.GaussianModel, LifetimeModelWidgetBase):

    plot_classes = plot_cls_dist_default

    def get_parameter_widgets(self):
        """
        Get all parameter widgets for this model.

        Returns
        -------
        list
            List of parameter widgets.
        """
        widgets = super().get_parameter_widgets() if hasattr(super(), 'get_parameter_widgets') else []
        widgets.append(self._orientation_widget)
        widgets.append(self._fret_parameters_widget)
        return widgets

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

        # Create parameter widgets
        self._fret_parameters_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters
        )
        self.layout.addWidget(self._fret_parameters_widget)

        # self._orientation_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
        #     self.orientation_parameter
        # )
        # self.layout.addWidget(self._orientation_widget)

        self.layout.addWidget(gaussians)

        anisotropy = AnisotropyWidget(
            name='anisotropy',
            short='rL',
            **kwargs
        )
        self.anisotropy = anisotropy
        self.layout.addWidget(self.anisotropy)