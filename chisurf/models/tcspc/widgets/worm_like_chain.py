from __future__ import annotations

import pathlib

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.widgets.fitting
import chisurf.fitting.fit

import chisurf.models.tcspc.fret as fret

from chisurf.models.tcspc.widgets.lifetime import LifetimeWidget, LifetimeModelWidgetBase

# Import plot_cls_dist_default from the original module
from chisurf.models.tcspc.widgets import plot_cls_dist_default


class WormLikeChainModelWidget(fret.WormLikeChainModel, LifetimeModelWidgetBase):

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
            title='Donor(0)',
            name='donor'
        )

        super().__init__(fit, **kwargs)

        # The donor widget is already set as an attribute of self at the beginning of the method
        # No need to set it again

        layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(layout)

        self._use_dye_linker = QtWidgets.QCheckBox()
        self._use_dye_linker.setText('Use linker')
        layout.addWidget(self._use_dye_linker)

        pw = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._sigma_linker)
        layout.addWidget(pw)

        # Create parameter widgets
        self._fret_parameters_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.fret_parameters)
        layout.addWidget(self._fret_parameters_widget)

        self._orientation_widget = chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_group_widget(
            self.orientation_parameter
        )
        layout.addWidget(self._orientation_widget)

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