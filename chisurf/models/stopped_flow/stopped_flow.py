from __future__ import annotations

import json
import os

import numpy as np
from qtpy import QtWidgets, uic
from chisurf.models.parse.widget import ParseModelWidget

import chisurf.settings
import chisurf.gui.widgets.fitting.widgets
from chisurf import plots
from chisurf.models.model import Model
from chisurf.math.reaction.continuous import ReactionSystem
from chisurf.gui.widgets.fitting.widgets import FittingParameterWidget


class ParseStoppedFlowWidget(ParseModelWidget):
    """Stopped-flow model widget using a parse-based equation from a JSON file."""

    plot_classes = [
        (
            plots.LinePlot, {
                'd_scalex': 'lin',
                'd_scaley': 'lin',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
            }
        )
        # ,(plots.SurfacePlot, {})
    ]

    def __init__(self, fit):
        """Initialize the stopped-flow parse widget.

        Parameters
        ----------
        fit : chisurf.fitting.fit.Fit
            Fit object this widget belongs to.
        """
        fn = os.path.join(
            chisurf.settings.package_directory,
            'settings', 'stopped_flow.models.json'
        )
        ParseModelWidget.__init__(self, fit, model_file=fn)


class ReactionWidget(QtWidgets.QWidget, ReactionSystem, Model):
    """Widget for kinetic reaction system modeling with GUI controls."""

    name = "Reaction-System"

    plot_classes = [
        (
            plots.LinePlot, {
                'd_scalex': 'lin',
                'd_scaley': 'lin',
                'r_scalex': 'lin',
                'r_scaley': 'lin',
            }
        )
    ]

    @property
    def autoscale(self):
        """Whether the model Y values should be auto-scaled to match data."""
        return bool(self.checkBox.isChecked())

    @property
    def y_values(self):
        """Compute the model Y values from species concentrations, scaling and background.

        Returns
        -------
        numpy.ndarray
            The computed model Y values.
        """
        try:
            y = self.signal_intensity * self.scaleing.value
            if self.autoscale:
                s = self.fit.data.y[self.xmin:self.xmax].sum()
                ys = sum(y[self.xmin:self.xmax])
                y *= s / ys
            y += self.background.value
            y = np.array(y, dtype=np.float64)
            return y
        except (ValueError, IndexError):
            print("Problem with y-values")
            return np.ones(10)

    @y_values.setter
    def y_values(self, v):
        """No-op setter to satisfy the read-only property protocol."""
        pass

    @property
    def times(self):
        """Time axis (x data) within the current fit window."""
        return self.fit.data.x[self.xmin:self.xmax]

    @property
    def new_brightness_fixed(self):
        """Whether new species brightness parameters are fixed by default."""
        return bool(self.checkBox_3.isChecked())

    @property
    def new_concentration_fixed(self):
        """Whether new species concentration parameters are fixed by default."""
        return bool(self.checkBox_4.isChecked())

    @property
    def xmax(self):
        """Maximum index of the fit window."""
        return self.fitting_widget.xmax

    @xmax.setter
    def xmax(self, v):
        """Set the maximum index of the fit window."""
        self.fitting_widget.xmax = int(v)

    @property
    def xmin(self):
        """Minimum index of the fit window."""
        return self.fitting_widget.xmin

    @xmin.setter
    def xmin(self, v):
        """Set the minimum index of the fit window."""
        self.fitting_widget.xmin = int(v)

    def clear(self):
        """Clear all reactions, species, and parameter widgets."""
        ReactionSystem.clear(self)
        chisurf.gui.widgets.clear_layout(self.verticalLayout_10)
        chisurf.gui.widgets.clear_layout(self.verticalLayout_7)

    def __init__(self, **kwargs):
        """Initialize the reaction system widget.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to the parent classes, including
            the optional ``parameter`` dict with reaction/species definitions.
        """
        self.scaleing = FittingParameterWidget(
            name='scaling',
            value=1.0
        )
        self.background = FittingParameterWidget(
            name='background',
            value=0.0
        )
        self.timeshift = FittingParameterWidget(
            name='timeshift',
            value=0.0
        )

        ReactionSystem.__init__(self, **kwargs)
        parameter = kwargs.get('parameter', None)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "reaction.ui"
            ),
            self
        )
        self.actionPlot.triggered.connect(self.onPlot)
        self.actionIntegrate.triggered.connect(self.calc)
        self.actionLoad_reaction.triggered.connect(self.onLoadReaction)
        self.actionUpdate_reaction.triggered.connect(self.onUpdateReaction)
        self.actionSave_reaction.triggered.connect(self.onSaveLabelingFile)
        Model.__init__(self, **kwargs)
        self.setParameter(parameter)
        self.fitting_widget = chisurf.gui.widgets.fitting.widgets.FittingControllerWidget(
            fit=self.fit
        )
        self.verticalLayout_4.addWidget(self.fitting_widget)
        self.verticalLayout_4.addWidget(self.scaleing)
        self.verticalLayout_4.addWidget(self.background)
        self.verticalLayout_4.addWidget(self.timeshift)

    def setParameter(self, parameter):
        """Configure the reaction system from a parameter dictionary.

        Parameters
        ----------
        parameter : dict
            Dictionary with ``'reactions'`` and ``'species'`` keys.
        """
        self.clear()
        if isinstance(parameter, dict):
            reactions = parameter['reactions']
            species = parameter['species']
            for reaction in reactions:
                self.add_reaction(**reaction)
            for s in species:
                self.onAddSpecies(**s)

    def onPlot(self):
        """Calculate the reaction and generate the plot."""
        self.calc()
        self.plot()

    def onLoadReaction(self):
        """Open a file dialog and load a reaction system from JSON."""
        self.clear()
        #filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Reaction-File', '.rc.json', 'Reaction-Files (*.rc.json)'))
        filename = chisurf.gui.widgets.get_filename('Open Reaction-File', 'Reaction-Files (*.rc.json)')
        j = json.load(open(filename))
        self.setParameter(j)
        self.lineEdit_6.setText(filename)
        self.plainTextEdit.setPlainText(open(filename).read())

    def onSaveLabelingFile(self):
        """Save the current reaction definition to a JSON file."""
        txt = str(self.plainTextEdit.toPlainText())
        json_file = str(QtWidgets.QFileDialog.getSaveFileName(self, 'Save Reaction-JSON File',
                                                                  '.rc.json', 'JSON-Files (*.rc.json)'))[0]
        open(json_file, 'w').write(txt)

    def onUpdateReaction(self):
        """Update the reaction system from the text editor contents."""
        self.clear()
        txt = str(self.plainTextEdit.toPlainText())
        j = json.loads(txt)
        self.setParameter(j)

    def onAddSpecies(self, **kwargs):
        """Add a species with brightness and concentration parameters.

        Parameters
        ----------
        **kwargs
            Keyword arguments with species parameters (brightness, concentration,
            brightness_fixed, concentration_fixed, species name).
        """
        brightness = kwargs.get('brightness', 1.0)
        brightness_fixed = kwargs.get('brightness_fixed', True)
        concentration = kwargs.get('concentration', 1.0)
        concentration_fixed = kwargs.get('concentration_fixed', True)
        species_name = kwargs.get('species', '-')

        species = self.n_species + 1
        b = FittingParameterWidget(
            name="Q(%s)" % species_name,
            value=brightness,
            lb=0.0, ub=1000,
            model=self,
            fixed=brightness_fixed,
            bounds_on=True,
            hide_bounds=True
        )
        c = FittingParameterWidget(
            name="c(%s)" % species_name,
            value=concentration,
            lb=0.0, ub=1000,
            model=self,
            fixed=concentration_fixed,
            bounds_on=True,
            hide_bounds=True
        )
        self._initial_concentrations.append(c)
        self._species_brightness.append(b)
        l = QtWidgets.QHBoxLayout()
        l.addWidget(c)
        l.addWidget(b)
        self.verticalLayout_10.addLayout(l)

    def add_reaction(self, **kwargs):
        """Add a chemical reaction to the system.

        Parameters
        ----------
        **kwargs
            Keyword arguments with reaction parameters (educts, products,
            educt_stoichiometry, product_stoichometry, rate).
        """
        educts = kwargs.get('educts', 0)
        products = kwargs.get('products', 0)
        educt_stoichiometry = np.array(
            kwargs.get('educt_stoichiometry', 1), dtype=np.float64
        )
        product_stoichometry = np.array(
            kwargs.get('product_stoichometry', 1), dtype=np.float64
        )
        rate = kwargs.get('rate', 0.1)

        self.educts.append(educts)
        self.products.append(products)
        self.educts_stoichometry.append(educt_stoichiometry)
        self.products_stoichometry.append(product_stoichometry)
        v = FittingParameterWidget(
            name="k(%i)" % self.n_reactions, value=rate, model=self,
            hide_bounds=True,
            bounds_on=True, **kwargs
        )
        self.rates.append(v)
        self.verticalLayout_7.addWidget(v)
