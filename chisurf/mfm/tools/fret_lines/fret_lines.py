from __future__ import annotations
from typing import Tuple

import sys

from qtpy import QtWidgets
from guiqwt.builder import make
from guiqwt.plot import CurveDialog
import qdarkstyle

import numpy as np

import mfm
import mfm.decorators
import mfm.experiments
import mfm.experiments.data
import mfm.models.tcspc
import mfm.models.tcspc as fret_models
import mfm.models.tcspc.lifetime
import mfm.models.tcspc.widgets
#from mfm.models import WormLikeChainModelWidget


class FRETLineGenerator(object):

    """Calculates FRET-lines for arbitrary model

    :param kwargs:

    Attributes
    ----------
    parameter_name : str
        The parameter name of the variable which is used to generate the FRET-line.
    fret_species_averaged_lifetime : float
        The species averages lifetime of the model.
    fret_fluorescence_averaged_lifetime : float
        The fluorescence averaged lifetime of the model.
    donor_species_averaged_lifetime : float
        The species averaged lifetime of the donor of the model.
    transfer_efficiency : float
        The transfer efficiency of the model.
    parameter_values : array
        The parameter values used for the calculation of the FRET-line
    parameter_range : array
        Start and stop value of the parameter range used for calculation of the FRET-line
    n_points : int
        The number of points used to calculate the FRET-line
    model : LifetimeModel
        The TCSPC-model used to calculate the FRET-line

    Methods
    -------
    calc()
        Recaclulates the FRET-line

    Examples
    --------

    >>> import mfm.fitting.model.tcspc as m
    >>> from mfm.tools.fret_lines import FRETLineGenerator
    >>> R1 = 80
    >>> R2 = 35
    >>> fl = FRETLineGenerator()
    >>> fl.model = m.GaussianModel
    >>> fl.model.parameter_dict['DOnly'].value = 0.0

    Adding donor lifetimes

    >>> fl.model.donors.append(1.0, 4)
    >>> fl.model.donors.append(1.0, 2)

    Add a new Gaussian distance

    >>> fl.model.append(55.0, 10, 1.0)

    The fluorescence/species averaged lifetime of the model is obtained by

    >>> fl.fret_fluorescence_averaged_lifetime
    2.5600809847436152
    >>> fl.fret_species_averaged_lifetime
    2.2459102590812412

    The model parameters can be changed using their names by the *parameter_dict*

    >>> fl.model.parameter_dict['R(G,1)'].value = 40.0
    >>> fl.model.parameter_dict['s(G,1)'].value = 8.0

    Set the name of the parameter changes to calculate the distributions used for the
    FRET-lines. Here a more common parameter as the donor-acceptor separation distance
    is used to generate a static FRET-line.

    >>> fl.parameter_name = 'R(G,1)'

    Set the range in which this parameter is modified to generate the line and calculate the FRET-line

    >>> fl.parameter_range = 0.1, 1000
    >>> fl.calc()

    By adding a second Gaussian and changing the species fraction you can also generate a dynamic-FRET line

    .. plot:: plots/fret-lines.py


    """

    def __init__(
            self,
            polynomial_degree: int = 4,
            verbose: bool = mfm.verbose,
            quantum_yield_donor: float = 0.8,
            quantum_yield_acceptor: float = 0.32,
            n_points: int = 500,
            parameter_name: str = None,
            **kwargs
    ):
        self.verbose = verbose
        self._polynomial_degree = polynomial_degree
        self.quantum_yield_donor = quantum_yield_donor
        self.quantum_yield_acceptor = quantum_yield_acceptor

        self._range = kwargs.get('range', [0.1, 100.0])
        self._n_points = n_points
        self._parameter_name = parameter_name
        self._t_max = kwargs.get('t_max', 1000.0)
        self._dt = kwargs.get('dt', 0.1)

        self.fit = mfm.fitting.fit.Fit()
        self._data_points = mfm.experiments.data.DataCurve(
            setup=None,
            x=np.linspace(0.0, self._t_max, self._n_points),
            y=np.zeros(self._n_points)
        )
        self.fit.data = self._data_points

        self._model = kwargs.get(
            'model',
            mfm.models.tcspc.lifetime.LifetimeModel(
                fit=self.fit,
                dt=self._dt,
                do_convolution=False
            )
        )
        self.fret_efficiencies = np.zeros(
            self.n_points,
            dtype=np.float
        )
        self.fluorescence_averaged_lifetimes = np.zeros_like(
            self.fret_efficiencies
        )

    @property
    def conversion_function(self):
        """
        The fluorescence averaged lifetime vs. the species averaged lifetime
        """
        return self.fluorescence_averaged_lifetimes, self.fret_species_averaged_lifetimes

    @property
    def parameter_name(
            self
    ) -> str:
        """
        The name of the parameter which is varies
        """
        return self._parameter_name

    @parameter_name.setter
    def parameter_name(
            self,
            v: str
    ):
        self._parameter_name = v

    @property
    def fret_species_averaged_lifetime(
            self
    ) -> float:
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self._model.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(
            self
    ) -> float:
        """
        The current fluorescence averaged lifetime of the FRET-sample = xi*taui**2 / species_averaged_lifetime
        """
        return self._model.fret_fluorescence_averaged_lifetime

    @property
    def donor_species_averaged_lifetime(
            self
    ) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self._model.lifetimes.species_averaged_lifetime

    @property
    def transfer_efficiency(
            self
    ) -> float:
        """
        The current transfer efficency
        """
        return 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime

    @property
    def polynomial_degree(
            self
    ) -> int:
        """
        The degree of the polynomial to approximate the conversion function
        """
        return self._polynomial_degree

    @polynomial_degree.setter
    def polynomial_degree(
            self,
            v: int
    ):
        """
        The degree of the polynomial used to approximate the transfer function
        """
        self._polynomial_degree = int(v)

    @property
    def polynom_coefficients(
            self
    ):
        """
        A numpy array with polynomial coefficients approximating the tauX(tauF) conversion function
        """
        x, y = self.fluorescence_averaged_lifetimes, self.fret_species_averaged_lifetimes
        return np.polyfit(x, y, self.polynomial_degree)

    @property
    def conversion_function_string(self):
        """
        A string used for plotting of the conversion function tauX(tauF)
        """
        c = self.polynom_coefficients
        s = ""
        for i, c in enumerate(c[::-1]):
            s += "%s*x^%i+" % (c, i)
        return s[:-1]

    @property
    def transfer_efficency_string(self):
        """
        Used for instance for plotting in origin
        """
        return "1.0-(%s)/(%s)" % (self.conversion_function_string, self.donor_species_averaged_lifetime)

    @property
    def fdfa_string(self):
        """
        Used for instance for plotting in origin
        """
        qd = self.quantum_yield_donor
        qa = self.quantum_yield_acceptor
        return "%s/%s / ((%s)/(%s) - 1)" % (qd, qa, self.donor_species_averaged_lifetime, self.conversion_function_string)


    @property
    def donor_lifetime_spectrum(self):
        """
        The donor lifetime spectrum in form amplitude, lifetime, amplitude, lifetime
        """
        return self.model.lifetimes.lifetime_spectrum

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(self, v):
        self.model.lifetimes.lifetime_spectrum = v

    @property
    def parameter_values(self):
        """
        The values the parameter as defined by :py:attr:`~mfm.fluorescence.fret_lines.FRETLineGenerator.parameter_name`
        """
        start, stop = self.parameter_range
        n_points = self.n_points
        return np.linspace(start, stop, n_points)

    @property
    def parameter_range(self):
        return self._range

    @parameter_range.setter
    def parameter_range(self, v):
        self._range = v

    @property
    def n_points(self):
        return self._n_points

    @n_points.setter
    def n_points(self, v):
        self._n_points = v
        self.calc()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        self._model = v(
            fit=self.fit,
            dt=self._dt
        )

    def calc(
            self,
            parameter_name: str = None,
            parameter_range: Tuple[float, float] = None,
            verbose: bool = None,
            **kwargs
    ):
        if verbose is None:
            verbose = self.verbose

        if isinstance(parameter_name, str):
            self.parameter_name = parameter_name
        if isinstance(parameter_range, list):
            self.parameter_range = parameter_range

        if verbose:
            print("Calculating FRET-Line")
            print("Using parameter: %s" % self.parameter_name)
            print("In a range: %.1f .. %.1f" % (self.parameter_range[0], self.parameter_range[1]))

        transfer_efficiencies = np.zeros(self.n_points)
        fluorescence_averaged_lifetimes = np.zeros(self.n_points)
        species_averaged_lifetimes = np.zeros(self.n_points)
        for i, parameter_value in enumerate(self.parameter_values):
            self.model.parameter_dict[self.parameter_name].value = parameter_value
            self.model.update_model()
            fluorescence_averaged_lifetimes[i] = self.fret_fluorescence_averaged_lifetime
            transfer_efficiencies[i] = self.transfer_efficiency
            species_averaged_lifetimes[i] = self.fret_species_averaged_lifetime
        self.fret_efficiencies = transfer_efficiencies
        self.fret_species_averaged_lifetimes = species_averaged_lifetimes
        self.fluorescence_averaged_lifetimes = fluorescence_averaged_lifetimes


class StaticFRETLine(
    FRETLineGenerator
):
    """
    This class is used to calculate static-FRET lines of Gaussian distributed states.


    Examples
    --------

    >>> import mfm.tools.fret_lines as fret_lines
    >>> s = fret_lines.StaticFRETLine()
    >>> s.calc()

    Now lets look at the conversion function in comparison to a 1:1 relation

    >>> import pylab as p
    >>> x, y = s.conversion_function
    >>> p.plot(x, y)
    >>> p.plot(x, x)
    >>> p.show()

    This class has basically only one relevant attribute that is the width of the DA-distance distirbution
    within a state.

    A conversion polynomial for plotting purposes can be obtained

    >>> s.polynomial_string

    """

    @property
    def sigma(self):
        """
        Width of the DA-distance distribution within the state
        """
        return self.model.parameter_dict['s(G,1)'].value

    @sigma.setter
    def sigma(self, v):
        self.model.parameter_dict['s(G,1)'].value = v

    def __init__(self, **kwargs):
        FRETLineGenerator.__init__(self, **kwargs)
        self.model = fret_models.GaussianModel
        self.model.parameter_dict['DOnly'].value = 0.0
        self.model.lifetimes.append(1.0, 4)
        self.model.append(40.0, 10, 1.0)

    def calc(self, parameter_name=None, parameter_range=[0, 500], **kwargs):
        self.model.parameter_dict['x(G,1)'].value = 1.0
        FRETLineGenerator.calc(self, parameter_name='R(G,1)', parameter_range=parameter_range, **kwargs)
        if kwargs.get('return_conversion', False):
            return self.conversion_function


class DynamicFRETLine(FRETLineGenerator):
    """
    This is a `convenience` class for a simple two-state Gaussian distance distribution
    dynamic FRET-model. Two Gaussian distance distributions are considered as limiting
    states and the dynamic-FRET line is calculated


    Examples
    --------

    >>> import mfm.tools.fret_lines as fret_lines
    >>> d = fret_lines.DynamicFRETLine()
    >>> print d.model
    Model: Gaussian-Donor
    Parameter       Value   Bounds  Fixed   Linke
    bg      0.0000  (None, None)    False   False
    sc      0.0000  (None, None)    False   False
    t(L,1)  4.0000  (None, None)    False   False
    x(L,1)  1.0000  (None, None)    False   False
    t0      4.1000  (None, None)    True    False
    R0      52.0000 (None, None)    True    False
    k2      0.6670  (None, None)    True    False
    DOnly   0.0000  (0.0, 1.0)      False   False
    x(G,1)  1.0000  (None, None)    False   False
    x(G,2)  1.0000  (None, None)    False   False
    s(G,1)  10.0000 (None, None)    False   False
    s(G,2)  10.0000 (None, None)    False   False
    R(G,1)  40.0000 (None, None)    False   False
    R(G,2)  80.0000 (None, None)    False   False
    dt      0.1000  (None, None)    True    False
    start   0.0000  (None, None)    True    False
    stop    20.0000 (None, None)    True    False
    rep     10.0000 (None, None)    True    False
    lb      0.0000  (None, None)    False   False
    ts      0.0000  (None, None)    False   False
    p0      1.0000  (None, None)    True    False
    r0      0.3800  (None, None)    False   False
    l1      0.0308  (None, None)    False   False
    l2      0.0368  (None, None)    False   False
    g       1.0000  (None, None)    False   False

    Set a new sigma

    >>> d.sigma = 3.0
    >>> print d.model
    Model: Gaussian-Donor
    Parameter       Value   Bounds  Fixed   Linke
    bg      0.0000  (None, None)    False   False
    sc      0.0000  (None, None)    False   False
    t(L,1)  4.0000  (None, None)    False   False
    x(L,1)  1.0000  (None, None)    False   False
    t0      4.1000  (None, None)    True    False
    R0      52.0000 (None, None)    True    False
    k2      0.6670  (None, None)    True    False
    DOnly   0.0000  (0.0, 1.0)      False   False
    x(G,1)  1.0000  (None, None)    False   False
    x(G,2)  1.0000  (None, None)    False   False
    s(G,1)  3.0000  (None, None)    False   False
    s(G,2)  3.0000  (None, None)    False   False
    R(G,1)  40.0000 (None, None)    False   False
    R(G,2)  80.0000 (None, None)    False   False
    dt      0.1000  (None, None)    True    False
    start   0.0000  (None, None)    True    False
    stop    20.0000 (None, None)    True    False
    rep     10.0000 (None, None)    True    False
    lb      0.0000  (None, None)    False   False
    ts      0.0000  (None, None)    False   False
    p0      1.0000  (None, None)    True    False
    r0      0.3800  (None, None)    False   False
    l1      0.0308  (None, None)    False   False
    l2      0.0368  (None, None)    False   False
    g       1.0000  (None, None)    False   False

    Calculate the FRET-line

    >>> d.calc()
    Calculating FRET-Line
    Using parameter: x(G,2)
    In a range: 0.1 .. 100.0

    Plot the conversion function

    >>> import pylab as p
    >>> tauf, taux = d.conversion_function
    >>> p.plot(tauf, taux)
    >>> p.show()
    """

    def __init__(self, **kwargs):
        FRETLineGenerator.__init__(self, **kwargs)
        self.model = fret_models.gaussion.GaussianModel
        self.model.parameter_dict['DOnly'].value = 0.0
        self.model.lifetimes.append(1.0, 4)
        self.model.append(40.0, 10, 1.0)
        self.model.append(80.0, 10, 1.0)

    @property
    def mean_distance_1(self):
        """
        Mean distance of first limiting state 1
       """
        return self.model.parameter_dict['R(G,1)'].value

    @mean_distance_1.setter
    def mean_distance_1(self, v):
        self.model.parameter_dict['R(G,1)'].value = v

    @property
    def mean_distance_2(self):
        """
        Mean distance of first limiting state 2
       """
        return self.model.parameter_dict['R(G,2)'].value

    @mean_distance_1.setter
    def mean_distance_2(self, v):
        self.model.parameter_dict['R(G,2)'].value = v

    @property
    def sigma_1(self):
        """
        Width of first limiting state
       """
        return self.model.parameter_dict['s(G,1)'].value

    @sigma_1.setter
    def sigma_1(self, v):
        self.model.parameter_dict['s(G,1)'].value = v

    @property
    def sigma_2(self):
        """
        Width of second limiting state
       """
        return self.model.parameter_dict['s(G,2)'].value

    @sigma_2.setter
    def sigma_1(self, v):
        self.model.parameter_dict['s(G,2)'].value = v

    @property
    def sigma(self):
        """
        The width of both sigmas
       """
        return self.model.parameter_dict['s(G,1)'].value, self.model.parameter_dict['s(G,2)'].value

    @sigma.setter
    def sigma(self, v):
        try:
            self.model.parameter_dict['s(G,1)'].value = v[0]
            self.model.parameter_dict['s(G,2)'].value = v[1]
        except TypeError:
            self.model.parameter_dict['s(G,1)'].value = v
            self.model.parameter_dict['s(G,2)'].value = v

    def calc(self, parameter_name=None, parameter_range=None, **kwargs):
        self.model.parameter_dict['x(G,1)'].value = 1.0
        self.model.parameter_dict['x(G,2)'].value = 0.0
        FRETLineGenerator.calc(self, parameter_name='x(G,2)', parameter_range=[0, 10], **kwargs)
        if kwargs.get('return_conversion', False):
            return self.conversion_function


class DynamicFRETLineSteps(DynamicFRETLine):
    """
    This class helps to generate a dynamic FRET line for a linear multi-state system
    if two end-points are known the number of steps `n` between the end-points is provided
    by the user of this class and and `n` dynamic FRET-lines are calculated. Based on these
    FRET-lines an overall conversion function covering the whole range from the start to
    the end is calculated.

    Examples
    --------

    >>> import mfm.tools.fret_lines as fret_lines
    >>> d = fret_lines.DynamicFRETLineSteps()

    We also want to plot the conversion function. I use pylab

    >>> import pylab as p

    Now lets calculate the conversion function. By default 10 intermediate steps
    are used

    >>> d.calc()
    >>> x, y = d.joint_conversion_function
    >>> p.plot(x, y)

    Now lets use only three steps and lets check how it looks like

    >>> d.number_of_intermediates = 3
    >>> d.calc()
    >>> x, y = d.joint_conversion_function
    >>> p.plot(x, y)

    Before we look at the plot a 1:1 conversion function might be useful to have a
    reference and a static FRET-line

    >>> p.plot(x, x)
    >>> s = fret_lines.StaticFRETLine()
    >>> s.calc()
    >>> x, y = s.conversion_function
    >>> p.plot(x, y)

    >>> p.show()
    """

    @property
    def joint_conversion_function(self):
        """
        The overall conversion function including all intermediates
        """
        return self._joint_conversion_function

    @property
    def start_point(self):
        return self._start_point

    @start_point.setter
    def start_point(self, v):
        """
        The start point in Angstrom
        """
        self._start_point = v

    @property
    def end_point(self):
        return self._end_point

    @end_point.setter
    def end_point(self, v):
        """
        The end point in Angstrom
        """
        self._end_point = v

    @property
    def number_of_intermediates(self):
        """
        The number of intermediate points between the start and the endpoint
        """
        return self._number_of_intermediates

    @number_of_intermediates.setter
    def number_of_intermediates(self, v):
        self._number_of_intermediates = v

    @property
    def mean_distances(self):
        """
        The array of mean distances between the two endpoints
        """
        return np.linspace(self.start_point, self.end_point, self.number_of_intermediates)

    def __init__(self, **kwargs):
        DynamicFRETLine.__init__(self, **kwargs)
        self._start_point = kwargs.get('start_point', 40)
        self._end_point = kwargs.get('end_point', 80)
        self._number_of_intermediates = kwargs.get('number_of_intermediates', 5)
        self._joint_conversion_function = None

    def calc(self, parameter_name=None, parameter_range=None, **kwargs):
        mean_distances = self.mean_distances
        conversion_functions = list()
        for i in range(len(mean_distances)-1):
            self.mean_distance_1 = mean_distances[i]
            self.mean_distance_2 = mean_distances[i + 1]
            DynamicFRETLine.calc(self, parameter_name=None, parameter_range=None, **kwargs)
            conversion_functions.append(self.conversion_function)
        self._joint_conversion_function = np.hstack(conversion_functions)
        if kwargs.get('return_conversion', False):
            return self.conversion_function


class FRETLineGeneratorWidget(QtWidgets.QWidget, FRETLineGenerator):

    name = "FRET-Line Generator"

    models = [(mfm.models.tcspc.widgets.GaussianModelWidget, {'hide_corrections': True,
                                                          'hide_fit': True,
                                                          'hide_generic': True,
                                                          'hide_convolve': True,
                                                          'hide_rotation': True,
                                                          'hide_error': True,
                                                           'hide_donor': True
                                                              }),
              # (mfm.models.tcspc.mix_model.LifetimeMixModelWidget, {'hide_corrections': True,
              #                                             'hide_fit': True,
              #                                             'hide_generic': True,
              #                                             'hide_convolve': True,
              #                                             'hide_rotation': True,
              #                                             'hide_error': True,
              #                                             'hide_donor': True,
              #                                                              'enable_mix_model_donor': True
              #                                                      }),
              (mfm.models.tcspc.widgets.FRETrateModelWidget, {'hide_corrections': True,
                                                          'hide_fit': True,
                                                          'hide_generic': True,
                                                          'hide_convolve': True,
                                                          'hide_rotation': True,
                                                          'hide_error': True,
                                                           'hide_donor': True,
                                                              'enable_mix_model_donor': True
                                                              }),
              ]

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_index):
        self.comboBox.setCurrentIndex(model_index)
        model, parameter = self.models[model_index]
        self._model = model(fit=self.fit, **parameter)

    @property
    def current_model_index(self):
        return int(self.comboBox.currentIndex())

    @property
    def n_points(self):
        return int(self.spinBox.value())

    @property
    def parameter_range(self):
        return float(self.doubleSpinBox.value()), float(self.doubleSpinBox_2.value())

    @mfm.decorators.init_with_ui(ui_filename="fret_line.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        win = CurveDialog(edit=False, toolbar=True)

        # Make Plot
        plot = win.get_plot()
        self.verticalLayout_5.addWidget(plot)
        self.fret_line_plot = plot

        FRETLineGenerator.__init__(self, **kwargs)
        self.fit = mfm.FitQtThread()  # the fit has to be a QtThread
        self.fit.data = self._data_points

        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.model_names = [str(model[0].name) for model in self.models]
        self.comboBox.addItems(self.model_names)
        self.model = self.current_model_index
        self.actionModel_changed.triggered.connect(self.onModelChanged)
        self.actionParameter_changed.triggered.connect(self.onParameterChanged)
        self.actionUpdate_Parameter.triggered.connect(self.update_parameter)
        self.actionCalculate_FRET_Line.triggered.connect(self.onCalculate)
        self.actionClear_plot.triggered.connect(self.onClearPlot)

        self.onModelChanged()
        self.update_parameter()
        self.hide()

    def onClearPlot(self):
        print("onClearPlot")
        self.fret_line_plot.del_all_items()

    def onCalculate(self):
        self.calc()
        fret_line = make.curve(self.fluorescence_averaged_lifetimes, self.fret_efficiencies,
                               color="r", linewidth=2)
        self.fret_line_plot.add_item(fret_line)
        self.fret_line_plot.do_autoscale()
        self.lineEdit.setText(self.transfer_efficency_string)
        self.lineEdit_2.setText(self.fdfa_string)
        self.lineEdit_3.setText("%s" % list(self.polynom_coefficients))

    def onParameterChanged(self):
        self.parameter_name = str(self.comboBox_2.currentText())

    def update_parameter(self):
        self.comboBox_2.clear()
        self.comboBox_2.addItems(self.model.parameter_names)

    def onModelChanged(self):
        self.model.close()
        self.model = self.current_model_index
        self.verticalLayout.addWidget(self.model)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FRETLineGeneratorWidget()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
