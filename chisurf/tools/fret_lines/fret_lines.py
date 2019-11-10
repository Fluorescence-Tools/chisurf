from __future__ import annotations

import sys

from qtpy import QtWidgets
from guiqwt.builder import make
from guiqwt.plot import CurveDialog

import numpy as np

import chisurf.settings
import chisurf.widgets
import chisurf.decorators
import chisurf.fitting
import chisurf.experiments
import chisurf.experiments.data
import chisurf.models.tcspc
import chisurf.models.tcspc as fret_models
import chisurf.models.tcspc.lifetime
import chisurf.models.tcspc.widgets
#from chisurf.models import WormLikeChainModelWidget
from chisurf.fluorescence.fret.fret_line import FRETLineGenerator


class StaticFRETLine(
    FRETLineGenerator
):
    """
    This class is used to calculate static-FRET lines of Gaussian distributed states.


    Examples
    --------

    >>> import chisurf.tools.fret_lines as fret_lines
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

    >>> import chisurf.tools.fret_lines as fret_lines
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

    >>> import chisurf.tools.fret_lines as fret_lines
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

    models = [(chisurf.models.tcspc.widgets.GaussianModelWidget, {'hide_corrections': True,
                                                          'hide_fit': True,
                                                          'hide_generic': True,
                                                          'hide_convolve': True,
                                                          'hide_rotation': True,
                                                          'hide_error': True,
                                                           'hide_donor': True
                                                              }),
              # (chisurf.models.tcspc.mix_model.LifetimeMixModelWidget, {'hide_corrections': True,
              #                                             'hide_fit': True,
              #                                             'hide_generic': True,
              #                                             'hide_convolve': True,
              #                                             'hide_rotation': True,
              #                                             'hide_error': True,
              #                                             'hide_donor': True,
              #                                                              'enable_mix_model_donor': True
              #                                                      }),
              (chisurf.models.tcspc.widgets.FRETrateModelWidget, {'hide_corrections': True,
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

    @chisurf.decorators.init_with_ui(ui_filename="fret_line.ui")
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
        self.fit = chisurf.widgets.FitQtThread()  # the fit has to be a QtThread
        self.fit.data = self._data_curve

        self.verbose = kwargs.get('verbose', chisurf.verbose)
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
    win.show()
    sys.exit(app.exec_())
