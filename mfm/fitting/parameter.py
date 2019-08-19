from __future__ import annotations
from typing import Tuple, List

import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore

import mfm
import mfm.base
import mfm.parameter

parameter_settings = mfm.settings.cs_settings['parameter']


class FittingParameter(mfm.parameter.Parameter):

    def __init__(
            self,
            link=None,
            model: mfm.models.model.Model = None,
            lb: float = -10000,
            ub: float = 10000,
            fixed: bool = False,
            bounds_on: bool = False,
            error_estimate: bool = None,
            **kwargs
    ):
        super(FittingParameter, self).__init__(**kwargs)
        self._link = link
        self.model = model

        self._lb = lb
        self._ub = ub
        self._fixed = fixed
        self._bounds_on = bounds_on
        self._error_estimate = error_estimate

        self._chi2s = None
        self._values = None

    @property
    def parameter_scan(self):
        return self._values, self._chi2s

    @parameter_scan.setter
    def parameter_scan(self, v):
        self._values, self._chi2s = v

    @property
    def error_estimate(self):
        if self.is_linked:
            return self._link.error_estimate
        else:
            if isinstance(self._error_estimate, float):
                return self._error_estimate
            else:
                return "NA"

    @error_estimate.setter
    def error_estimate(self, v):
        self._error_estimate = v

    @property
    def bounds(self) -> Tuple[float, float]:
        if self.bounds_on:
            return self._lb, self._ub
        else:
            return None, None

    @bounds.setter
    def bounds(
            self,
            b: Tuple[float, float]
    ):
        self._lb, self._ub = b

    @property
    def bounds_on(self) -> bool:
        return self._bounds_on

    @bounds_on.setter
    def bounds_on(
            self,
            v: bool
    ):
        self._bounds_on = bool(v)

    @property
    def value(self):
        v = self._value
        if callable(v):
            return v()
        else:
            if self.is_linked:
                return self.link.value
            else:
                if self.bounds_on:
                    lb, ub = self.bounds
                    if lb is not None:
                        v = max(lb, v)
                    if ub is not None:
                        v = min(ub, v)
                    return v
                else:
                    return v

    @value.setter
    def value(self, value):
        self._value = float(value)
        if self.is_linked:
            self.link.value = value

    @property
    def fixed(self) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(
            self,
            v: bool
    ):
        self._fixed = v

    @property
    def link(self):
        return self._link

    @link.setter
    def link(
            self,
            link: mfm.fitting.FittingParameter
    ):
        if isinstance(link, FittingParameter):
            self._link = link
        elif link is None:
            try:
                self._value = self._link.value
            except AttributeError:
                pass
            self._link = None

    @property
    def is_linked(self) -> bool:
        return isinstance(self._link, FittingParameter)

    def to_dict(self) -> dict:
        d = mfm.parameter.Parameter.to_dict(self)
        d['lb'], d['ub'] = self.bounds
        d['fixed'] = self.fixed
        d['bounds_on'] = self.bounds_on
        d['error_estimate'] = self.error_estimate
        return d

    def from_dict(
            self,
            d: dict
    ):
        mfm.parameter.Parameter.from_dict(self, d)
        self._lb, self._ub = d['lb'], d['ub']
        self._fixed = d['fixed']
        self._bounds_on = d['bounds_on']
        self._error_estimate = d['error_estimate']

    def scan(
            self,
            fit: mfm.fitting.fit.Fit,
            **kwargs):
        fit.chi2_scan(self.name, **kwargs)

    def __str__(self):
        s = "\nVariable\n"
        s += "name: %s\n" % self.name
        s += "internal-value: %s\n" % self._value
        if self.bounds_on:
            s += "bounds: %s\n" % self.bounds
        if self.is_linked:
            s += "linked to: %s\n" % self.link.name
            s += "link-value: %s\n" % self.value
        return s

    def make_widget(
            self,
            **kwargs
    ) -> FittingParameterWidget:
        text = kwargs.get('text', self.name)
        layout = kwargs.get('layout', None)
        update_widget = kwargs.get('update_widget', lambda x: x)
        decimals = kwargs.get('decimals', self.decimals)
        kw = {
            'text': text,
            'decimals': decimals,
            'layout': layout
        }
        widget = FittingParameterWidget(self, **kw)
        self.controller = widget
        return widget


class FittingParameterWidget(QtWidgets.QWidget):

    def make_linkcall(self, fit_idx, parameter_name):

        def linkcall():
            tooltip = " linked to " + parameter_name
            mfm.run(
                "cs.current_fit.model.parameters_all_dict['%s'].link = mfm.fits[%s].model.parameters_all_dict['%s']" %
                (self.name, fit_idx, parameter_name)
            )
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.update()
        return linkcall

    def contextMenuEvent(self, event):

        menu = QtWidgets.QMenu(self)
        menu.setTitle("Link " + self.fitting_parameter.name + " to:")

        for fit_idx, f in enumerate(mfm.fits):
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)

                # Sorted by "Aggregation"
                for a in fs.model.aggregated_parameters:
                    action_submenu = QtWidgets.QMenu(submenu)
                    action_submenu.setTitle(a.name)
                    ut = a.parameters
                    ut.sort(key=lambda x: x.name, reverse=False)
                    for p in ut:
                        if p is not self:
                            Action = action_submenu.addAction(p.name)
                            Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                    submenu.addMenu(action_submenu)
                action_submenu = QtWidgets.QMenu(submenu)

                # Simply all parameters
                action_submenu.setTitle("All parameters")
                for p in fs.model.parameters_all:
                    if p is not self:
                        Action = action_submenu.addAction(p.name)
                        Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                submenu.addMenu(action_submenu)

                menu.addMenu(submenu)
        menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    def __init__(self, fitting_parameter, **kwargs):
        layout = kwargs.pop('layout', None)
        label_text = kwargs.pop('text', self.__class__.__name__)
        hide_bounds = kwargs.pop('hide_bounds', parameter_settings['hide_bounds'])
        hide_link = kwargs.pop('hide_link', parameter_settings['hide_link'])
        fixable = kwargs.pop('fixable', parameter_settings['fixable'])
        hide_fix_checkbox = kwargs.pop('hide_fix_checkbox', parameter_settings['fixable'])
        hide_error = kwargs.pop('hide_error', parameter_settings['hide_error'])
        hide_label = kwargs.pop('hide_label', parameter_settings['hide_label'])
        decimals = kwargs.pop('decimals', parameter_settings['decimals'])

        super(FittingParameterWidget, self).__init__(**kwargs)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('mfm/ui/variable_widget.ui', self)
        self.fitting_parameter = fitting_parameter
        self.widget_value = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_value.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.widget_value.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addWidget(self.widget_value)

        self.widget_lower_bound = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_lower_bound.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_lower_bound)

        self.widget_upper_bound = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_upper_bound.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_upper_bound)

        # Hide and disable widgets
        self.label.setVisible(not hide_label)
        self.lineEdit.setVisible(not hide_error)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget_fix.setVisible(fixable or not hide_fix_checkbox)
        self.widget.setHidden(hide_bounds)
        self.widget_link.setDisabled(hide_link)

        # Display of values
        self.widget_value.setValue(float(self.fitting_parameter.value))

        self.label.setText(label_text.ljust(5))

        # variable bounds
        if not self.fitting_parameter.bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)

        # variable fixed
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        self.widget.hide()

        # The variable value
        self.widget_value.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].value = %s\n"
            "cs.current_fit.update()" %
            (self.fitting_parameter.name, self.widget_value.value()))
        )

        self.widget_fix.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].fixed = %s" %
            (self.fitting_parameter.name, self.widget_fix.isChecked()))
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds_on = %s" %
            (self.fitting_parameter.name, self.widget_bounds_on.isChecked()))
        )

        self.widget_lower_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.fitting_parameter.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_upper_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.fitting_parameter.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_link.clicked.connect(self.onLinkFitGroup)

        if isinstance(layout, QtWidgets.QLayout):
            layout.addWidget(self)

    def onLinkFitGroup(self):
        self.blockSignals(True)
        cs = self.widget_link.checkState()
        if cs == 2:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       if p is not s:
           p.link = s
   except KeyError:
       pass
            """ % (self.fitting_parameter.name, self.fitting_parameter.name)
            mfm.run(t)
            self.widget_link.setCheckState(QtCore.Qt.Checked)
            self.widget_value.setEnabled(False)
        elif cs == 0:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       p.link = None
   except KeyError:
       pass
            """ % (self.fitting_parameter.name, self.fitting_parameter.name)
            mfm.run(t)
        self.widget_value.setEnabled(True)
        self.widget_link.setCheckState(QtCore.Qt.Unchecked)
        self.blockSignals(False)

    def finalize(self, *args):
        QtWidgets.QWidget.update(self, *args)
        self.blockSignals(True)
        # Update value of widget
        self.widget_value.setValue(float(self.fitting_parameter.value))
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        # Tooltip
        s = "bound: (%s,%s)\n" % self.fitting_parameter.bounds if self.fitting_parameter.bounds_on else "bounds: off\n"
        if self.fitting_parameter.is_linked:
            s += "linked to: %s" % self.fitting_parameter.link.name
        self.widget_value.setToolTip(s)

        # Error-estimate
        value = self.fitting_parameter.value
        if self.fitting_parameter.fixed or not isinstance(self.fitting_parameter.error_estimate, float):
            self.lineEdit.setText("NA")
        else:
            rel_error = abs(self.fitting_parameter.error_estimate / (value + 1e-12) * 100.0)
            self.lineEdit.setText("%.0f%%" % rel_error)
        #link
        if self.fitting_parameter.link is not None:
            tooltip = " linked to " + self.fitting_parameter.link.name
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.blockSignals(False)


class GlobalFittingParameter(FittingParameter):

    @property
    def value(self):
        g = self.g
        f = self.f
        r = eval(self.formula)
        return r.value

    @value.setter
    def value(self, v):
        pass

    @property
    def name(self):
        return self.formula

    @name.setter
    def name(self, v):
        pass

    def __init__(self, f, g, formula, **kwargs):
        #FittingParameter.__init__(self, **kwargs)
        args = [f, g, formula]
        super(GlobalFittingParameter, self).__init__(*args, **kwargs)
        self.f, self.g = f, g
        self.formula = formula


class FittingParameterGroup(mfm.base.Base):

    @property
    def parameters_all(self):
        return self._parameters

    @property
    def parameters_all_dict(self):
        return dict([(p.name, p) for p in self.parameters_all])

    @property
    def parameters(self):
        return self.parameters_all

    @property
    def aggregated_parameters(self):
        d = self.__dict__
        a = list()
        for key, value in d.items():
            if isinstance(value, FittingParameterGroup):
                a.append(value)
        return list(set(a))

    @property
    def parameter_dict(self):
        re = dict()
        for p in self.parameters:
            re[p.name] = p
        return re

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    @property
    def parameter_values(self):
        return [p.value for p in self.parameters]

    @parameter_values.setter
    def parameter_values(self, vs):
        ps = self.parameters
        for i, v in enumerate(vs):
            ps[i].value = v

    @property
    def outputs(self):
        """The outputs of a ParameterGroup are a dictionary

        Returns
        -------

        """
        a = dict()
        return a

    def to_dict(self):
        s = dict()
        parameters = dict()
        s['parameter'] = parameters
        for parameter in self._parameters:
            parameters[parameter.name] = parameter.to_dict()
        return s

    def from_dict(self, v):
        self.find_parameters()
        parameter_target = self.parameters_all_dict
        parameter = v['parameter']
        for parameter_name in parameter:
            pn = str(parameter_name)
            try:
                parameter_target[pn].from_dict(parameter[pn])
            except KeyError:
                print("Key %s not found skipping" % pn)

    def find_parameters(self, parameter_type=mfm.parameter.Parameter):
        self._aggregated_parameters = None
        self._parameters = None

        ag = mfm.find_objects(self.__dict__.values(), FittingParameterGroup)
        self._aggregated_parameters = ag

        ap = list()
        for o in set(ag):
            if not isinstance(o, mfm.fitting.model.model.Model):
                o.find_parameters()
                self.__dict__[o.name] = o
                ap += o._parameters

        mp = mfm.find_objects(self.__dict__.values(), parameter_type)
        self._parameters = list(set(mp + ap))

    def append_parameter(self, p):
        if isinstance(p, mfm.parameter.Parameter):
            self._parameters.append(p)

    def finalize(self):
        pass

    def __getattr__(self, item):
        item = mfm.base.Base.__getattr__(self, item)
        if isinstance(item, mfm.parameter.Parameter):
            return item.value
        else:
            return item

    def __len__(self):
        return len(self.parameters_all)

    def __init__(
            self,
            fit: mfm.fitting.Fit = None,
            model: mfm.fitting.model.model.Model = None,
            short: str = '',
            parameters: List[mfm.parameter.Parameter] = list(),
            *args, **kwargs):
        """

        :param fit: the fit to which the parameter group is associated to
        :param model: the model to which the parameter group is associated to
        :param short: a short name for the parameter group
        :param parameters: a list of the fitting parameters that are grouped by the fitting parameter group
        :param args:
        :param kwargs:
        """
        if mfm.verbose:
            print("---------------")
            print("Class: %s" % self.__class__.name)
            print(kwargs)
            print("---------------")
        # super(mfm.Base, self).__init__(*args, **kwargs)
        self.short = short
        self.model = model
        self.fit = fit
        self._parameters = parameters
        self._aggregated_parameters = list()
        self._parameter_names = None

        # Copy parameters from provided ParameterGroup
        if len(args) > 0:
            p0 = args[0]
            if isinstance(p0, FittingParameterGroup):
                self.__dict__ = p0.__dict__

    def to_widget(self, *args, **kwargs) -> FittingParameterGroupWidget:
        return FittingParameterGroupWidget(self, *args, **kwargs)


class FittingParameterGroupWidget(FittingParameterGroup, QtWidgets.QGroupBox):

    def __init__(self, parameter_group, *args, **kwargs):
        self.parameter_group = parameter_group
        super(FittingParameterGroupWidget, self).__init__(*args, **kwargs)
        self.setTitle(self.name)

        self.n_col = kwargs.get('n_cols', mfm.settings.cs_settings['gui']['fit_models']['n_columns'])
        self.n_row = 0
        l = QtWidgets.QGridLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        self.setLayout(l)

        for i, p in enumerate(parameter_group.parameters_all):
            pw = p.make_widget()
            col = i % self.n_col
            row = i // self.n_col
            l.addWidget(pw, row, col)