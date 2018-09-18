from PyQt4 import QtGui, uic, QtCore
import mfm
import json
import pyqtgraph as pg
import weakref
from dotmap import DotMap

parameter_settings = mfm.settings['parameter']


class Parameter(mfm.Base):

    _instances = set()

    @property
    def decimals(self):
        return self._decimals

    @decimals.setter
    def decimals(self, v):
        self._decimals = v

    @property
    def value(self):
        v = self._value
        if callable(v):
            return v()
        else:
            return v

    @value.setter
    def value(self, value):
        self._value = float(value)

    @classmethod
    def getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    def __add__(self, other):
        if isinstance(other, (int, float)):
            a = self.value + other
        else:
            a = self.value + other.value
        return Parameter(value=a)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            a = self.value * other
        else:
            a = self.value * other.value
        return Parameter(value=a)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            a = self.value - other
        else:
            a = self.value - other.value
        return Parameter(value=a)

    def __div__(self, other):
        if isinstance(other, (int, float)):
            a = self.value / other
        else:
            a = self.value / other.value
        return Parameter(value=a)

    def __float__(self):
        return float(self.value)

    def __invert__(self):
        return FittingParameter(value=float(1.0 / self.value))

    def __init__(self, *args, **kwargs):
        self._instances.add(weakref.ref(self))
        mfm.Base.__init__(self, **kwargs)
        self._link = kwargs.get('link', None)
        self.model = kwargs.get('model', None)
        value = args[0] if len(args) > 0 else 1.0
        self._value = kwargs.get('value', value)
        self._decimals = kwargs.get('decimals', mfm.settings['parameter']['decimals'])

    def to_dict(self):
        v = mfm.Base.to_dict(self)
        v['value'] = self.value
        v['decimals'] = self.decimals
        return v

    def from_dict(self, v):
        mfm.Base.from_dict(self, v)
        self._value = v['value']
        self._decimals = v['decimals']

    def update(self):
        pass

    def finalize(self):
        pass


class FittingParameter(Parameter):

    def __init__(self, **kwargs):
        Parameter.__init__(self, **kwargs)
        self._link = kwargs.get('link', None)
        self.model = kwargs.get('model', None)

        self._lb = kwargs.get('lb', float('-inf'))
        self._ub = kwargs.get('ub', float('inf'))
        self._fixed = kwargs.get('fixed', False)
        self._bounds_on = kwargs.get('bounds_on', False)
        self._error_estimate = kwargs.get('error_estimate', None)

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
    def bounds(self):
        if self.bounds_on:
            return self._lb, self._ub
        else:
            return None, None

    @bounds.setter
    def bounds(self, b):
        self._lb, self._ub = b

    @property
    def bounds_on(self):
        return self._bounds_on

    @bounds_on.setter
    def bounds_on(self, v):
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
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, v):
        self._fixed = v

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, link):
        if isinstance(link, FittingParameter):
            self._link = link
        elif link is None:
            try:
                self._value = self._link.value
            except AttributeError:
                pass
            self._link = None

    @property
    def is_linked(self):
        return isinstance(self._link, FittingParameter)

    def to_dict(self):
        d = Parameter.to_dict(self)
        d['lb'], d['ub'] = self.bounds
        d['fixed'] = self.fixed
        d['bounds_on'] = self.bounds_on
        d['error_estimate'] = self.error_estimate
        return d

    def from_dict(self, d):
        Parameter.from_dict(self, d)
        self._lb, self._ub = d['lb'], d['ub']
        self._fixed = d['fixed']
        self._bounds_on = d['bounds_on']
        self._error_estimate = d['error_estimate']

    def scan(self, fit, **kwargs):
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

    def widget(self, **kwargs):
        text = kwargs.get('text', self.name)
        layout = kwargs.get('layout', None)
        update_widget = kwargs.get('update_widget', lambda x: x)
        decimals = kwargs.get('decimals', self.decimals)
        fixed = kwargs.get('fixed', self.fixed)
        lb, ub = self.bounds
        lb = kwargs.get('lb', lb)
        ub = kwargs.get('ub', ub)
        kw = {
            'layout': layout,
            'model': self.model,
            'text': text,
            'update_function': update_widget,
            'decimals': decimals,
            'fixed': fixed,
            'lb': lb,
            'ub': ub,
            'bounds_on': self.bounds_on,
            'value': self.value,
            'name': self.name
        }
        return FittingParameterWidget(**kw)


class FittingParameterWidget(QtGui.QWidget, FittingParameter):

    def make_linkcall(self, fit_idx, parameter_name):

        def linkcall():
            tooltip = " linked to " + parameter_name
            mfm.run("cs.current_fit.model.parameters_all_dict['%s'].link = mfm.fits[%s].model.parameters_all_dict['%s']" % (self.name, fit_idx, parameter_name))
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.update()
        return linkcall

    def contextMenuEvent(self, event):

        menu = QtGui.QMenu(self)
        menu.setTitle("Link " + self.name + " to:")

        for fit_idx, f in enumerate(mfm.fits):
            for fs in f:
                submenu = QtGui.QMenu(menu)
                submenu.setTitle(fs.name)

                # Sorted by "Aggregation"
                for a in fs.model.aggregated_parameters:
                    action_submenu = QtGui.QMenu(submenu)
                    action_submenu.setTitle(a.name)
                    ut = a.parameters
                    ut.sort(key=lambda x: x.name, reverse=False)
                    for p in ut:
                        if p is not self:
                            Action = action_submenu.addAction(p.name)
                            Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                    submenu.addMenu(action_submenu)
                action_submenu = QtGui.QMenu(submenu)

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

    def __init__(self, **kwargs):
        FittingParameter.__init__(self, **kwargs)
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/variable_widget.ui', self)
        self.widget_value = pg.SpinBox(dec=True, **kwargs)
        self.widget_value.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addWidget(self.widget_value)

        self.widget_lower_bound = pg.SpinBox(dec=True, **kwargs)
        self.widget_lower_bound.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_lower_bound)

        self.widget_upper_bound = pg.SpinBox(dec=True, **kwargs)
        self.widget_upper_bound.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_upper_bound)

        layout = kwargs.get('layout', None)
        if isinstance(layout, QtGui.QLayout):
            layout.addWidget(self)

        hide_bounds = kwargs.get('hide_bounds', parameter_settings['hide_bounds'])
        hide_link = kwargs.get('hide_link', parameter_settings['hide_link'])
        fixable = kwargs.get('fixable', parameter_settings['fixable'])
        hide_fix_checkbox = kwargs.get('hide_fix_checkbox', parameter_settings['fixable'])
        hide_error = kwargs.get('hide_error', parameter_settings['hide_error'])
        hide_label = kwargs.get('hide_label', parameter_settings['hide_label'])
        label_text = kwargs.get('text', self.name)

        # Hide and disable widgets
        self.label.setVisible(not hide_label)
        self.lineEdit.setVisible(not hide_error)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget_fix.setVisible(fixable or not hide_fix_checkbox)
        self.widget.setHidden(hide_bounds)
        self.widget_link.setDisabled(hide_link)

        # Display of values
        self.widget_value.setValue(float(self.value))

        self.label.setText(label_text.ljust(5))

        # variable bounds
        if not self.bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)

        # variable fixed
        if self.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        self.widget.hide()

        # The variable value
        self.widget_value.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].value = %s\n"
            "cs.current_fit.update()"
            % (self.name, self.widget_value.value()))
        )

        self.widget_fix.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].fixed = %s" % (self.name, self.widget_fix.isChecked()))
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds_on = %s" %
            (self.name, self.widget_bounds_on.isChecked()))
        )

        self.widget_lower_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_upper_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_link.clicked.connect(self.onLinkFitGroup)

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
           p.widget_link.setCheckState(QtCore.Qt.Checked)
           p.widget_value.setEnabled(False)
   except KeyError:
       pass
            """ % (self.name, self.name)
            mfm.run(t)
        elif cs == 0:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       p.link = None
       p.widget_value.setEnabled(True)
       p.widget_link.setCheckState(QtCore.Qt.Unchecked)
   except KeyError:
       pass
            """ % (self.name, self.name)
            mfm.run(t)
        self.blockSignals(False)

    def finalize(self, *args):
        QtGui.QWidget.update(self, *args)
        self.blockSignals(True)
        # Update value of widget
        self.widget_value.setValue(float(self.value))
        if self.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)

        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        # Tooltip
        s = "bound: (%s,%s)\n" % self.bounds if self.bounds_on else "bounds: off\n"
        if self.is_linked:
            s += "linked to: %s" % self.link.name
        self.widget_value.setToolTip(s)

        # Error-estimate
        value = self.value
        if self.fixed or not isinstance(self.error_estimate, float):
            self.lineEdit.setText("NA")
        else:
            rel_error = abs(self.error_estimate / (value + 1e-12) * 100.0)
            self.lineEdit.setText("%.0f%%" % rel_error)

        #link
        if self.link is not None:
            tooltip = " linked to " + self.link.name
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.blockSignals(False)


class AggregatedParameters(mfm.Base):

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            return self.__class__.__name__

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def parameters_all(self):
        return self._parameters

    @property
    def parameters_all_dict(self):
        return dict([(p.name, p) for p in self.parameters_all])

    @property
    def parameters(self):
        return self._parameters

    @property
    def aggregated_parameters(self):
        a = list()
        for p in self.__dict__:
            v = self.__dict__[p]
            if isinstance(v, AggregatedParameters):
                a.append(v)
        return a#self._aggregated_parameters

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

    @parameter_names.setter
    def parameter_names(self, v):
        self._parameterNames = v

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
            try:
                parameter_target[parameter_name].from_dict(parameter[parameter_name])
                parameter_target[parameter_name].update()
            except KeyError:
                print "Key %s not found skipping" % parameter_name

    def find_parameters(self, parameter_type=Parameter):
        self._aggregated_parameters = None
        self._parameters = None

        ag = mfm.find_objects(self.__dict__.values(), AggregatedParameters)
        self._aggregated_parameters = ag

        ap = list()
        for o in set(ag):
            if not isinstance(o, mfm.fitting.models.Model):
                o.find_parameters()
                self.__dict__[o.name] = o
                ap += o._parameters

        mp = mfm.find_objects(self.__dict__.values(), parameter_type)
        self._parameters = list(set(mp + ap))

    def finalize(self):
        pass

    def __len__(self):
        return len(self.parameters_all)

    def __init__(self, *args, **kwargs):
        mfm.Base.__init__(self, **kwargs)
        #DotMap.__init__(self, *args, **kwargs)
        self.short = kwargs.get('short', '')
        self.model = kwargs.get('model', None)
        self.fit = kwargs.get('fit', None)
        self._parameters = list()
        self._aggregated_parameters = list()


# class AggregatedParametersWidget(AggregatedParameters, QtGui.QGroupBox):
#
#     def __init__(self, *args, **kwargs):
#         AggregatedParameters.__init__(self, *args, **kwargs)
#         self.n_col = kwargs.get('n_cols', mfm.settings['gui']['fit_models']['n_columns'])
#         self.n_row = 0
#         QtGui.QGroupBox.__init__(self, self.name)
#         l = QtGui.QGridLayout()
#         l.setSpacing(0)
#         l.setMargin(0)
#         self.setLayout(l)
#
#     def add_parameter(self, parameter, layout):
#         l = layout
#         col = (len(self) - 1) % self.n_col
#         l.addWidget(parameter, self.n_row, col)
#         if col == 0:
#             self.n_row += 1


class GlobalParameter(FittingParameter):

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
        FittingParameter.__init__(self, **kwargs)
        self.f, self.g = f, g
        self.formula = formula