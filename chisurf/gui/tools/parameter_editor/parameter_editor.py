from __future__ import annotations

import pathlib

from chisurf import typing

import sys

import chisurf.fio
import chisurf.settings
import chisurf.gui.widgets
from chisurf.gui import QtGui, QtWidgets

import pyqtgraph as pg
import pyqtgraph.parametertree
import pyqtgraph.parametertree.parameterTypes

import json
import re

import scikit_fluorescence as skf
import scikit_fluorescence.io



def dict_to_parameter_tree(origin) -> typing.List:
    """Creates an array from a dictionary that can be used to initialize a pyqtgraph parameter-tree
    :param origin:
    :return:
    """
    target = list()
    for key in origin.keys():
        d = dict()
        d['name'] = key
        if isinstance(origin[key], dict):
            d['type'] = 'group'
            d['children'] = dict_to_parameter_tree(origin[key])
        elif callable(origin[key]):
            d['type'] = 'str'
            d['value'] = origin[key].__name__
            d['call'] = origin[key]
            d['expanded'] = False
        elif isinstance(origin[key], list):
            d['type'] = 'list'
            d['expanded'] = True
            d['values'] = origin[key]
        else:
            value = origin[key]
            t = value.__class__.__name__
            if t == 'unicode':
                t = 'str'
            d['type'] = t
            d['value'] = value
            d['expanded'] = False
        target.append(d)
    return target


def parameter_tree_to_dict(parameter_tree) -> typing.Dict:
    """Converts a pyqtgraph parameter tree to an ordinary dictionary that could be saved
    as JSON file
    """
    target = dict()

    children = parameter_tree.children()
    for child in children:
        if child.type() == "action":
            continue
        if not child.children():
            name = child.name()
            if child.opts.get('call', None):
                target[name] = child.opts['call']
            elif isinstance(child, pg.parametertree.parameterTypes.ListParameter):
                values = child.opts['values']
                values = [None if value == 'None' else value for value in values]
                target[name] = values
            elif isinstance(child.opts['value'], QtGui.QColor):
                target[name] = str(child.opts['value'].name())
            else:
                value = child.opts['value']
                if value == 'None':
                    value = None
                target[name] = value
        else:
            target[child.name()] = parameter_tree_to_dict(child)
    return target


class ParameterEditor(QtWidgets.QWidget):

    def __init__(
            self,
            target: typing.Dict = None,
            json_file: pathlib.Path = None,
            windows_title: str = None,
            callback: typing.Callable = None,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if json_file is None:
            json_file = chisurf.gui.widgets.get_filename()
        if target is None:
            target = chisurf.settings.cs_settings
        if windows_title is None:
            windows_title = "Configuration: %s" % json_file

        self.callback = callback
        self._json_file = json_file
        self._dict = dict()
        self._target = target

        if pathlib.Path(json_file).is_file():
            self.json_file = json_file
        else:
            self._dict = target

        self._p = None

        self.create_parameter()

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.setWindowTitle(windows_title)

        t = pg.parametertree.ParameterTree()
        t.setParameters(self._p, showTop=False)
        self.t = t
        layout.addWidget(t, 1, 0, 1, 1)

        if self._p is not None and self.callback is not None:
            self.connect_value_changed(self._p, self.callback)

    def create_parameter(self):
        self._p = pg.parametertree.Parameter.create(
            name='Parameter',
            type='group',
            children=self.parameter_dict,
            expanded=True
        )
        if pathlib.Path(self._json_file).is_file():
            self._p.param('Save').sigActivated.connect(self.save)

    def connect_value_changed(self, param, value_changed):
        if isinstance(param, pg.parametertree.Parameter):
            param.sigValueChanged.connect(value_changed)
            if isinstance(param, pg.parametertree.parameterTypes.GroupParameter):
                for child in param:
                    self.connect_value_changed(child, self.callback)

    def update(self):
        self.clear()
        self.create_parameter()
        self.t.setParameters(self._p)
        if self.callback is not None:
            self.connect_value_changed(self._p, self.callback)

    def clear(self):
        self._p = None
        self.t.clear()

    def save(self, event: QtWidgets.QAction = None, filename: str = None):
        if filename is None:
            filename = self._json_file
        with open(filename, 'w+') as fp:
            obj = self.dict
            json.dump(obj, fp, indent=4)
        self._target = obj

    @property
    def dict(self) -> dict:
        if self._p is not None:
            return parameter_tree_to_dict(self._p)
        else:
            return self._dict

    @property
    def parameter_dict(self) -> typing.List:
        od = dict(self.dict)
        params = dict_to_parameter_tree(od)
        if pathlib.Path(self.json_file).is_file():
            params.append(
                {
                    'name': 'Save',
                    'type': 'action'
                }
            )
        return params

    @property
    def json_file(self) -> str:
        return self._json_file

    @json_file.setter
    def json_file(self, v: str):
        with skf.io.zipped.open_maybe_zipped(v, mode='r') as fp:
            self._dict = json.load(fp)
        self._json_file = v


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ParameterEditor()
    win.show()
    sys.exit(app.exec_())
