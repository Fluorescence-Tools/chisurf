"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types

"""
from __future__ import annotations
from typing import Dict

import sys

from qtpy import QtGui, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import ListParameter
import qdarkstyle

import yaml
import re
from collections import OrderedDict

import mfm


def dict2pt(target, origin):
    """Creates an array from a dictionary that can be used to initialize a pyqtgraph parameter-tree

    :param target:
    :param origin:
    :return:
    """
    for key in origin.keys():
        if key.endswith('_options'):
            target[-1]['values'] = origin[key]
            target[-1]['type'] = 'list'
            continue
        d = dict()
        d['name'] = key
        if isinstance(origin[key], dict):
            d['type'] = 'group'
            d['children'] = dict2pt(list(), origin[key])
        else:
            value = origin[key]
            type = value.__class__.__name__
            if type == 'unicode':
                type = 'str'
            iscolor = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', str(value))
            if iscolor:
                type = 'color'
            # {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
            d['type'] = type
            d['value'] = value
            d['expanded'] = False
        target.append(d)
    return target


def pt2dict(
        parameter_tree,
        target=OrderedDict()
):
    """Converts a pyqtgraph parameter tree to an ordinary dictionary that could be saved
    as JSON file

    :param parameter_tree:
    :param target:
    :return:
    """
    children = parameter_tree.children()
    for child in children:
        if child.type() == "action":
            continue
        if not child.children():
            value = child.opts['value']
            name = child.name()
            if isinstance(value, QtGui.QColor):
                value = str(value.name())
            if isinstance(child, ListParameter):
                target[name + '_options'] = child.opts['values']
            target[name] = value
        else:
            target[child.name()] = pt2dict(child, OrderedDict())
    return target


class ParameterEditor(QtWidgets.QWidget):

    def __init__(
            self,
            target: Dict = None,
            json_file: str = None,
            windows_title: str = None,
    ):
        super(ParameterEditor, self).__init__()

        if json_file is None:
            json_file = mfm.widgets.get_filename()
        if target is None:
            target = mfm

        if windows_title is None:
            windows_title = "Configuration: %s" % json_file

        self._p = Parameter.create(
            name='params',
            type='group',
            children=self.parameter_dict,
            expanded=True
        )
        self._p.param('Save').sigActivated.connect(self.save)

        self._json_file = json_file
        self._dict = dict()
        self._target = target
        self.json_file = json_file
        self._p = None

        t = ParameterTree()
        t.setParameters(self._p, showTop=False)

        self.setWindowTitle(windows_title)
        win = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        win.setLayout(layout)
        layout.addWidget(t, 1, 0, 1, 1)
        win.show()
        win.resize(450, 400)
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(t, 1, 0, 1, 1)
        self.resize(450, 400)

    def save(
            self,
            filename: str = None,
    ):
        if filename is None:
            filename = self._json_file
        with open(filename, 'w+') as fp:
            obj = self.dict
            yaml.dump(obj, fp, indent=4)
        self._target.settings = obj

    @property
    def dict(self) -> dict:
        if self._p is not None:
            return pt2dict(self._p, OrderedDict())
        else:
            return self._dict

    @property
    def parameter_dict(self) -> dict:
        od = OrderedDict(self.dict)
        params = dict2pt(list(), od)
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
    def json_file(
            self,
            v: str
    ):
        with open(v, 'r') as fp:
            self._dict = yaml.safe_load(fp)
        self._json_file = v


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ParameterEditor()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
