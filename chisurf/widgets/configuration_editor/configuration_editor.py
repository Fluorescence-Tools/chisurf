from __future__ import annotations
import typing

import sys

from qtpy import QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.parametertree
import qdarkstyle

import json
import re

import chisurf.fio
import chisurf.settings
import chisurf.widgets


def dict_to_parameter_tree(
        origin
) -> typing.List:
    """Creates an array from a dictionary that can be used to initialize a pyqtgraph parameter-tree
    :param origin:
    :return:
    """
    target = list()
    for key in origin.keys():
        if key.endswith('_options'):
            target[-1]['values'] = origin[key]
            target[-1]['type'] = 'list'
            continue
        d = dict()
        d['name'] = key
        if isinstance(origin[key], dict):
            d['type'] = 'group'
            d['children'] = dict_to_parameter_tree(origin[key])
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


def parameter_tree_to_dict(
        parameter_tree
) -> typing.Dict:
    """Converts a pyqtgraph parameter tree to an ordinary dictionary that could be saved
    as JSON file

    :param parameter_tree:
    :param target:
    :return:
    """
    target = dict()

    children = parameter_tree.children()
    for child in children:
        if child.type() == "action":
            continue
        if not child.children():
            value = child.opts['value']
            name = child.name()
            if isinstance(
                    value,
                    QtGui.QColor
            ):
                value = str(value.name())
            if isinstance(
                    child,
                    pg.parametertree.parameterTypes.ListParameter
            ):
                target[name + '_options'] = child.opts['values']
            target[name] = value
        else:
            target[child.name()] = parameter_tree_to_dict(child)
    return target


class ParameterEditor(QtWidgets.QWidget):

    def __init__(
            self,
            target: typing.Dict = None,
            json_file: str = None,
            windows_title: str = None,
    ):
        super().__init__()

        if json_file is None:
            json_file = chisurf.widgets.get_filename()
        if target is None:
            target = chisurf.settings.cs_settings
        if windows_title is None:
            windows_title = "Configuration: %s" % json_file

        self._json_file = json_file
        self._dict = dict()
        self._target = target
        self.json_file = json_file
        self._p = None

        self._p = pg.parametertree.Parameter.create(
            name='params',
            type='group',
            children=self.parameter_dict,
            expanded=True
        )
        self._p.param('Save').sigActivated.connect(self.save)

        t = pg.parametertree.ParameterTree()
        t.setParameters(self._p, showTop=False)

        self.setWindowTitle(windows_title)
        win = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        win.setLayout(layout)
        layout.addWidget(t, 1, 0, 1, 1)
        win.show()
        win.resize(450, 400)
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)
        layout.addWidget(t, 1, 0, 1, 1)

        self.resize(450, 400)

    def save(
            self,
            event: QtWidgets.QAction = None,
            filename: str = None,
    ):
        if filename is None:
            filename = self._json_file
        with open(filename, 'w+') as fp:
            obj = self.dict
            json.dump(obj, fp, indent=4)
        self._target.settings = obj

    @property
    def dict(self) -> dict:
        if self._p is not None:
            print("dict")
            return parameter_tree_to_dict(self._p)
        else:
            print("dict2")
            return self._dict

    @property
    def parameter_dict(self) -> typing.List:
        od = dict(self.dict)
        params = dict_to_parameter_tree(od)
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
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=v,
                mode='r'
        ) as fp:
            self._dict = json.load(fp)
        self._json_file = v


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ParameterEditor()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
