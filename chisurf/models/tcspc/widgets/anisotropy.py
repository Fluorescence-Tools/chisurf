from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting

from chisurf.models.tcspc.anisotropy import Anisotropy


class AnisotropyWidget(Anisotropy, QtWidgets.QGroupBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setTitle("Rotational-times")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.setLayout(self.lh)
        self.rot_vis = False
        self._rho_widgets = list()
        self._b_widgets = list()

        self.radioButtonVM = QtWidgets.QRadioButton("VM")
        self.radioButtonVM.setToolTip(
            "Excitation: Vertical\nDetection: Magic-Angle"
        )
        self.radioButtonVM.setChecked(True)
        self.radioButtonVM.clicked.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.anisotropy.polarization_type = 'vm'"
            )
        )
        self.radioButtonVM.clicked.connect(self.hide_roation_parameters)

        self.radioButtonVV = QtWidgets.QRadioButton("VV")
        self.radioButtonVV.setToolTip(
            "Excitation: Vertical\nDetection: Vertical"
        )
        self.radioButtonVV.clicked.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.anisotropy.polarization_type = 'vv'"
            )
        )
        self.radioButtonVV.clicked.connect(self.hide_roation_parameters)

        self.radioButtonVH = QtWidgets.QRadioButton("VH")
        self.radioButtonVH.setToolTip(
            "Excitation: Vertical\nDetection: Horizontal"
        )
        self.radioButtonVH.clicked.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.anisotropy.polarization_type = 'vh'"
            )
        )
        self.radioButtonVH.clicked.connect(self.hide_roation_parameters)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        add_rho = QtWidgets.QPushButton()
        add_rho.setText("add")
        layout.addWidget(add_rho)
        add_rho.clicked.connect(self.onAddRotation)

        remove_rho = QtWidgets.QPushButton()
        remove_rho.setText("del")
        layout.addWidget(remove_rho)
        remove_rho.clicked.connect(self.onRemoveRotation)

        spacerItem = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        layout.addItem(spacerItem)

        layout.addWidget(self.radioButtonVM)
        layout.addWidget(self.radioButtonVV)
        layout.addWidget(self.radioButtonVH)

        self.lh.addLayout(layout)

        self.gb = QtWidgets.QGroupBox()
        self.lh.addWidget(self.gb)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)
        self.gb.setLayout(self.lh)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._r0,
            label_text='r<sub>0</sub>',
            layout=layout
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._g,
            label_text='g',
            layout=layout
        )
        self.lh.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._l1,
            label_text='l<sub>1</sub>',
            layout=layout,
            decimals=4
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._l2,
            label_text='l<sub>2</sub>',
            layout=layout,
            decimals=4
        )
        self.lh.addLayout(layout)

        self.lh.addLayout(layout)
        self.add_rotation()
        self.hide_roation_parameters()

    def hide_roation_parameters(self):
        # Hide rotation parameters when VM is selected, show otherwise
        if self.radioButtonVM.isChecked():
            self.gb.hide()
        else:
            self.gb.show()

    def onAddRotation(self):
        chisurf.run(
            "\n".join(
                [
                    "for f in cs.current_fit:",
                    "   f.model.anisotropy.add_rotation()",
                    "cs.current_fit.update()"
                ]
            )
        )

    def onRemoveRotation(self):
        chisurf.run(
            "\n".join(
                [
                    "for f in cs.current_fit:",
                    "   f.model.anisotropy.remove_rotation()",
                    "cs.current_fit.update()"
                ]
            )
        )

    def add_rotation(self, **kwargs):
        super().add_rotation(**kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.lh.addLayout(layout)
        self._b_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._bs[-1],
                decimals=4,
                layout=layout
            )
        )
        self._rho_widgets.append(
            chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._rhos[-1],
                decimals=4,
                layout=layout
            )
        )

    def remove_rotation(self):
        self._rhos.pop()
        self._bs.pop()
        self._rho_widgets.pop().close()
        self._b_widgets.pop().close()