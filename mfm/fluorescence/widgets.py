from PyQt5 import QtCore, QtWidgets

import mfm
from mfm.fluorescence.anisotropy import Anisotropy


class AnisotropyWidget(Anisotropy, QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        Anisotropy.__init__(self, **kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setTitle("Rotational-times")
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
        self.rot_vis = False
        self._rho_widgets = list()
        self._b_widgets = list()

        self.radioButtonVM = QtWidgets.QRadioButton("VM")
        self.radioButtonVM.setToolTip("Excitation: Vertical\nDetection: Magic-Angle")
        self.radioButtonVM.setChecked(True)
        self.radioButtonVM.clicked.connect(lambda: mfm.run("cs.current_fit.model.anisotropy.polarization_type = 'vm'"))
        self.radioButtonVM.clicked.connect(self.hide_roation_parameters)

        self.radioButtonVV = QtWidgets.QRadioButton("VV")
        self.radioButtonVV.setToolTip("Excitation: Vertical\nDetection: Vertical")
        self.radioButtonVV.clicked.connect(lambda: mfm.run("cs.current_fit.model.anisotropy.polarization_type = 'vv'"))

        self.radioButtonVH = QtWidgets.QRadioButton("VH")
        self.radioButtonVH.setToolTip("Excitation: Vertical\nDetection: Horizontal")
        self.radioButtonVH.clicked.connect(lambda: mfm.run("cs.current_fit.model.anisotropy.polarization_type = 'vh'"))

        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)

        add_rho = QtWidgets.QPushButton()
        add_rho.setText("add")
        l.addWidget(add_rho)
        add_rho.clicked.connect(self.onAddRotation)

        remove_rho = QtWidgets.QPushButton()
        remove_rho.setText("del")
        l.addWidget(remove_rho)
        remove_rho.clicked.connect(self.onRemoveRotation)

        spacerItem = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        l.addItem(spacerItem)

        l.addWidget(self.radioButtonVM)
        l.addWidget(self.radioButtonVV)
        l.addWidget(self.radioButtonVH)

        self.lh.addLayout(l)

        self.gb = QtWidgets.QGroupBox()
        self.lh.addWidget(self.gb)
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)

        l = QtWidgets.QHBoxLayout()
        self._r0.make_widget(layout=l, text='r0', fixed=True)
        self._g.make_widget(layout=l, text='g', decimals=2, fixed=True)
        self.lh.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        self._l1.make_widget(layout=l, decimals=4, fixed=True, text='l1')
        self._l2.make_widget(layout=l, decimals=4, fixed=True, text='l2')
        self.lh.addLayout(l)

        self.lh.addLayout(l)
        self.add_rotation()
        self.hide_roation_parameters()

    def hide_roation_parameters(self):
        self.rot_vis = not self.rot_vis
        if self.rot_vis:
            self.gb.show()
        else:
            self.gb.hide()

    def onAddRotation(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.anisotropy.add_rotation()"
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onRemoveRotation(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.anisotropy.remove_rotation()"
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def add_rotation(self, **kwargs):
        Anisotropy.add_rotation(self, **kwargs)
        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        self.lh.addLayout(l)
        rho = self._rhos[-1].make_widget(layout=l, decimals=2)
        x = self._bs[-1].make_widget(layout=l, decimals=2)
        self._rho_widgets.append(rho)
        self._b_widgets.append(x)

    def remove_rotation(self):
        self._rhos.pop()
        self._bs.pop()
        self._rho_widgets.pop().close()
        self._b_widgets.pop().close()
