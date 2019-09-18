from __future__ import annotations

import os

from qtpy import QtWidgets, uic


class AVProperties(QtWidgets.QWidget):

    def __init__(
            self,
            av_type: str = "AV1"
    ):
        super(AVProperties, self).__init__()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "av_property.ui"
            ),
            self
        )
        self._av_type = av_type
        self.av_type = av_type
        self.groupBox.hide()

    @property
    def av_type(
            self
    ) -> str:
        return self._av_type

    @av_type.setter
    def av_type(
            self,
            v: str
    ):
        self._av_type = v
        if v == 'AV1':
            self.label_4.setEnabled(False)
            self.label_5.setEnabled(False)
            self.doubleSpinBox_4.setEnabled(False)
            self.doubleSpinBox_5.setEnabled(False)
        if v == 'AV0':
            self.doubleSpinBox_4.setEnabled(False)
            self.doubleSpinBox_5.setEnabled(False)
        elif v == 'AV3':
            self.label_4.setEnabled(True)
            self.label_5.setEnabled(True)
            self.doubleSpinBox_4.setEnabled(True)
            self.doubleSpinBox_5.setEnabled(True)

    @property
    def linker_length(
            self
    ) -> float:
        return float(self.doubleSpinBox.value())

    @linker_length.setter
    def linker_length(
            self,
            v: float
    ):
        self.doubleSpinBox.setValue(v)

    @property
    def linker_width(
            self
    ) -> float:
        return float(self.doubleSpinBox_2.value())

    @linker_width.setter
    def linker_width(
            self,
            v: float
    ):
        self.doubleSpinBox_2.setValue(v)

    @property
    def radius_1(
            self
    ) -> float:
        return float(self.doubleSpinBox_3.value())

    @radius_1.setter
    def radius_1(
            self,
            v: float
    ):
        self.doubleSpinBox_3.setValue(v)

    @property
    def radius_2(self):
        return float(self.doubleSpinBox_4.value())

    @radius_2.setter
    def radius_2(
            self,
            v: float
    ):
        self.doubleSpinBox_4.setValue(v)

    @property
    def radius_3(
            self
    ) -> float:
        return float(self.doubleSpinBox_5.value())

    @radius_3.setter
    def radius_3(
            self,
            v: float
    ):
        self.doubleSpinBox_5.setValue(v)

    @property
    def resolution(
            self
    ) -> float:
        return float(self.doubleSpinBox_6.value())

    @resolution.setter
    def resolution(
            self,
            v: float
    ):
        self.doubleSpinBox_6.setValue(v)

    @property
    def initial_linker_sphere(
            self
    ) -> float:
        return float(self.doubleSpinBox_7.value())

    @initial_linker_sphere.setter
    def initial_linker_sphere(
            self,
            v: float
    ):
        self.doubleSpinBox_7.setValue(v)

    @property
    def initial_linker_sphere_min(
            self
    ) -> float:
        return float(self.doubleSpinBox_8.value())

    @initial_linker_sphere_min.setter
    def initial_linker_sphere_min(
            self,
            v: float
    ):
        self.doubleSpinBox_8.setValue(v)

    @property
    def initial_linker_sphere_max(
            self
    ) -> float:
        return float(self.doubleSpinBox_9.value())

    @initial_linker_sphere_max.setter
    def initial_linker_sphere_max(
            self,
            v: float
    ):
        self.doubleSpinBox_9.setValue(v)

