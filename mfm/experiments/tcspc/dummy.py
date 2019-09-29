import os

import numpy as np
from qtpy import QtWidgets, uic

import mfm
from mfm.experiments.tcspc import TCSPCReader


class TCSPCSetupDummy(
    TCSPCReader
):

    name = "Dummy-TCSPC"

    def __init__(
            self,
            *args,
            n_tac: int = 4096,
            dt: float = 0.0141,
            p0: float = 10000.0,
            rep_rate: float = 10.0,
            lifetime: float = 4.1,
            name: str = 'Dummy',
            sample_name: str = 'TCSPC-Dummy',
            parent: QtWidgets.QWidget = None,
            verbose: bool = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        TCSPCReader.__init__(self, **kwargs)
        self.parent = parent

        if verbose is None:
            verbose = mfm.verbose
        self.verbose = verbose

        self.sample_name = sample_name
        self.name = name
        self.lifetime = lifetime
        self.n_tac = n_tac
        self.dt = dt
        self.p0 = p0
        self.rep_rate = rep_rate

    def read(
            self,
            filename: str = None,
            **kwargs
    ):
        if filename is None:
            filename = self.sample_name

        x = np.arange(self.n_tac) * self.dt
        y = np.exp(-x/self.lifetime) * self.p0
        ey = 1./mfm.fluorescence.tcspc.weights(y)

        d = mfm.experiments.data.DataCurve(
            x=x,
            y=y,
            ey=ey,
            setup=self,
            name=filename
        )
        d.setup = self

        return d

    def __str__(self):
        s = 'TCSPCSetup: Dummy\n'
        return s


class TCSPCSetupDummyWidget(
    QtWidgets.QWidget,
    TCSPCSetupDummy
):

    @property
    def sample_name(
            self
    ) -> str:
        name = str(self.lineEdit.text())
        return name

    @sample_name.setter
    def sample_name(
            self,
            v: str
    ):
        pass

    @property
    def p0(
            self
    ) -> int:
        return self.spinBox_2.value()

    @p0.setter
    def p0(
            self,
            v: int
    ):
        pass

    @property
    def lifetime(
            self
    ) -> float:
        return self.doubleSpinBox_2.value()

    @lifetime.setter
    def lifetime(
            self,
            v: float
    ):
        pass

    @property
    def n_tac(
            self
    ) -> int:
        return self.spinBox.value()

    @n_tac.setter
    def n_tac(
            self,
            v: int
    ):
        pass

    @property
    def dt(
            self
    ) -> float:
        return self.doubleSpinBox.value()

    @dt.setter
    def dt(
            self,
            v: float
    ):
        pass

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "tcspcDummy.ui"
            ),
            self
        )