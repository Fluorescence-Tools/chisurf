"""

"""
from __future__ import annotations
from typing import Tuple

import os.path
from typing import Sequence, List
import numpy as np

import chisurf.base
import chisurf.curve
import chisurf.fio
import chisurf.experiments


class ExperimentalData(
    chisurf.base.Data
):
    """

    """

    @property
    def experiment(
            self
    ) -> chisurf.experiments.experiment.Experiment:
        if self._experiment is None:
            if isinstance(
                self.data_reader,
                chisurf.experiments.reader.ExperimentReader
            ):
                return self.data_reader.experiment
        else:
            return self._experiment

    @experiment.setter
    def experiment(
            self,
            v: chisurf.experiments.experiment.Experiment
    ) -> None:
        self._experiment = v

    def __init__(
            self,
            data_reader: chisurf.experiments.reader.ExperimentReader = None,
            experiment: chisurf.experiments.experiment.Experiment = None,
            *args,
            **kwargs
    ):
        """

        :param args:
        :param data_reader:
        :param experiment:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        self._experiment = experiment
        self.data_reader = data_reader

    def to_dict(self):
        d = super().to_dict()
        try:
            d['data_reader'] = self.data_reader.to_dict()
        except AttributeError:
            d['data_reader'] = None
        try:
            d['experiment'] = self.experiment.to_dict()
        except AttributeError:
            d['experiment'] = None
        return d


class DataCurve(
    chisurf.curve.Curve,
    ExperimentalData
):

    @property
    def data(
            self
    ) -> np.ndarray:
        return np.vstack(
            [
                self.x,
                self.y,
                self.ex,
                self.ey
            ]
        )

    @data.setter
    def data(
            self,
            v: np.ndarray
    ):
        self.set_data(
            *v
        )

    def __init__(
            self,
            x: np.array = None,
            y: np.array = None,
            ex: np.array = None,
            ey: np.array = None,
            filename: str = '',
            *args,
            **kwargs
    ):
        super().__init__(
            x=x,
            y=y,
            *args,
            **kwargs
        )
        if os.path.isfile(filename):
            self.load(
                filename,
                **kwargs
            )
        if ex is None:
            ex = np.ones_like(self.x)
        if ey is None:
            ey = np.ones_like(self.y)
        self.ex = ex
        self.ey = ey

    def __str__(self):
        s = "Dataset:\n"
        try:
            s += "filename: " + self.filename + "\n"
            s += "length  : %s\n" % len(self)
            s += "x\ty\terror-x\terror-y\n"

            lx = self.x[:3]
            ly = self.y[:3]
            lex = self.ex[:3]
            ley = self.ey[:3]

            ux = self.x[-3:]
            uy = self.y[-3:]
            uex = self.ex[-3:]
            uey = self.ey[-3:]

            for i in range(2):
                x, y, ex, ey = lx[i], ly[i], lex[i], ley[i]
                s += "{0:<12.3e}\t".format(x)
                s += "{0:<12.3e}\t".format(y)
                s += "{0:<12.3e}\t".format(ex)
                s += "{0:<12.3e}\t".format(ey)
                s += "\n"
            s += "....\n"
            for i in range(1):
                x, y, ex, ey = ux[i], uy[i], uex[i], uey[i]
                s += "{0:<12.3e}\t".format(x)
                s += "{0:<12.3e}\t".format(y)
                s += "{0:<12.3e}\t".format(ex)
                s += "{0:<12.3e}\t".format(ey)
                s += "\n"
        except (AttributeError, KeyError):
            s += "This curve does not 'own' data..."
        return s

    def to_dict(
            self
    ) -> dict:
        d = super().to_dict()
        d.update(ExperimentalData.to_dict(self))
        d['ex'] = list(self.ex)
        d['ey'] = list(self.ey)
        return d

    def from_dict(
            self,
            v: dict
    ) -> None:
        super().from_dict(v)
        self.ex = np.array(v['ex'], dtype=np.float64)
        self.ey = np.array(v['ey'], dtype=np.float64)

    def save(
            self,
            filename: str = None,
            file_type: str = 'json',
            **kwargs
    ) -> None:
        if filename is None:
            filename = os.path.join(
                self.name,
                '_data.txt'
            )
        self.filename = filename
        if file_type == 'txt':
            chisurf.fio.ascii.Csv().save(
                np.array(self[:]),
                filename=filename,
                **kwargs
            )
        else:
            with chisurf.fio.zipped.open_maybe_zipped(filename, 'w') as fp:
                fp.write(self.to_json())

    def load(
            self,
            filename: str,
            skiprows: int = 0,
            file_type: str = 'csv',
            **kwargs
    ) -> None:
        if file_type == 'csv':
            csv = chisurf.fio.ascii.Csv()
            csv.load(
                filename=filename,
                skiprows=skiprows,
                file_type=file_type,
                **kwargs
            )
            # First assume four columns
            # if this fails use three columns
            try:
                self.x = csv.data[0]
                self.y = csv.data[1]
                self.ex = csv.data[2]
                self.ey = csv.data[3]
            except IndexError:
                self.x = csv.data[0]
                self.y = csv.data[1]
                self.ey = csv.data[2]
                self.ex = np.ones_like(self.ey)

    def set_data(
            self,
            x: np.array,
            y: np.array,
            ex: np.array = None,
            ey: np.array = None,
    ) -> None:
        self.x = x
        self.y = y

        if ex is None:
            ex = np.ones_like(x)
        if ey is None:
            ey = np.ones_like(y)
        self.ex = ex
        self.ey = ey

    def set_weights(
            self,
            w: np.array
    ):
        self.ey = 1. / w

    def __getitem__(
            self,
            key: str
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        x, y = super().__getitem__(key)
        return x, y, self.ex[key], self.ey[key]


class DataGroup(
    list,
    chisurf.base.Base
):

    @property
    def names(
            self
    ) -> List[str]:
        return [d.name for d in self]

    @property
    def current_dataset(
            self
    ) -> chisurf.base.Data:
        return self[self._current_dataset]

    @current_dataset.setter
    def current_dataset(
            self,
            i: int
    ):
        self._current_dataset = i

    @property
    def name(
            self
    ) -> str:
        try:
            return self.__dict__['name']
        except KeyError:
            return self.names[0]

    @name.setter
    def name(
            self,
            v: str
    ) -> None:
        self._name = v

    def append(
            self,
            dataset: chisurf.base.Data
    ):
        if isinstance(dataset, ExperimentalData):
            list.append(self, dataset)
        if isinstance(dataset, list):
            for d in dataset:
                if isinstance(d, ExperimentalData):
                    list.append(self, d)

    def __init__(
            self,
            seq: Sequence,
            *args,
            **kwargs
    ):
        super().__init__(seq)
        self._current_dataset = 0


class DataCurveGroup(DataGroup):

    @property
    def x(self) -> np.array:
        return self.current_dataset.x

    @x.setter
    def x(self,
          v: np.array):
        self.current_dataset.x = v

    @property
    def y(self) -> np.array:
        return self.current_dataset.y

    @y.setter
    def y(self,
          v: np.array):
        self.current_dataset.y = v

    @property
    def ex(self) -> np.array:
        return self.current_dataset.ex

    @ex.setter
    def ex(self,
           v: np.array):
        self.current_dataset.ex = v

    @property
    def ey(self) -> np.array:
        return self.current_dataset.ey

    @ey.setter
    def ey(self,
           v: np.array):
        self.current_dataset.ey = v

    def __str__(self):
        return [str(d) + "\n------\n" for d in self]

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)


class ExperimentDataGroup(DataGroup):

    @property
    def setup(self):
        return self[0].setup

    @setup.setter
    def setup(self, v):
        pass

    @property
    def experiment(self):
        return self.setup.experiment

    @experiment.setter
    def experiment(self, v):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExperimentDataCurveGroup(
    ExperimentDataGroup,
    DataCurveGroup
):

    @property
    def setup(self):
        return self[0].setup

    @setup.setter
    def setup(self, v):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )
