from __future__ import annotations

import pathlib

from chisurf import typing
import numpy as np
import yaml

import chisurf.base
import chisurf.curve
import chisurf.fio
import chisurf.fio.ascii
import chisurf.experiments


class ExperimentalData(chisurf.base.Data):

    meta_data: typing.Dict = None
    data_reader: chisurf.experiments.reader.ExperimentReader = None
    _experiment: chisurf.experiments.experiment.Experiment = None

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
            filename: str = "None",
            data: bytes = None,
            embed_data: bool = None,
            read_file_size_limit: int = None,
            name: object = None,
            verbose: bool = False,
            unique_identifier: str = None,
            **kwargs
    ):
        """

        :param args:
        :param data_reader:
        :param experiment:
        :param kwargs:
        """
        super().__init__(
            filename=filename,
            data=data,
            embed_data=embed_data,
            read_file_size_limit=read_file_size_limit,
            name=name,
            verbose=verbose,
            unique_identifier=unique_identifier,
            **kwargs
        )
        self._experiment = experiment
        self.data_reader = data_reader

    def to_dict(
            self,
            remove_protected: bool = True,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ):
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        try:
            d['data_reader'] = self.data_reader.to_dict(
                remove_protected=remove_protected,
                copy_values=copy_values,
                convert_values_to_elementary=convert_values_to_elementary
            )
        except AttributeError:
            d['data_reader'] = None
        try:
            d['experiment'] = self.experiment.to_dict(
                remove_protected=remove_protected,
                copy_values=copy_values,
                convert_values_to_elementary=convert_values_to_elementary
            )
        except AttributeError:
            d['experiment'] = None
        return d


class DataCurve(chisurf.curve.Curve, ExperimentalData):

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
    def data(self, v: np.ndarray):
        self.set_data(*v)

    def __init__(
            self,
            x: np.ndarray = None,
            y: np.ndarray = None,
            ex: np.ndarray = None,
            ey: np.ndarray = None,
            copy_array: bool = True,
            filename: str = '',
            data_reader: chisurf.experiments.reader.ExperimentReader = None,
            experiment: chisurf.experiments.experiment.Experiment = None,
            load_filename_on_init: bool = True,
            *args,
            **kwargs
    ):
        super().__init__(
            x=x,
            y=y,
            copy_array=copy_array,
            data_reader=data_reader,
            experiment=experiment,
            *args,
            **kwargs
        )
        if load_filename_on_init:
            if pathlib.Path(filename).is_file():
                self.load(filename, **kwargs)

        # Compute errors
        self.ex: np.ndarray = None
        self.ey: np.ndarray = None
        if not isinstance(ex, np.ndarray):
            ex = np.zeros_like(self.x)
        if not isinstance(ey, np.ndarray):
            ey = np.ones_like(self.y)
        self.ex = np.copy(ex) if copy_array else ex
        self.ey = np.copy(ey) if copy_array else ey

    def __str__(self):
        s = "Dataset:\n"
        try:
            s += "filename: " + self.filename + "\n"
            s += "length  : %s\n" % len(self)
            s += "x\ty\terror-x\terror-y\n"

            if len(self.x) > 10:
                lx = self.x[:4]
                ly = self.y[:4]
                lex = self.ex[:4]
                ley = self.ey[:4]
                for i in range(3):
                    x, y, ex, ey = lx[i], ly[i], lex[i], ley[i]
                    s += "{0:<12.3e}\t".format(x)
                    s += "{0:<12.3e}\t".format(y)
                    s += "{0:<12.3e}\t".format(ex)
                    s += "{0:<12.3e}\t".format(ey)
                    s += "\n"
                s += "....\n"
                ux = self.x[-4:]
                uy = self.y[-4:]
                uex = self.ex[-4:]
                uey = self.ey[-4:]
                for i in range(2):
                    x, y, ex, ey = ux[i], uy[i], uex[i], uey[i]
                    s += "{0:<12.3e}\t".format(x)
                    s += "{0:<12.3e}\t".format(y)
                    s += "{0:<12.3e}\t".format(ex)
                    s += "{0:<12.3e}\t".format(ey)
                    s += "\n"
            else:
                for i in range(len(self.x)):
                    x, y, ex, ey = self.x[i], self.y[i], self.ex[i], self.ey[i]
                    s += "{0:<12.3e}\t".format(x)
                    s += "{0:<12.3e}\t".format(y)
                    s += "{0:<12.3e}\t".format(ex)
                    s += "{0:<12.3e}\t".format(ey)
        except (AttributeError, KeyError):
            s += "This curve does not data..."
        return s

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ) -> typing.Dict:
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        d['ex'] = self.ex.tolist()
        d['ey'] = self.ey.tolist()
        return d

    def from_dict(
            self,
            v: typing.Dict
    ) -> None:
        super().from_dict(v)
        self.ex = np.array(v['ex'], dtype=np.float64)
        self.ey = np.array(v['ey'], dtype=np.float64)

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
            n_col, _ = csv.data.shape
            if n_col == 1:
                self.x = csv.data[0]
                self.y = np.ones_like(self.x)
                self.ex = np.zeros_like(self.x)
                self.ey = np.ones_like(self.y)
            elif n_col == 2:
                self.x = csv.data[0]
                self.y = csv.data[1]
                self.ex = np.zeros_like(self.x)
                self.ey = np.ones_like(self.y)
            elif n_col == 3:
                self.x = csv.data[0]
                self.y = csv.data[1]
                self.ex = np.zeros_like(self.x)
                self.ey = csv.data[2]
            elif n_col == 4:
                self.x = csv.data[0]
                self.y = csv.data[1]
                self.ex = csv.data[2]
                self.ey = csv.data[3]
            else:
                self.x = np.ones(1)
                self.y = np.ones(1)
                self.ex = np.ones(1)
                self.ey = np.ones(1)
        else:
            super().load(
                filename=filename,
                file_type=file_type,
                **kwargs
            )

    def save(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False,
            xmin: int = None,
            xmax: int = None
    ) -> None:
        self.filename = filename
        if file_type == "csv":
            csv = chisurf.fio.ascii.Csv()
            x, y, ex, ey = self[xmin:xmax]
            csv.save(
                data=np.vstack([x, y, ex, ey]),
                filename=filename
            )
        else:
            super().save(
                filename=filename,
                file_type=file_type,
                verbose=verbose
            )

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
    ) -> typing.Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]:
        x, y = super().__getitem__(key)
        return x, y, self.ex[key], self.ey[key]


class DataGroup(list, chisurf.base.Base):

    @property
    def names(self) -> typing.List[str]:
        return [d.name for d in self]

    @property
    def current_dataset(self) -> chisurf.base.Data:
        return self[self._current_dataset]

    @current_dataset.setter
    def current_dataset(self, i: int):
        self._current_dataset = i

    @property
    def name(self) -> str:
        try:
            return self.__dict__['name']
        except KeyError:
            return self.names[self._current_dataset]

    def to_yaml(
            self,
            remove_protected: bool = False,
            convert_values_to_elementary: bool = True
    ):
        d = self.to_dict(
            remove_protected=remove_protected,
            convert_values_to_elementary=convert_values_to_elementary
        )
        data = [
            d.to_dict(
                remove_protected=remove_protected,
                convert_values_to_elementary=convert_values_to_elementary
            ) for d in self
        ]
        d['data'] = data
        return yaml.dump(data=d)

    def append(self, dataset: chisurf.base.Data):
        if isinstance(dataset, ExperimentalData):
            list.append(self, dataset)
        if isinstance(dataset, list):
            for d in dataset:
                if isinstance(d, ExperimentalData):
                    list.append(self, d)

    def __init__(
            self,
            seq: typing.Sequence,
            *args,
            **kwargs
    ):
        self._current_dataset: int = 0
        super().__init__(seq)


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
    def ex(self, v: np.array):
        self.current_dataset.ex = v

    @property
    def ey(self) -> np.array:
        return self.current_dataset.ey

    @ey.setter
    def ey(self, v: np.array):
        self.current_dataset.ey = v

    def __str__(self):
        return [str(d) + "\n------\n" for d in self]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExperimentDataGroup(DataGroup):

    @property
    def setup(self):
        return self[self._current_dataset].setup

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


class ExperimentDataCurveGroup(ExperimentDataGroup, DataCurveGroup):

    @property
    def setup(self):
        return self[0].setup

    @setup.setter
    def setup(self, v):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_data(
        curve_type: str = 'experiment',
        data_set: typing.List[
            chisurf.data.ExperimentalData
        ] = None,
        excludes_names: typing.List[str] = None
) -> typing.List[
    chisurf.data.ExperimentalData
]:
    """Returns all curves `chisurf.curve.DataCurve` except curve that are
    excluded by their name

    Parameters
    ----------
    curve_type : str
        if this value is set to `experiment` only curves
        that are experimental curves, i.e., curves that inherit from
        `experiments.data.ExperimentalData` are returned.
    data_set : list
        A list containing the
    excludes_names : list
        A list containing names that should be excluded (default:
        ["Global-fit"]).

    Returns
    -------
    list
        A list containing curves. If `curve_type` is 'experiment' only
        curves that inherit from `experiments.data.ExperimentalData` or
        groups inheriting from ExperimentDataGroup are returned.

    Examples
    --------


    """
    if excludes_names is None:
        excludes_names = ["Global-fit"]
    if curve_type == 'experiment':
        return [
            d for d in data_set if (
                    (
                            isinstance(d, ExperimentalData) or
                            isinstance(d, ExperimentDataGroup)
                    ) and
                    d.name not in excludes_names
            )
        ]
    else: #elif curve_type == 'all':
        return [
            d for d in data_set if
            isinstance(d, ExperimentalData) or
            isinstance(d, ExperimentDataGroup)
        ]