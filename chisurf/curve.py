from __future__ import annotations
import typing

import numpy as np
import os

import chisurf.fio
import chisurf.base
import chisurf.decorators
import chisurf.math

T = typing.TypeVar('T', bound='Curve')


class Curve(
    chisurf.base.Base
):

    @property
    def fwhm(self) -> float:
        return chisurf.math.signal.calculate_fwhm(
            self
        )[0]

    @property
    def cdf(
            self
    ) -> Curve:
        """Cumulative distribution function
        """
        return self.__class__(
            x=self.x,
            y=np.cumsum(self.y)
        )

    @property
    def dx(self) -> np.ndarray:
        """
        The derivative of the x-axis
        """
        return np.diff(self.x)

    def save(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False,
            xmin: int = None,
            xmax: int = None
    ) -> None:
        super().save(
            filename=filename,
            file_type=file_type,
            verbose=verbose
        )
        if file_type == "csv":
            csv = chisurf.fio.ascii.Csv()
            x, y = self[xmin:xmax]
            csv.save(
                data=np.vstack([x, y]),
                filename=filename
            )

    def load(
            self,
            filename: str,
            skiprows: int = 0,
            file_type: str = 'csv',
            **kwargs
    ) -> None:
        super().load(
            filename=filename,
            file_type=file_type
        )
        if file_type == 'csv':
            csv = chisurf.fio.ascii.Csv()
            csv.load(
                filename=filename,
                skiprows=skiprows,
                file_type=file_type,
                **kwargs
            )
            try:
                self.x = csv.data[0]
                self.y = csv.data[1]
            except IndexError:
                self.x = csv.data[0]
                self.y = csv.data[1]

    def from_dict(
            self,
            v: dict
    ):
        super().from_dict(v)
        self.__dict__['y'] = np.array(v['y'], dtype=np.float64)
        self.__dict__['x'] = np.array(v['x'], dtype=np.float64)

    def __init__(
            self,
            x: np.array = None,
            y: np.array = None,
            copy_array: bool = True,
            *args,
            **kwargs
    ):
        if x is None:
            x = np.array(list(), dtype=np.float64)
        if y is None:
            y = np.array(list(), dtype=np.float64)
        if len(y) != len(x):
            raise ValueError(
                "length of x (%s) and y (%s) differ" % (len(x), len(y))
            )
        if copy_array:
            self.x = np.copy(x)
            self.y = np.copy(y)
        else:
            self.x = x
            self.y = y
        super().__init__(
            *args,
            **kwargs
        )

    def normalize(
            self,
            mode: str = "max",
            curve: chisurf.curve.Curve = None,
            inplace: bool = True
    ) -> float:
        """Calculates a scaling parameter for the Curve object and (optionally) scales the Curve object.

        :param mode: either 'max' to normalize the maximum to one, or 'sum' to normalize to sum to one
        :param curve:
        :param inplace: if True the Curve object is modified in place. Otherwise, only the scaling parameter
        is returned
        :return: the parameter that scales the Curve object
        """
        factor = 1.0
        if not isinstance(curve, Curve):
            if mode == "sum":
                factor = sum(self.y)
            elif mode == "max":
                factor = max(self.y)
        else:
            if mode == "sum":
                factor = sum(self.y) * sum(curve.y)
            elif mode == "max":
                if max(self.y) != 0:
                    factor = max(self.y) * max(curve.y)
        if inplace:
            self.y /= factor
        return factor

    def __add__(
            self,
            c: T
    ) -> Curve:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__add__(c)
        )

    def __sub__(
            self,
            c: T
    ) -> Curve:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__sub__(c)
        )

    def __mul__(
            self,
            c: T
    ) -> Curve:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__mul__(c)
        )

    def __truediv__(
            self,
            c: T
    ) -> Curve:
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__truediv__(c)
        )

    def __lshift__(
            self,
            shift: float
    ) -> Curve:
        return self.__class__(
            x=self.x,
            y=chisurf.math.signal.shift_array(
                self.y,
                shift
            )
        )

    def __len__(
            self
    ) -> int:
        return len(self.y)

    def __getitem__(
            self,
            key
    ) -> typing.Tuple[
        np.ndarray,
        np.ndarray
    ]:
        x = self.x.__getitem__(key)
        y = self.y.__getitem__(key)
        return x, y
