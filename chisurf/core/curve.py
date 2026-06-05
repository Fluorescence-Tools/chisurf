from __future__ import annotations
from chisurf import typing

import abc
import numpy as np

import chisurf.core.fio
import chisurf.core.fio.ascii
import chisurf.core.base
import chisurf.core.decorators
import chisurf.core.math


T = typing.TypeVar('T', bound='Curve')


class NCurve(chisurf.core.base.Base):
    """Base class for 1D numeric arrays.

    This class stores a single NumPy array ``d`` and provides basic
    slicing/serialization support. Subclasses such as :class:`Curve`
    interpret the array in more structured ways.
    """

    def __init__(
            self,
            d: np.ndarray = None,
            copy_array: bool = True,
            *args,
            **kwargs
    ):
        """Initialize an NCurve with an optional 1D numpy array.

        Parameters
        ----------
        d : np.ndarray, optional
            Data array.
        copy_array : bool
            If True (default), the array is copied.
        """
        if d is None:
            self.d = np.array(list(), dtype=np.float64)
        if copy_array:
            self.d = np.atleast_1d(np.copy(d))
        else:
            self.d = d
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        """Return the instance ``__dict__`` for pickling."""
        state = self.__dict__.copy()
        return state

    def __getitem__(self, key) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Index into the flattened data array.

        Parameters
        ----------
        key : int, slice, or np.ndarray
            Index.

        Returns
        -------
        tuple
            ``(x, y)`` where *x* is an index array and *y* is the
            selected data values.
        """
        y = self.d.flatten().__getitem__(key)
        x = np.arange(0, len(self.y))
        return x, y


class Curve(NCurve):
    """Simple 1D curve represented by paired ``(x, y)`` arrays.

    The underlying storage ``d`` is a 2×N array with ``d[0] = x`` and
    ``d[1] = y``. The class provides basic arithmetic, slicing and
    (de-)serialization helpers.

    Examples
    --------
    Construct a small curve and access its data:

    >>> import numpy as np
    >>> from chisurf.core.curve import Curve
    >>> x = np.array([0.0, 1.0, 2.0])
    >>> y = np.array([1.0, 2.0, 3.0])
    >>> c = Curve(x=x, y=y)
    >>> len(c)
    3
    >>> float(c.y[0])
    1.0
    """

    @property
    def fwhm(self) -> float:
        """Full width at half maximum of the curve.

        The calculation is delegated to
        :func:`chisurf.core.math.signal.calculate_fwhm`.
        """
        v, _, _ = chisurf.core.math.signal.calculate_fwhm(
            x_values=self.x,
            y_values=self.y
        )
        return v

    @property
    def cdf(self) -> Curve:
        """Return the cumulative distribution function of ``y``.

        The returned object is a new :class:`Curve` with the same ``x``
        grid and ``y`` replaced by ``np.cumsum(self.y)``.

        Examples
        --------
        >>> import numpy as np
        >>> from chisurf.core.curve import Curve
        >>> c = Curve(x=np.array([0., 1., 2.]), y=np.array([1., 2., 3.]))
        >>> c.cdf.y[-1]
        6.0
        """
        return self.__class__(
            x=self.x,
            y=np.cumsum(self.y)
        )

    @property
    def x(self) -> np.ndarray:
        """Abscissa array of the curve."""
        return self.d[0]

    @x.setter
    def x(self, v):
        """Set the curve's x-values (broadcast into the existing storage)."""
        self.d[0] = v

    @property
    def y(self) -> np.ndarray:
        """Ordinate array of the curve."""
        return self.d[1]

    @y.setter
    def y(self, v):
        """Set the curve's y-values (broadcast into the existing storage)."""
        self.d[1] = v

    @property
    def dx(self) -> np.ndarray:
        """First differences of the ``x`` array (``np.diff(self.x)``)."""
        return np.diff(self.x)

    def save(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False,
            x_min: int = None,
            x_max: int = None
    ) -> None:
        """Save the curve to disk via :mod:`chisurf.core.fio`.

        When ``file_type == 'csv'`` the data are written as two rows
        ``[x, y]`` using :class:`chisurf.core.fio.ascii.Csv`.
        """
        super().save(
            filename=filename,
            file_type=file_type,
            verbose=verbose
        )
        if file_type == "csv":
            csv = chisurf.core.fio.ascii.Csv()
            x, y = self[x_min:x_max]
            csv.save(
                data=np.vstack([x, y]),
                filename=filename
            )

    def load(
            self,
            filename: str,
            file_type: str = 'csv',
            skiprows: int = 0,
            **kwargs
    ) -> None:
        """Load curve data from disk.

        For ``file_type == 'csv'`` the first two rows are interpreted
        as ``x`` and ``y``.
        """
        super().load(
            filename=filename,
            file_type=file_type
        )
        if file_type == 'csv':
            csv = chisurf.core.fio.ascii.Csv()
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

    def to_dict(
            self,
            remove_protected: bool = True,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ) -> typing.Dict:
        """Serialize the curve to a dictionary.

        Depending on ``convert_values_to_elementary`` the arrays are stored
        as plain Python lists or NumPy arrays.
        """
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        if convert_values_to_elementary:
            d['x'] = self.x.tolist()
            d['y'] = self.y.tolist()
        else:
            if copy_values:
                d['x'] = np.copy(self.x)
                d['y'] = np.copy(self.y)
            else:
                d['x'] = self.x
                d['y'] = self.y
        return d

    def from_dict(self, v: dict):
        """Restore a curve from :meth:`to_dict` output."""
        super().from_dict(v)
        y = np.array(v['y'], dtype=np.float64)
        x = np.array(v['x'], dtype=np.float64)
        d = np.vstack([x, y])
        self.d = d

    def __init__(
            self,
            x: np.ndarray = None,
            y: np.ndarray = None,
            *args,
            **kwargs
    ):
        """Create a curve from x/y arrays.

        Parameters
        ----------
        x, y : array_like
            Arrays of identical length defining the abscissa and
            ordinate of the curve.
        """
        d = np.vstack([x, y])
        super().__init__(*args, d=d, **kwargs)

    def normalize(
            self,
            mode: str = "max",
            curve: chisurf.core.curve.Curve = None,
            inplace: bool = True
    ) -> float:
        """Calculates a scaling parameter for the Curve object and (optionally)
        scales the Curve object.

        :param mode: either 'max' to normalize the maximum to one, or 'sum' to
        normalize to sum to one
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

    def __add__(self, c: T) -> Curve:
        """Return the pointwise sum of two curves or curve and array."""
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__add__(c)
        )

    def __sub__(self, c: T) -> Curve:
        """Return the pointwise difference between two curves or curve and array."""
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__sub__(c)
        )

    def __mul__(self, c: T) -> Curve:
        """Return the pointwise product of two curves or curve and array."""
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__mul__(c)
        )

    def __truediv__(self, c: T) -> Curve:
        """Return the pointwise ratio of two curves or curve and array."""
        if isinstance(c, Curve):
            if not np.array_equal(self.x, c.x):
                raise ValueError("The x-axis differ")
            c = c.y
        return self.__class__(
            x=self.x,
            y=self.y.__truediv__(c)
        )

    def __lshift__(self, shift: float) -> Curve:
        """Return a copy with ``y`` shifted by ``shift`` samples."""
        return self.__class__(
            x=self.x,
            y=chisurf.core.math.signal.shift_array(self.y, shift),
            copy_array=False
        )
    def __len__(self) -> int:
        """Number of points in the curve (length of ``y``)."""
        return len(self.y)

    def __getitem__(self, key: typing.Union[slice, int, np.ndarray, str]) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Return a slice of curve as (x, y)."""
        return self.x[key], self.y[key]


class CurveGroup(object):
    """Light-weight container for a sequence of :class:`Curve` objects.

    The default implementation simply stores a list of curves and provides

    Examples
    --------
    >>> import numpy as np
    >>> from chisurf.core.curve import Curve, CurveGroup
    >>> c1 = Curve(x=np.array([0., 1.]), y=np.array([1., 2.]))
    >>> group = CurveGroup([c1])
    >>> len(group.get_data_curves())
    1
    """

    _curves: typing.List[chisurf.core.curve.Curve]

    def __init__(
            self,
            seq: typing.List[chisurf.core.curve.Curve] = None
    ):
        """Initialize a CurveGroup with an optional list of curves.

        Parameters
        ----------
        seq : list of Curve, optional
            Initial curve list.
        """
        if seq is None:
            seq = []
        self._curves = seq

    def clear_curves(self):
        """Remove all curves from the group."""
        self._curves.clear()

    def get_data_curves(
            self,
            *args,
            **kwargs
    ) -> typing.List[chisurf.core.curve.Curve]:
        """Return the list of curves stored in the group."""
        return self._curves

    @abc.abstractmethod
    def remove_curve(
            self,
            selected_index: typing.List[int] = None
    ):
        """Remove curves whose indices are listed in ``selected_index``."""
        if selected_index is None:
            selected_index = list()
        curve_list = list()
        for i, c in enumerate(self._curves):
            if i not in selected_index:
                curve_list.append(c)
        self._curves = curve_list

    @abc.abstractmethod
    def add_curve(
            self,
            *args,
            v: chisurf.core.curve.Curve = None,
            **kwargs
    ):
        """Append a new curve ``v`` to the group if it is not ``None``."""
        if v is not None:
            self._curves.append(v)

