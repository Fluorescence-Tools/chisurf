from __future__ import annotations

import pathlib

import chisurf.core.fio.ascii
import chisurf.core.fio.fluorescence.fcs
import chisurf.core.fluorescence.fcs
import chisurf.core.data
import chisurf.core.fio.fluorescence
from chisurf.core.experiments.core.reader import ExperimentReader

_VIEW_JSON = pathlib.Path(__file__).parent / "fcs.view.json"


class FCS(ExperimentReader):
    """Reader for fluorescence correlation spectroscopy (FCS) data.

    This reader wraps :func:`chisurf.core.fio.fluorescence.fcs.read_fcs` and
    returns an :class:`chisurf.core.data.ExperimentDataCurveGroup` for a
    single FCS file.

    Parameters
    ----------
    name : str, optional
        Human-readable name of the reader.
    use_header : bool, optional
        Whether to parse a header row when present.
    experiment_reader : str, optional
        Name of the low-level FCS reader implementation.
    skiprows : int, optional
        Number of header rows to skip before numerical data.

    Examples
    --------
    Only exercise attribute handling (no file I/O):

    >>> from chisurf.core.experiments.fcs import FCS
    >>> r = FCS(name='demo', experiment_reader='CSV', skiprows=2)
    >>> (r.name, r.experiment_reader, r.skiprows)
    ('demo', 'csv', 2)
    """

    operation_type = "fcs_correlation_load"
    artifact_kind_source = "raw_data"
    artifact_kind_derived = "fcs_correlation"
    derived_data_format = "json"
    derived_mime_type = "application/json"

    name: str = "FCS-CSV"
    skiprows: int = 0
    use_header: bool = False

    def __init__(
            self,
            name: str = 'FCS',
            use_header: bool = False,
            experiment_reader='kristine',
            skiprows: int = 0,
            *args,
            **kwargs
    ):
        """Initialize an FCS reader.

        Parameters
        ----------
        name : str
            Human-readable reader name.
        use_header : bool
            Whether to parse a header row if present.
        experiment_reader : str
            Name of the low-level FCS reader implementation.
        skiprows : int
            Number of header rows to skip.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.skiprows = skiprows
        self.use_header = use_header
        self.experiment_reader = experiment_reader.lower()
        # Optional weighting configuration; if set, these are passed through
        # to :func:`chisurf.core.fio.fluorescence.fcs.read_fcs` so that FCS
        # correlation-amplitude weights can be recomputed on import.
        self.weight_mode = None
        self.weight_kwargs = dict()
        # Column assignments for the generic csv reader (experiment_reader='csv').
        self.col_x: int = 0
        self.col_y: int = 1
        self.col_ex: int = 2
        self.col_ey: int = 3
        self.error_x_on: bool = False
        self.error_y_on: bool = True

    def read(
            self,
            filename: str = None,
            verbose: bool = None,
            **kwargs
    ) -> chisurf.core.data.ExperimentDataCurveGroup:
        """Read an FCS data file.

        Parameters
        ----------
        filename : str, optional
            Path to the FCS file.
        verbose : bool, optional
            If *True*, enable verbose output during reading.

        Returns
        -------
        chisurf.core.data.ExperimentDataCurveGroup
            Group containing the loaded FCS curves.
        """
        csv_kwargs = {}
        if self.experiment_reader == 'csv':
            csv_kwargs = dict(
                col_x=self.col_x,
                col_y=self.col_y,
                col_ex=self.col_ex,
                col_ey=self.col_ey,
                error_x_on=self.error_x_on,
                y_on=self.error_y_on,
            )
        r = chisurf.core.fio.fluorescence.fcs.read_fcs(
            filename=filename,
            data_reader=self,
            skiprows=self.skiprows,
            use_header=self.use_header,
            reader_name=self.experiment_reader,
            experiment=getattr(self, 'experiment', None),
            weight_mode=getattr(self, 'weight_mode', None),
            weight_kwargs=getattr(self, 'weight_kwargs', None) or {},
            **csv_kwargs,
        )
        r.current_dataset.data_reader = self
        r.data_reader = self
        return r

    def autofitrange(
            self,
            data: chisurf.core.base.Data,
            **kwargs
    ) -> typing.Tuple[int, int]:
        """Return the full data range as the default fit interval.

        Parameters
        ----------
        data : chisurf.core.base.Data
            The experimental data object.

        Returns
        -------
        tuple of int
            ``(0, len(y) - 1)`` for curve data, ``(0, 0)`` otherwise.
        """
        if isinstance(data, (chisurf.core.data.DataCurve, chisurf.core.data.DataCurveGroup)):
            return 0, max(0, len(data.y) - 1)
        return 0, 0

    @property
    def weight_mode_key(self) -> str:
        """String key for the noise/weighting model; empty string means 'from file'."""
        return self.weight_mode if self.weight_mode is not None else ""

    @weight_mode_key.setter
    def weight_mode_key(self, value: str) -> None:
        self.weight_mode = None if value == "" else value

    @property
    def reading_routine(self) -> str:
        """Alias for experiment_reader used by the format UI."""
        return self.experiment_reader

    @reading_routine.setter
    def reading_routine(self, value: str) -> None:
        self.experiment_reader = value.lower() if value else "kristine"

    def view_spec(self):
        """Return the declarative editor spec for FCS reader settings."""
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_VIEW_JSON)
