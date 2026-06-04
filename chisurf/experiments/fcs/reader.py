from __future__ import annotations

import chisurf.fio.ascii
import chisurf.fio.fluorescence.fcs
import chisurf.fluorescence.fcs
import chisurf.data
import chisurf.fio.fluorescence
from chisurf.experiments.core.reader import ExperimentReader


class FCS(ExperimentReader):
    """Reader for fluorescence correlation spectroscopy (FCS) data.

    This reader wraps :func:`chisurf.fio.fluorescence.fcs.read_fcs` and
    returns an :class:`chisurf.data.ExperimentDataCurveGroup` for a
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

    >>> from chisurf.experiments.fcs import FCS
    >>> r = FCS(name='demo', experiment_reader='CSV', skiprows=2)
    >>> (r.name, r.experiment_reader, r.skiprows)
    ('demo', 'csv', 2)
    """

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
        # to :func:`chisurf.fio.fluorescence.fcs.read_fcs` so that FCS
        # correlation-amplitude weights can be recomputed on import.
        self.weight_mode = None
        self.weight_kwargs = dict()

    def read(
            self,
            filename: str = None,
            verbose: bool = None,
            **kwargs
    ) -> chisurf.data.ExperimentDataCurveGroup:
        """Read an FCS data file.

        Parameters
        ----------
        filename : str, optional
            Path to the FCS file.
        verbose : bool, optional
            If *True*, enable verbose output during reading.

        Returns
        -------
        chisurf.data.ExperimentDataCurveGroup
            Group containing the loaded FCS curves.
        """
        r = chisurf.fio.fluorescence.fcs.read_fcs(
            filename=filename,
            data_reader=self,
            skiprows=self.skiprows,
            use_header=self.use_header,
            reader_name=self.experiment_reader,
            experiment=getattr(self, 'experiment', None),
            weight_mode=getattr(self, 'weight_mode', None),
            weight_kwargs=getattr(self, 'weight_kwargs', None) or {},
        )
        r.current_dataset.data_reader = self
        r.data_reader = self
        return r

    def autofitrange(
            self,
            data: chisurf.base.Data,
            **kwargs
    ) -> typing.Tuple[int, int]:
        """Return the full data range as the default fit interval.

        Parameters
        ----------
        data : chisurf.base.Data
            The experimental data object.

        Returns
        -------
        tuple of int
            ``(0, len(y))`` for curve data, ``(0, 0)`` otherwise.
        """
        if isinstance(data, (chisurf.data.DataCurve, chisurf.data.DataCurveGroup)):
            return 0, len(data.y)
        return 0, 0
