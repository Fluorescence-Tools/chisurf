"""Typed MFDB result payload models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

FieldSpec = tuple[str, bool, str | None]


@dataclass
class FcsCorrelation:
    """FCS or FCCS correlation curve payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "fcs_correlation"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "lag": ("f8[]", True, "s"),
        "correlation": ("f8[]", True, "1"),
        "error": ("f8[]", False, "1"),
        "curve_names": ("str[]", False, None),
        "weights": ("f8[]", False, "1"),
    }

    lag: np.ndarray
    correlation: np.ndarray
    error: np.ndarray | None = None
    curve_names: list[str] | None = None
    weights: np.ndarray | None = None


@dataclass
class Spectrum:
    """Absorption, emission, or excitation spectrum payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "spectra"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "wavelength": ("f8[]", True, "nm"),
        "intensity": ("f8[]", True, "1"),
        "spectrum_type": ("str", True, None),
        "normalized": ("bool", False, None),
    }

    wavelength: np.ndarray
    intensity: np.ndarray
    spectrum_type: str
    normalized: bool | None = None

    @classmethod
    def from_spectrum_row(cls, row: Any) -> Spectrum:
        """Build a payload from a legacy ``spectra`` row.

        Parameters
        ----------
        row : Any
            Row or mapping with ``wavelengths`` and ``intensity_values`` blobs.

        Returns
        -------
        Spectrum
            Typed spectrum payload.
        """
        data = dict(row)
        return cls(
            wavelength=np.frombuffer(data["wavelengths"], dtype=np.float64).copy(),
            intensity=np.frombuffer(data["intensity_values"], dtype=np.float64).copy(),
            spectrum_type=str(data["spectrum_type"]),
            normalized=str(data.get("intensity_unit") or "").lower() in {"1", "normalized"},
        )


@dataclass
class TcspcDecay:
    """TCSPC fluorescence decay payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "tcspc_decay"
    FLRCIF_ITEMS: ClassVar[dict[str, str]] = {
        "adc_resolution_ns": "_flr_chisurf_parameter.dtTAC[ns]",
        "micro_time_resolution_ns": "_flr_chisurf_parameter.dtMT[ns]",
    }
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "time": ("f8[]", True, "ns"),
        "counts": ("i8[]|f8[]", True, "1"),
        "irf": ("f8[]", False, "1"),
        "channel": ("str", False, None),
        "adc_resolution_ns": ("f8", False, "ns"),
        "micro_time_resolution_ns": ("f8", False, "ns"),
    }

    time: np.ndarray
    counts: np.ndarray
    irf: np.ndarray | None = None
    channel: str | None = None
    adc_resolution_ns: float | None = None
    micro_time_resolution_ns: float | None = None


@dataclass
class AnisotropyCurve:
    """Time-resolved anisotropy or polarized decay pair payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "anisotropy_curve"
    FLRCIF_ITEMS: ClassVar[dict[str, str]] = {
        "l1": "_flr_chisurf_parameter.l1",
        "l2": "_flr_chisurf_parameter.l2",
        "g_factor": "_flr_chisurf_parameter.g",
    }
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "time": ("f8[]", True, "ns"),
        "vv": ("f8[]", True, "1"),
        "vh": ("f8[]", True, "1"),
        "l1": ("f8[]", False, "1"),
        "l2": ("f8[]", False, "1"),
        "g_factor": ("f8", False, "1"),
    }

    time: np.ndarray
    vv: np.ndarray
    vh: np.ndarray
    l1: np.ndarray | None = None
    l2: np.ndarray | None = None
    g_factor: float | None = None


@dataclass
class PdaHistogram:
    """Photon-distribution-analysis histogram payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "pda_histogram"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "edges": ("f8[]|f8[]list", True, None),
        "counts": ("f8[]", True, "1"),
        "axis_names": ("str[]", False, None),
    }

    edges: np.ndarray | list[np.ndarray]
    counts: np.ndarray
    axis_names: list[str] | None = None


@dataclass
class BurstTable:
    """Columnar per-burst table payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "burst_table"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "columns": ("str[]", True, None),
        "dtypes": ("str[]", True, None),
        "data": ("array_dict", True, None),
    }

    columns: list[str]
    dtypes: list[str]
    data: dict[str, np.ndarray]

    @classmethod
    def from_dataframe(cls, df: Any) -> BurstTable:
        """Create a columnar table from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Source table.

        Returns
        -------
        BurstTable
            Dtype-preserving columnar payload.
        """
        columns = [str(column) for column in df.columns]
        data = {column: _coerce_table_array(df[column].to_numpy(copy=True), column) for column in columns}
        dtypes = [str(data[column].dtype) for column in columns]
        return cls(columns=columns, dtypes=dtypes, data=data)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> BurstTable:
        """Create a columnar table from a mapping of columns to values.

        Parameters
        ----------
        mapping : dict
            Column mapping. Scalars become one-row columns.

        Returns
        -------
        BurstTable
            Columnar payload.
        """
        columns = [str(column) for column in mapping.keys()]
        data: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for column in columns:
            value = mapping[column]
            array = _coerce_table_array(np.asarray(value), column)
            if array.ndim == 0:
                array = array.reshape(1)
            if array.ndim != 1:
                raise ValueError(f"generic table column {column!r} must be one-dimensional")
            data[column] = array
            lengths.add(len(array))
        if len(lengths) > 1:
            raise ValueError("generic table columns must have the same length")
        return cls(columns=columns, dtypes=[str(data[column].dtype) for column in columns], data=data)

    def to_dataframe(self) -> Any:
        """Convert this payload to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Reconstructed table with original column order.
        """
        import pandas as pd

        return pd.DataFrame({column: self.data[column] for column in self.columns})


def _coerce_table_array(array: np.ndarray, column: str) -> np.ndarray:
    """Normalize table columns to msgpack-safe ndarray dtypes.

    Parameters
    ----------
    array : np.ndarray
        Candidate column array.
    column : str
        Column name used for diagnostics.

    Returns
    -------
    np.ndarray
        Array with a supported non-object dtype.
    """
    if array.dtype.hasobject:
        flat = array.reshape(-1)
        # Accept string columns that contain missing values (None/NaN/NA) — common
        # in burst summary tables (e.g. a "First File" column). Missing entries are
        # normalized to empty strings so the column serializes as a string array.
        if all(isinstance(item, str) or _is_table_missing(item) for item in flat):
            normalized = [
                "" if _is_table_missing(item) else str(item) for item in flat
            ]
            return np.asarray(normalized, dtype=np.str_).reshape(array.shape)
        raise ValueError(f"generic table column {column!r} has unsupported object dtype")
    return array


def _is_table_missing(value: Any) -> bool:
    """Return whether a table cell is a missing value (None / NaN / pandas NA)."""
    if value is None or type(value).__name__ == "NAType":
        return True
    try:
        return bool(isinstance(value, float) and np.isnan(value))
    except (TypeError, ValueError):
        return False


@dataclass
class BurstSelection:
    """Selected burst IDs, ranges, or masks derived from a burst table or TTTR stream."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "burst_selection"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "source_artifact_id": ("str", False, None),
        "burst_ids": ("i8[]", False, None),
        "start_indices": ("u8[]", False, None),
        "stop_indices": ("u8[]", False, None),
        "mask": ("bool[]", False, None),
        "criteria": ("json", False, None),
        "labels": ("str[]", False, None),
    }

    source_artifact_id: str | None = None
    burst_ids: np.ndarray | None = None
    start_indices: np.ndarray | None = None
    stop_indices: np.ndarray | None = None
    mask: np.ndarray | None = None
    criteria: dict[str, Any] | None = None
    labels: list[str] | None = None


@dataclass
class TttrReference:
    """Structured reference to a raw TTTR artifact."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "tttr_reference"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "source_artifact_id": ("str", True, None),
        "vendor_format": ("str", True, None),
        "n_records": ("int", False, None),
        "routing_channels": ("int[]", False, None),
        "macro_time_resolution_s": ("f8", False, "s"),
        "micro_time_resolution_s": ("f8", False, "s"),
        "header_json": ("json", False, None),
        "tags": ("json", False, None),
    }

    source_artifact_id: str
    vendor_format: str
    n_records: int | None = None
    routing_channels: list[int] | None = None
    macro_time_resolution_s: float | None = None
    micro_time_resolution_s: float | None = None
    header_json: dict[str, Any] | str | None = None
    tags: dict[str, Any] | None = None


@dataclass
class TttrPhotonStream:
    """Open, normalized TTTR photon stream payload with reader metadata.

    This payload is for optional portable exports. The canonical archival path
    keeps the raw vendor file and stores a ``TttrReference`` to it.
    """

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "tttr_photon_stream"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "macro_times": ("u8[]", True, "ticks"),
        "micro_times": ("u4[]", True, "channels"),
        "routing_channels": ("u2[]|u4[]", True, None),
        "event_types": ("u1[]|u2[]|u4[]", False, None),
        "macro_time_resolution_s": ("f8", True, "s"),
        "micro_time_resolution_s": ("f8", True, "s"),
        "source_artifact_id": ("str", False, None),
        "vendor_format": ("str", False, None),
        "record_type": ("str", False, None),
        "container_type": ("str", False, None),
        "header_json": ("json", False, None),
        "tags": ("json", False, None),
    }

    macro_times: np.ndarray
    micro_times: np.ndarray
    routing_channels: np.ndarray
    macro_time_resolution_s: float
    micro_time_resolution_s: float
    event_types: np.ndarray | None = None
    source_artifact_id: str | None = None
    vendor_format: str | None = None
    record_type: str | None = None
    container_type: str | None = None
    header_json: dict[str, Any] | str | None = None
    tags: dict[str, Any] | None = None


@dataclass
class GenericCurve:
    """Generic 1D curve payload compatible with ``DataCurve``."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "generic_curve"
    FIELDS: ClassVar[dict[str, FieldSpec]] = {
        "x": ("f8[]", True, None),
        "y": ("f8[]", True, None),
        "ex": ("f8[]", False, None),
        "ey": ("f8[]", False, None),
        "mask": ("bool[]", False, None),
    }

    x: np.ndarray
    y: np.ndarray
    ex: np.ndarray | None = None
    ey: np.ndarray | None = None
    mask: np.ndarray | None = None

    @classmethod
    def from_data_curve(cls, dc: Any) -> GenericCurve:
        """Create a payload from ``chisurf.core.data.DataCurve``.

        Parameters
        ----------
        dc : DataCurve
            Source curve.

        Returns
        -------
        GenericCurve
            Typed curve payload.
        """
        return cls(
            x=np.asarray(dc.x, dtype=np.float64).copy(),
            y=np.asarray(dc.y, dtype=np.float64).copy(),
            ex=np.asarray(dc.ex, dtype=np.float64).copy() if dc.ex is not None else None,
            ey=np.asarray(dc.ey, dtype=np.float64).copy() if dc.ey is not None else None,
            mask=np.asarray(dc.mask, dtype=bool).copy() if dc.mask is not None else None,
        )

    @classmethod
    def from_analysis_data_row(cls, row: Any) -> GenericCurve:
        """Build a payload from a legacy ``analysis_data`` row.

        Parameters
        ----------
        row : Any
            Row or mapping with ``x_values`` and ``y_values`` blobs.

        Returns
        -------
        GenericCurve
            Typed curve payload.
        """
        data = dict(row)
        return cls(
            x=np.frombuffer(data["x_values"], dtype=np.float64).copy(),
            y=np.frombuffer(data["y_values"], dtype=np.float64).copy(),
        )

    def to_data_curve(self) -> Any:
        """Convert this payload to ``chisurf.core.data.DataCurve``.

        Returns
        -------
        DataCurve
            Reconstructed ChiSurf curve.
        """
        from chisurf.core.data import DataCurve

        return DataCurve(
            x=self.x.copy(),
            y=self.y.copy(),
            ex=None if self.ex is None else self.ex.copy(),
            ey=None if self.ey is None else self.ey.copy(),
            mask=None if self.mask is None else self.mask.copy(),
        )


@dataclass
class GenericTable(BurstTable):
    """Generic columnar table payload."""

    SCHEMA_VERSION: ClassVar[int] = 1
    KIND: ClassVar[str] = "generic_table"


PAYLOAD_MODELS: tuple[type, ...] = (
    FcsCorrelation,
    Spectrum,
    TcspcDecay,
    AnisotropyCurve,
    PdaHistogram,
    BurstTable,
    BurstSelection,
    TttrReference,
    TttrPhotonStream,
    GenericCurve,
    GenericTable,
)
