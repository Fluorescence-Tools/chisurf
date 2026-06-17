"""Language-agnostic serialization for derived experimental data.

Derived data (histograms, correlation curves, fit results) is stored in the
object store as canonical JSON with base64-encoded NumPy arrays. This ensures
interoperability across programming languages while preserving array dtypes
and shapes.

Canonical JSON structure::

    {
        "schema_version": "1.0",
        "data_type": "tcspc_decay",
        "created_by": "TCSPCTTTRReader",
        "source_object_uuids": ["uuid-of-source-tttr"],
        "curves": [
            {
                "name": "sample_ch0",
                "x": {"dtype": "float64", "shape": [N], "data": "<base64>"},
                "y": {"dtype": "float64", "shape": [N], "data": "<base64>"},
                "ey": {"dtype": "float64", "shape": [N], "data": "<base64>"},
                "mask": {"dtype": "bool", "shape": [N], "data": "<base64>"},
                "meta_data": {}
            }
        ],
        "reader_settings": {}
    }
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

import chisurf.core.data


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    """Encode a NumPy array as a JSON-serializable dict with base64 data.

    Parameters
    ----------
    arr : np.ndarray
        The array to encode.

    Returns
    -------
    dict
        Dictionary with keys ``dtype``, ``shape``, and ``data``.
    """
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(obj: dict[str, Any]) -> np.ndarray:
    """Decode a base64-encoded array dict back to a NumPy array.

    Parameters
    ----------
    obj : dict
        Dictionary with keys ``dtype``, ``shape``, and ``data``.

    Returns
    -------
    np.ndarray
        The decoded array.
    """
    return np.frombuffer(
        base64.b64decode(obj["data"]),
        dtype=obj["dtype"],
    ).reshape(obj["shape"]).copy()


def _curve_to_dict(curve: chisurf.core.data.DataCurve) -> dict[str, Any]:
    """Serialize a single DataCurve to a JSON-serializable dict."""
    result: dict[str, Any] = {"name": getattr(curve, "name", "")}

    for attr in ("x", "y", "ex", "ey"):
        arr = getattr(curve, attr, None)
        if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
            result[attr] = encode_array(np.asarray(arr))

    mask = getattr(curve, "mask", None)
    if mask is not None and hasattr(mask, "__len__") and len(mask) > 0:
        result["mask"] = encode_array(np.asarray(mask, dtype=bool))

    meta = getattr(curve, "meta_data", None)
    if meta is not None:
        result["meta_data"] = _sanitize_metadata(meta)
    else:
        result["meta_data"] = {}

    return result


def _sanitize_metadata(obj: Any) -> Any:
    """Recursively convert metadata to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, (list, tuple)):
        return [_sanitize_metadata(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_metadata(v) for k, v in obj.items()}
    return str(obj)


def data_to_json(
    data: chisurf.core.data.DataCurve | chisurf.core.data.DataGroup,
    data_type: str = "experimental_data",
    created_by: str = "",
    source_object_uuids: list[str] | None = None,
    reader_settings: dict[str, Any] | None = None,
) -> bytes:
    """Serialize experimental data to canonical JSON bytes.

    Parameters
    ----------
    data : DataCurve or DataGroup
        The data to serialize.
    data_type : str
        Type identifier (e.g. ``'tcspc_decay'``, ``'pda_histogram'``).
    created_by : str
        Name of the reader or processor that created the data.
    source_object_uuids : list of str, optional
        UUIDs of source objects in the object store.
    reader_settings : dict, optional
        Reader configuration settings.

    Returns
    -------
    bytes
        UTF-8 encoded JSON bytes.
    """
    if isinstance(data, chisurf.core.data.DataCurve):
        curves = [_curve_to_dict(data)]
    elif isinstance(data, (chisurf.core.data.DataGroup, chisurf.core.data.ExperimentDataGroup)):
        curves = []
        for item in data:
            if isinstance(item, chisurf.core.data.DataCurve):
                curves.append(_curve_to_dict(item))
            elif isinstance(item, chisurf.core.data.DataCurveGroup):
                for sub in item:
                    if isinstance(sub, chisurf.core.data.DataCurve):
                        curves.append(_curve_to_dict(sub))
    else:
        curves = []

    doc = {
        "schema_version": "1.0",
        "data_type": data_type,
        "created_by": created_by,
        "source_object_uuids": source_object_uuids or [],
        "curves": curves,
        "reader_settings": _sanitize_metadata(reader_settings) if reader_settings else {},
    }
    return json.dumps(doc, sort_keys=True, ensure_ascii=True).encode("utf-8")


def _dict_to_curve(d: dict[str, Any]) -> chisurf.core.data.DataCurve:
    """Deserialize a single curve dict back to a DataCurve."""
    kwargs: dict[str, Any] = {"name": d.get("name", "")}

    for attr in ("x", "y", "ex", "ey"):
        if attr in d:
            kwargs[attr] = decode_array(d[attr])

    if "mask" in d:
        kwargs["mask"] = decode_array(d["mask"])

    meta = d.get("meta_data", {})
    for k, v in meta.items():
        if isinstance(v, dict) and "dtype" in v and "data" in v:
            meta[k] = decode_array(v)
    kwargs["meta_data"] = meta
    kwargs["load_filename_on_init"] = False

    return chisurf.core.data.DataCurve(**kwargs)


def data_from_json(
    blob: bytes,
) -> chisurf.core.data.DataCurve | chisurf.core.data.ExperimentDataGroup:
    """Deserialize JSON bytes back to experimental data objects.

    Parameters
    ----------
    blob : bytes
        UTF-8 encoded JSON bytes produced by :func:`data_to_json`.

    Returns
    -------
    DataCurve or ExperimentDataGroup
        The deserialized data. Returns a single DataCurve if only one curve
        is present, otherwise an ExperimentDataGroup.
    """
    doc = json.loads(blob.decode("utf-8"))
    curves = [_dict_to_curve(c) for c in doc.get("curves", [])]

    if len(curves) == 1:
        return curves[0]
    group = chisurf.core.data.ExperimentDataGroup(curves)
    return group
