"""Msgpack codec for typed MFDB result payloads."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import msgpack
import numpy as np

from mfdb.payload_models import (
    PAYLOAD_MODELS,
    BurstTable,
    GenericCurve,
    Spectrum,
)

ENVELOPE_VERSION = 1
REGISTRY: dict[str, type] = {model.KIND: model for model in PAYLOAD_MODELS}
MIGRATIONS: dict[tuple[str, int, int], Any] = {}


class PayloadSchemaError(ValueError):
    """Raised when a payload does not match its declared schema."""


def encode_payload(kind: str, obj: Any, *, meta: dict | None = None) -> tuple[bytes, str]:
    """Validate and encode a typed result payload.

    Parameters
    ----------
    kind : str
        Payload kind.
    obj : Any
        Dataclass instance or mapping compatible with the kind schema.
    meta : dict, optional
        Small JSON-safe metadata stored in the msgpack envelope.

    Returns
    -------
    tuple
        ``(msgpack_bytes, "msgpack")``.
    """
    model = _model_for_kind(kind)
    payload = _coerce_to_model(model, obj)
    _validate_payload(payload)
    envelope = {
        "v": ENVELOPE_VERSION,
        "kind": model.KIND,
        "schema": model.SCHEMA_VERSION,
        "units": _units_for_model(model),
        "meta": meta or {},
        "body": _encode_body(payload),
    }
    return msgpack.packb(envelope, use_bin_type=True, strict_types=True), "msgpack"


def decode_payload(blob: bytes) -> Any:
    """Decode a self-describing msgpack result payload.

    Parameters
    ----------
    blob : bytes
        Msgpack envelope bytes.

    Returns
    -------
    Any
        Typed payload dataclass.
    """
    try:
        envelope = msgpack.unpackb(blob, raw=False)
    except Exception as exc:
        raise PayloadSchemaError(f"invalid msgpack payload: {exc}") from exc
    if not isinstance(envelope, dict):
        raise PayloadSchemaError("payload envelope must be a map")
    if envelope.get("v") != ENVELOPE_VERSION:
        raise PayloadSchemaError(f"unsupported envelope version {envelope.get('v')!r}")
    kind = envelope.get("kind")
    if not isinstance(kind, str):
        raise PayloadSchemaError("payload envelope kind must be a string")
    model = _model_for_kind(kind)
    envelope = _migrate_envelope(envelope, model)
    body = envelope.get("body")
    if not isinstance(body, dict):
        raise PayloadSchemaError("payload envelope body must be a map")
    decoded = {name: _decode_value(body[name]) for name in body}
    try:
        payload = model(**decoded)
    except TypeError as exc:
        raise PayloadSchemaError(f"{kind} body does not match schema: {exc}") from exc
    _validate_payload(payload)
    return payload


def get_payload_schema(kind: str) -> dict:
    """Return the formal schema for a payload kind.

    Parameters
    ----------
    kind : str
        Payload kind.

    Returns
    -------
    dict
        Schema description derived from the model ``FIELDS`` spec.
    """
    model = _model_for_kind(kind)
    return {
        "envelope_version": ENVELOPE_VERSION,
        "kind": model.KIND,
        "schema_version": model.SCHEMA_VERSION,
        "fields": {
            name: _schema_field(model, name, spec)
            for name, spec in model.FIELDS.items()
        },
        "units": _units_for_model(model),
    }


def analysis_data_row_to_payload(row: Any) -> GenericCurve:
    """Adapt a legacy ``analysis_data`` row to a typed curve payload.

    Parameters
    ----------
    row : Any
        SQLite row or mapping.

    Returns
    -------
    GenericCurve
        Typed curve payload.
    """
    return GenericCurve.from_analysis_data_row(row)


def spectrum_row_to_payload(row: Any) -> Spectrum:
    """Adapt a legacy ``spectra`` row to a typed spectrum payload.

    Parameters
    ----------
    row : Any
        SQLite row or mapping.

    Returns
    -------
    Spectrum
        Typed spectrum payload.
    """
    return Spectrum.from_spectrum_row(row)


def _model_for_kind(kind: str) -> type:
    try:
        return REGISTRY[kind]
    except KeyError as exc:
        raise PayloadSchemaError(f"unknown payload kind {kind!r}") from exc


def _schema_field(model: type, name: str, spec: tuple[str, bool, str | None]) -> dict[str, Any]:
    field = {"type": spec[0], "required": spec[1], "unit": spec[2]}
    flrcif_items = getattr(model, "FLRCIF_ITEMS", {})
    flrcif_item_id = flrcif_items.get(name)
    if flrcif_item_id:
        field["flrcif_item_id"] = flrcif_item_id
    return field


def _coerce_to_model(model: type, obj: Any) -> Any:
    if isinstance(obj, model):
        return obj
    if is_dataclass(obj):
        payload_kind = getattr(obj, "KIND", None)
        if payload_kind != model.KIND:
            raise PayloadSchemaError(f"payload kind {payload_kind!r} does not match requested {model.KIND!r}")
        return obj
    if isinstance(obj, dict):
        try:
            return model(**{key: _coerce_plain_value(value) for key, value in obj.items()})
        except TypeError as exc:
            raise PayloadSchemaError(f"{model.KIND} mapping does not match schema: {exc}") from exc
    raise PayloadSchemaError(f"cannot encode {type(obj)!r} as {model.KIND!r}")


def _coerce_plain_value(value: Any) -> Any:
    if isinstance(value, list) and value and all(isinstance(item, (int, float, bool)) for item in value):
        return np.asarray(value)
    return value


def _encode_body(payload: Any) -> dict[str, Any]:
    body = {}
    for name in payload.FIELDS:
        value = getattr(payload, name)
        if value is not None:
            body[name] = _encode_value(value)
    return body


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        return _decode_array(value)
    if isinstance(value, dict):
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "__ndarray__": True,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def _decode_array(encoded: dict[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(encoded["dtype"])
        shape = tuple(int(size) for size in encoded["shape"])
        data = encoded["data"]
    except KeyError as exc:
        raise PayloadSchemaError(f"invalid ndarray encoding, missing {exc.args[0]!r}") from exc
    if not isinstance(data, bytes):
        raise PayloadSchemaError("invalid ndarray encoding: data must be bytes")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _validate_payload(payload: Any) -> None:
    if not is_dataclass(payload):
        raise PayloadSchemaError("payload must be a dataclass instance")
    model = type(payload)
    expected_fields = set(model.FIELDS)
    dataclass_fields = {field.name for field in fields(payload)}
    unexpected = dataclass_fields - expected_fields
    if unexpected:
        raise PayloadSchemaError(f"{model.KIND} dataclass has unexpected fields {sorted(unexpected)!r}")
    for name, spec in model.FIELDS.items():
        value = getattr(payload, name)
        type_spec, required, _unit = spec
        if value is None:
            if required:
                raise PayloadSchemaError(f"{model.KIND}.{name} is required")
            continue
        _validate_value(model.KIND, name, value, type_spec)
    _validate_cross_field_shapes(payload)


def _validate_value(kind: str, name: str, value: Any, type_spec: str) -> None:
    errors = []
    for option in type_spec.split("|"):
        try:
            _validate_value_option(kind, name, value, option)
            return
        except PayloadSchemaError as exc:
            errors.append(str(exc))
    raise PayloadSchemaError("; ".join(errors))


def _validate_value_option(kind: str, name: str, value: Any, type_spec: str) -> None:
    if type_spec == "f8[]":
        _validate_array(kind, name, value, np.dtype("float64"))
    elif type_spec == "i8[]":
        _validate_array(kind, name, value, np.dtype("int64"))
    elif type_spec == "u1[]":
        _validate_array(kind, name, value, np.dtype("uint8"))
    elif type_spec == "u2[]":
        _validate_array(kind, name, value, np.dtype("uint16"))
    elif type_spec == "u4[]":
        _validate_array(kind, name, value, np.dtype("uint32"))
    elif type_spec == "u8[]":
        _validate_array(kind, name, value, np.dtype("uint64"))
    elif type_spec == "bool[]":
        _validate_array(kind, name, value, np.dtype("bool"))
    elif type_spec == "int[]":
        if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            raise PayloadSchemaError(f"{kind}.{name} must be a list of int")
    elif type_spec == "f8[]list":
        if not isinstance(value, list) or not value:
            raise PayloadSchemaError(f"{kind}.{name} must be a non-empty list of float64 arrays")
        for index, item in enumerate(value):
            _validate_array(kind, f"{name}[{index}]", item, np.dtype("float64"))
    elif type_spec == "array_dict":
        if not isinstance(value, dict):
            raise PayloadSchemaError(f"{kind}.{name} must be a dictionary of arrays")
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadSchemaError(f"{kind}.{name} keys must be strings")
            if not isinstance(item, np.ndarray):
                raise PayloadSchemaError(f"{kind}.{name}[{key!r}] must be an ndarray")
            if item.ndim != 1:
                raise PayloadSchemaError(f"{kind}.{name}[{key!r}] must be one-dimensional")
            if item.dtype.hasobject:
                raise PayloadSchemaError(f"{kind}.{name}[{key!r}] has unsupported object dtype")
    elif type_spec == "str[]":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise PayloadSchemaError(f"{kind}.{name} must be a list of str")
    elif type_spec == "str":
        if not isinstance(value, str):
            raise PayloadSchemaError(f"{kind}.{name} must be a str")
    elif type_spec == "bool":
        if not isinstance(value, bool):
            raise PayloadSchemaError(f"{kind}.{name} must be a bool")
    elif type_spec == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise PayloadSchemaError(f"{kind}.{name} must be an int")
    elif type_spec == "f8":
        if not isinstance(value, (int, float, np.floating)) or isinstance(value, bool):
            raise PayloadSchemaError(f"{kind}.{name} must be a float")
    elif type_spec == "json":
        _validate_json_safe(kind, name, value)
    else:
        raise PayloadSchemaError(f"unsupported field type spec {type_spec!r}")


def _validate_array(kind: str, name: str, value: Any, dtype: np.dtype) -> None:
    if not isinstance(value, np.ndarray):
        raise PayloadSchemaError(f"{kind}.{name} must be an ndarray")
    if value.dtype != dtype:
        raise PayloadSchemaError(f"{kind}.{name} must have dtype {dtype}, got {value.dtype}")
    if value.ndim < 1:
        raise PayloadSchemaError(f"{kind}.{name} must have at least one dimension")


def _validate_cross_field_shapes(payload: Any) -> None:
    payload_kind = getattr(payload, "KIND", "")
    if payload_kind == GenericCurve.KIND:
        for name in ("y", "ex", "ey", "mask"):
            value = getattr(payload, name)
            if value is not None and value.shape != payload.x.shape:
                raise PayloadSchemaError(f"{payload.KIND}.{name} must match x shape")
    if payload_kind == "spectra" and payload.intensity.shape != payload.wavelength.shape:
        raise PayloadSchemaError("spectra.intensity must match wavelength shape")
    if payload_kind == "tcspc_decay":
        for name in ("counts", "irf"):
            value = getattr(payload, name)
            if value is not None and value.shape != payload.time.shape:
                raise PayloadSchemaError(f"tcspc_decay.{name} must match time shape")
    if payload_kind == "anisotropy_curve":
        for name in ("vv", "vh", "l1", "l2"):
            value = getattr(payload, name)
            if value is not None and value.shape != payload.time.shape:
                raise PayloadSchemaError(f"anisotropy_curve.{name} must match time shape")
    if payload_kind == "fcs_correlation":
        if payload.lag.ndim != 1:
            raise PayloadSchemaError("fcs_correlation.lag must be one-dimensional")
        if payload.correlation.ndim not in (1, 2):
            raise PayloadSchemaError("fcs_correlation.correlation must be one- or two-dimensional")
        if payload.correlation.shape[-1] != payload.lag.shape[0]:
            raise PayloadSchemaError("fcs_correlation.correlation last axis must match lag length")
        if payload.curve_names is not None:
            expected_names = 1 if payload.correlation.ndim == 1 else payload.correlation.shape[0]
            if len(payload.curve_names) != expected_names:
                raise PayloadSchemaError("fcs_correlation.curve_names length must match curve count")
        for name in ("error", "weights"):
            value = getattr(payload, name)
            if value is not None and value.shape != payload.correlation.shape:
                raise PayloadSchemaError(f"fcs_correlation.{name} must match correlation shape")
    if isinstance(payload, BurstTable):
        if len(payload.columns) != len(payload.dtypes):
            raise PayloadSchemaError(f"{payload.KIND}.columns and dtypes must have the same length")
        if set(payload.columns) != set(payload.data):
            raise PayloadSchemaError(f"{payload.KIND}.data keys must match columns")
        lengths = {len(payload.data[column]) for column in payload.columns}
        if len(lengths) > 1:
            raise PayloadSchemaError(f"{payload.KIND}.data columns must have the same length")
        for column, dtype in zip(payload.columns, payload.dtypes):
            actual = str(payload.data[column].dtype)
            if actual != dtype:
                raise PayloadSchemaError(f"{payload.KIND}.{column} dtype {actual!r} != declared {dtype!r}")
    if getattr(payload, "KIND", "") == "burst_selection":
        if payload.burst_ids is None and payload.start_indices is None and payload.mask is None:
            raise PayloadSchemaError("burst_selection requires burst_ids, start_indices/stop_indices, or mask")
        if (payload.start_indices is None) != (payload.stop_indices is None):
            raise PayloadSchemaError("burst_selection start_indices and stop_indices must be provided together")
        if payload.start_indices is not None and payload.start_indices.shape != payload.stop_indices.shape:
            raise PayloadSchemaError("burst_selection start_indices and stop_indices must have the same shape")
        if payload.labels is not None:
            count = _selection_count(payload)
            if count is not None and len(payload.labels) != count:
                raise PayloadSchemaError("burst_selection labels length must match selected burst count")
    if getattr(payload, "KIND", "") == "tttr_photon_stream":
        n_events = payload.macro_times.shape[0]
        for name in ("micro_times", "routing_channels", "event_types"):
            value = getattr(payload, name)
            if value is not None and value.shape != (n_events,):
                raise PayloadSchemaError(f"tttr_photon_stream.{name} must match macro_times shape")
    if getattr(payload, "KIND", "") == "pda_histogram":
        if isinstance(payload.edges, np.ndarray):
            if payload.counts.ndim != 1 or len(payload.edges) != payload.counts.shape[0] + 1:
                raise PayloadSchemaError("pda_histogram.edges must be one longer than 1D counts")
        else:
            if len(payload.edges) != payload.counts.ndim:
                raise PayloadSchemaError("pda_histogram edge axis count must match counts ndim")
            for axis, edges in enumerate(payload.edges):
                if len(edges) != payload.counts.shape[axis] + 1:
                    raise PayloadSchemaError(
                        f"pda_histogram.edges[{axis}] must be one longer than counts axis"
                    )


def _selection_count(payload: Any) -> int | None:
    if payload.burst_ids is not None:
        return int(payload.burst_ids.shape[0])
    if payload.start_indices is not None:
        return int(payload.start_indices.shape[0])
    if payload.mask is not None:
        return int(np.count_nonzero(payload.mask))
    return None


def _validate_json_safe(kind: str, name: str, value: Any) -> None:
    """Validate JSON-like values used for reader headers and tags.

    Parameters
    ----------
    kind : str
        Payload kind for diagnostics.
    name : str
        Field name for diagnostics.
    value : Any
        Candidate JSON-like value.

    Returns
    -------
    None
        Raises when a value cannot be represented safely.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_safe(kind, f"{name}[{index}]", item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadSchemaError(f"{kind}.{name} JSON map keys must be strings")
            _validate_json_safe(kind, f"{name}.{key}", item)
        return
    raise PayloadSchemaError(f"{kind}.{name} must be JSON-safe")


def _units_for_model(model: type) -> dict[str, str]:
    units: dict[str, str] = {}
    for name, spec in model.FIELDS.items():
        unit = spec[2]
        if unit is not None:
            units[name] = unit
    return units


def _migrate_envelope(envelope: dict[str, Any], model: type) -> dict[str, Any]:
    schema_version = envelope.get("schema")
    if not isinstance(schema_version, int):
        raise PayloadSchemaError("payload envelope schema must be an int")
    while schema_version < model.SCHEMA_VERSION:
        key = (model.KIND, schema_version, schema_version + 1)
        upgrader = MIGRATIONS.get(key)
        if upgrader is None:
            raise PayloadSchemaError(f"no migration for {model.KIND} schema {schema_version}")
        envelope = upgrader(envelope)
        schema_version = schema_version + 1
        envelope["schema"] = schema_version
    if schema_version != model.SCHEMA_VERSION:
        raise PayloadSchemaError(
            f"unsupported {model.KIND} schema {schema_version}, expected {model.SCHEMA_VERSION}"
        )
    return envelope
