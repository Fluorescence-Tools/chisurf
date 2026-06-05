from __future__ import annotations

from typing import Any, Dict, List, Optional

from chisurf.server.services import (
    ServiceResult,
    service_error,
    NOT_FOUND,
    INVALID_INPUT,
    OPERATION_FAILED,
    INVALID_STATE,
)
from chisurf.server.session import SessionState


def _is_global_fit_dataset(dataset: Any) -> bool:
    """Return ``True`` if *dataset* is a global-fit placeholder.

    Parameters
    ----------
    dataset : object
        Dataset instance.

    """
    try:
        name = str(getattr(dataset, "name", "") or "").strip().lower()
    except Exception:
        name = ""
    return name in {"global-fit", "global dataset", "global-fit dataset"}


def _find_fits_using_dataset(dataset: Any, state: SessionState) -> List[Any]:
    """Return all fits whose data references *dataset*.

    Parameters
    ----------
    dataset : object
        Dataset instance.
    state : SessionState
        Server-side session state.

    """
    dataset_id = id(dataset)
    dependent = []
    for fit in state.fits:
        grouped = getattr(fit, "grouped_fits", None)
        if isinstance(grouped, (list, tuple)):
            for gf in grouped:
                if id(getattr(gf, "data", None)) == dataset_id:
                    dependent.append(fit)
                    break
        else:
            if id(getattr(fit, "data", None)) == dataset_id:
                dependent.append(fit)
    return dependent


def _dataset_dto(dataset: Any, index: Optional[int], *, include_length: bool = False) -> Dict[str, Any]:
    """Build a serialisable summary dict for a dataset.

    Parameters
    ----------
    dataset : object
        Dataset instance.
    index : int or None
        Positional index.
    include_length : bool
        If ``True``, include the ``length`` field.

    """
    dto = {
        "index": index,
        "uid": str(getattr(dataset, "unique_identifier", "") or ""),
        "name": str(getattr(dataset, "name", "") or ""),
        "type": type(dataset).__name__,
        "experiment": str(getattr(getattr(dataset, "experiment", None), "name", "") or ""),
        "filename": str(getattr(dataset, "filename", "") or ""),
    }
    if include_length:
        dto["length"] = int(len(getattr(dataset, "y", []))) if hasattr(dataset, "y") else None
    return dto


def _resolve_dataset(
    state: SessionState,
    dataset_index: Optional[int] = None,
    dataset_uid: Optional[str] = None,
) -> tuple[Any, int]:
    """Look up a dataset by index or uid.

    Returns ``(dataset, index)`` or ``(None, -1)``.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    dataset_index : int, optional
        Positional index.
    dataset_uid : str, optional
        Unique identifier.

    """
    datasets = list(state.datasets)
    if dataset_uid is not None:
        for i, dataset in enumerate(datasets):
            if str(getattr(dataset, "unique_identifier", "")) == dataset_uid:
                return dataset, i
    if dataset_index is not None and 0 <= dataset_index < len(datasets):
        return datasets[dataset_index], dataset_index
    return None, -1


def _validated_dataset_indices(
    datasets: List[Any],
    dataset_indices: Any,
    *,
    empty_message: str,
) -> tuple[List[int], Optional[ServiceResult]]:
    """Validate and deduplicate a list of dataset indices.

    Returns ``(indices, None)`` on success or ``([], error)`` on failure.

    Parameters
    ----------
    datasets : list
        Current list of datasets.
    dataset_indices : iterable
        Raw indices to validate.
    empty_message : str
        Error message for an empty result.

    """
    if not hasattr(dataset_indices, "__iter__") or isinstance(dataset_indices, (str, bytes)):
        return [], service_error("dataset_indices must be an iterable of ints", error_code=INVALID_INPUT)
    try:
        indices = sorted(set(int(i) for i in dataset_indices))
    except Exception as e:
        return [], service_error(str(e), error_code=INVALID_INPUT, exception=e)
    if not indices:
        return [], service_error(empty_message, error_code=INVALID_INPUT)
    for i in indices:
        if i < 0 or i >= len(datasets):
            return [], service_error(f"dataset index {i} out of range", error_code=INVALID_INPUT)
    return indices, None


def list_datasets(state: SessionState) -> ServiceResult:
    """Return a summary of all datasets in the session.

    Parameters
    ----------
    state : SessionState
        Server-side session state.

    """
    return {
        "ok": True,
        "datasets": [_dataset_dto(d, idx) for idx, d in enumerate(state.datasets)],
    }


def get_dataset_info(
    state: SessionState,
    dataset_index: Optional[int] = None,
    dataset_uid: Optional[str] = None,
) -> ServiceResult:
    """Return detailed info for a single dataset.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    dataset_index : int, optional
        Positional index.
    dataset_uid : str, optional
        Unique identifier.

    """
    d, _ = _resolve_dataset(state, dataset_index, dataset_uid)
    if d is None:
        return service_error("dataset not found", error_code=NOT_FOUND)
    return {
        "ok": True,
        "dataset": _dataset_dto(d, dataset_index, include_length=True),
    }


def add_dataset(
    state: SessionState,
    reader: Any = None,
    reader_name: Optional[str] = None,
    filename: Optional[str] = None,
    name: Optional[str] = None,
    curve_data: Optional[Dict[str, Any]] = None,
    _from_controller: bool = False,
    event_bus: Any = None,
) -> ServiceResult:
    """Add a dataset to the session, either via a reader or raw curve data.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    reader : object, optional
        Data reader instance.
    reader_name : str, optional
        Name of the reader (ignored if *reader* is provided).
    filename : str, optional
        Path to the data file (used with *reader_name*).
    name : str, optional
        Display name for the dataset.
    curve_data : dict, optional
        Inline ``{"x": [...], "y": [...], "ex": [...], "ey": [...]}``.
    _from_controller : bool
        Internal flag to suppress duplicate event publishing.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    if reader is None and reader_name and filename:
        try:
            import numpy as np
            from chisurf.core.data import DataCurve
            import pathlib
            p = pathlib.Path(filename)
            x = np.array(curve_data.get("x", []), dtype=float) if curve_data and "x" in curve_data else None
            y = np.array(curve_data.get("y", []), dtype=float) if curve_data and "y" in curve_data else None
            ex = np.array(curve_data.get("ex", []), dtype=float) if curve_data and "ex" in curve_data else None
            ey = np.array(curve_data.get("ey", []), dtype=float) if curve_data and "ey" in curve_data else None
            ds = DataCurve(name=str(name or p.stem or "dataset"), x=x, y=y, ex=ex, ey=ey)
            ds.filename = str(p.resolve())
            state.add_dataset(ds)
            if event_bus is not None and not _from_controller:
                event_bus.publish("dataset.added", {"dataset_index": len(state.datasets) - 1, "name": str(name or p.name)})
            return {
                "ok": True,
                "uid": str(getattr(ds, "unique_identifier", "") or ""),
                "name": str(name or p.name),
                "dataset_index": len(state.datasets) - 1,
            }
        except Exception as e:
            return service_error(f"failed to create remote dataset: {e}", error_code=OPERATION_FAILED, exception=e)

    if reader is None:
        return service_error("no reader provided", error_code=INVALID_INPUT)

    dataset_name = str(name or getattr(reader, "name", "dataset"))
    try:
        dataset_group = reader.read(name=dataset_name)
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)

    if isinstance(dataset_group, list):
        for ds in dataset_group:
            state.add_dataset(ds)
    else:
        if hasattr(dataset_group, "__iter__") and not isinstance(dataset_group, (str, bytes)):
            for ds in dataset_group:
                state.add_dataset(ds)
        else:
            state.add_dataset(dataset_group)

    if event_bus is not None and not _from_controller:
        event_bus.publish("dataset.added", {"dataset_index": len(state.datasets) - 1, "name": dataset_name})

    return {
        "ok": True,
        "uid": str(getattr(dataset_group, "unique_identifier", "") or ""),
        "name": dataset_name,
        "dataset_index": len(state.datasets) - 1,
    }


def dataset_rename(
    state: SessionState,
    dataset_index: Optional[int] = None,
    dataset_uid: Optional[str] = None,
    new_name: str = "",
) -> ServiceResult:
    """Rename a dataset.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    dataset_index : int, optional
        Positional index.
    dataset_uid : str, optional
        Unique identifier.
    new_name : str
        New name.

    """
    d, _ = _resolve_dataset(state, dataset_index, dataset_uid)
    if d is None:
        return service_error("dataset not found", error_code=NOT_FOUND)
    try:
        d.name = str(new_name)
        return {"ok": True, "uid": dataset_uid, "new_name": new_name}
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def dataset_group(
    state: SessionState,
    dataset_indices: List[int],
    group_name: Optional[str] = None,
) -> ServiceResult:
    """Group selected datasets into an ``ExperimentDataGroup``.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    dataset_indices : list of int
        Indices of datasets to group.
    group_name : str, optional
        Name for the new group.

    """
    datasets = list(state.datasets)
    indices, error = _validated_dataset_indices(
        datasets,
        dataset_indices,
        empty_message="no dataset indices provided",
    )
    if error is not None:
        return error

    selected = []
    remaining = []
    for i, d in enumerate(datasets):
        if i in indices:
            selected.append(d)
        else:
            remaining.append(d)

    try:
        from chisurf.core.data import ExperimentDataGroup
        group = ExperimentDataGroup()
        group.name = str(group_name or "Data-Group")
        for ds in selected:
            group.append(ds)
        remaining.append(group)
        state.datasets[:] = remaining
        return {
            "ok": True,
            "group_uid": str(getattr(group, "unique_identifier", "") or ""),
            "group_name": group.name,
            "member_count": len(selected),
            "dataset_count": len(remaining),
        }
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)


def dataset_ungroup(
    state: SessionState,
    dataset_indices: List[int],
) -> ServiceResult:
    """Ungroup previously grouped datasets at *dataset_indices*.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    dataset_indices : list of int
        Indices of groups to ungroup.

    """
    datasets = list(state.datasets)
    indices, error = _validated_dataset_indices(
        datasets,
        dataset_indices,
        empty_message="no dataset indices provided",
    )
    if error is not None:
        return error

    new_datasets = []
    ungrouped = 0
    for i, d in enumerate(datasets):
        if i in indices:
            try:
                if hasattr(d, "__iter__") and not isinstance(d, (str, bytes)):
                    for member in d:
                        new_datasets.append(member)
                        ungrouped += 1
                    continue
            except Exception:
                pass
            new_datasets.append(d)
        else:
            new_datasets.append(d)

    state.datasets[:] = new_datasets
    return {
        "ok": True,
        "ungrouped_count": ungrouped,
        "dataset_count": len(new_datasets),
    }


def remove_datasets(
    state: SessionState,
    dataset_indices: Optional[List[int]] = None,
    dataset_uids: Optional[List[str]] = None,
    _from_controller: bool = False,
    event_bus: Any = None,
) -> ServiceResult:
    """Remove datasets by index or uid, preventing removal if fits depend on them.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    dataset_indices : list of int, optional
        Indices to remove.
    dataset_uids : list of str, optional
        UIDs to remove.
    _from_controller : bool
        Internal flag to suppress duplicate event publishing.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    indices: List[int] = []

    if dataset_uids:
        all_ds = list(state.datasets)
        for i, d in enumerate(all_ds):
            if str(getattr(d, "unique_identifier", "")) in dataset_uids:
                indices.append(i)

    if dataset_indices:
        indices.extend(int(i) for i in dataset_indices)

    if not indices:
        return service_error("no datasets specified for removal", error_code=INVALID_INPUT)

    datasets = list(state.datasets)
    indices, error = _validated_dataset_indices(
        datasets,
        indices,
        empty_message="no datasets specified for removal",
    )
    if error is not None:
        return error

    for i in indices:
        deps = _find_fits_using_dataset(datasets[i], state)
        if deps:
            ds_name = getattr(datasets[i], "name", f"index {i}")
            fit_names = ", ".join(str(getattr(f, "name", "?")) for f in deps)
            return service_error(
                f"dataset '{ds_name}' is used by fit(s): {fit_names}. Remove fits first.",
                error_code=INVALID_STATE,
            )

    to_remove = [datasets[i] for i in indices if not _is_global_fit_dataset(datasets[i])]
    kept = [d for d in datasets if d not in to_remove]
    state.datasets[:] = kept

    if event_bus is not None and not _from_controller:
        event_bus.publish("dataset.removed", {"removed_count": len(to_remove), "remaining_count": len(kept)})

    return {
        "ok": True,
        "removed_count": len(to_remove),
        "remaining_count": len(kept),
    }


def clear_datasets(state: SessionState, event_bus: Any = None) -> ServiceResult:
    """Remove all datasets from the session.

    Parameters
    ----------
    state : SessionState
        Server-side session state.
    event_bus : object, optional
        Event bus for broadcasting.

    """
    count = len(state.datasets)
    state.datasets.clear()
    if event_bus is not None:
        event_bus.publish("dataset.cleared", {"cleared_count": count})
    return {"ok": True, "cleared_count": count}


def _sanitize_float_list(values: Any) -> Optional[List[Optional[float]]]:
    """Convert an iterable to a JSON-safe list of floats, replacing NaN/Inf with None."""
    import numpy as np
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float)
        result: List[Optional[float]] = []
        for v in arr.flat:
            if np.isnan(v) or np.isinf(v):
                result.append(None)
            else:
                result.append(float(v))
        return result
    except Exception:
        return None


def get_dataset_curve_data(
    state: SessionState,
    dataset_index: Optional[int] = None,
    dataset_uid: Optional[str] = None,
) -> ServiceResult:
    """Return x/y/ex/ey arrays for a dataset."""
    d, _ = _resolve_dataset(state, dataset_index, dataset_uid)
    if d is None:
        return service_error("dataset not found", error_code=NOT_FOUND)
    try:
        result: Dict[str, Any] = {"ok": True}
        result["x"] = _sanitize_float_list(getattr(d, "x", None))
        result["y"] = _sanitize_float_list(getattr(d, "y", None))
        result["ex"] = _sanitize_float_list(getattr(d, "ex", None))
        result["ey"] = _sanitize_float_list(getattr(d, "ey", None))
        return result
    except Exception as e:
        return service_error(str(e), error_code=OPERATION_FAILED, exception=e)
