from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from chisurf.api.context import PluginContext
from chisurf.server.session import SessionState


def _extract_curve_data(dataset: Any) -> Optional[Dict[str, List[float]]]:
    """Extract serializable x/y/ex/ey arrays from a DataCurve or container.

    Returns ``None`` if the dataset does not have curve-like data.
    """
    try:
        x = getattr(dataset, "x", None)
        y = getattr(dataset, "y", None)
        if x is None or y is None:
            return None
        result: Dict[str, List[float]] = {
            "x": np.asarray(x, dtype=float).tolist(),
            "y": np.asarray(y, dtype=float).tolist(),
        }
        ex = getattr(dataset, "ex", None)
        if ex is not None:
            result["ex"] = np.asarray(ex, dtype=float).tolist()
        ey = getattr(dataset, "ey", None)
        if ey is not None:
            result["ey"] = np.asarray(ey, dtype=float).tolist()
        return result
    except Exception:
        return None


def _local_datasets() -> list[Any]:
    import chisurf
    return list(getattr(chisurf, "imported_datasets", []) or [])


def _local_fits() -> list[Any]:
    import chisurf
    return list(getattr(chisurf, "fits", []) or [])


def _resolve_indexed(items: list[Any], index: Optional[int] = None, uid: Optional[str] = None) -> tuple[Any, int]:
    if uid is not None:
        for i, item in enumerate(items):
            if str(getattr(item, "unique_identifier", "")) == uid:
                return item, i
    if index is not None and 0 <= index < len(items):
        return items[index], index
    return None, -1


def _local_dataset(dataset_index: Optional[int] = None, dataset_uid: Optional[str] = None) -> tuple[Any, int]:
    return _resolve_indexed(_local_datasets(), dataset_index, dataset_uid)


def _local_fit(fit_index: Optional[int] = None, fit_uid: Optional[str] = None) -> tuple[Any, int]:
    return _resolve_indexed(_local_fits(), fit_index, fit_uid)


def _local_parameter(
    parameter_name: str,
    fit_index: int = 0,
    fit_uid: Optional[str] = None,
    require_parameters: bool = False,
) -> tuple[Any, Any, Optional[Dict[str, Any]]]:
    fit, _ = _local_fit(fit_index, fit_uid)
    if fit is None:
        return None, None, {"ok": False, "error": "fit not found"}
    try:
        parameters = getattr(fit.model, "parameters_all_dict", {}) or {}
    except Exception:
        if require_parameters:
            return fit, None, {"ok": False, "error": "cannot access model parameters"}
        parameters = {}
    parameter = parameters.get(parameter_name)
    if parameter is None:
        return fit, None, {"ok": False, "error": f"parameter '{parameter_name}' not found"}
    return fit, parameter, None


class ChiSurfAPI:
    """Single stable API facade for GUI, macros, plugins, and QtConsole.

    Routes to local in-process objects in ``local`` mode, server RPC in
    ``server`` mode, and a mix in ``hybrid`` mode.

    Modes
    -----
    local
        Use current in-process ``chisurf.fits`` / ``chisurf.imported_datasets``.
    hybrid
        Local reads allowed; server commands preferred for migrated paths.
    server
        Pure client/server operation through a ``ChisurfClient``.
    """

    def __init__(
        self,
        client: Any = None,
        mode: str = "hybrid",
    ):
        self.client = client
        self.mode = mode

    # ── datasets ─────────────────────────────────────────────────

    def list_datasets(self) -> List[Dict[str, Any]]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__list()
        import chisurf
        result: List[Dict[str, Any]] = []
        for idx, d in enumerate(getattr(chisurf, "imported_datasets", []) or []):
            result.append({
                "index": idx,
                "uid": str(getattr(d, "unique_identifier", "") or ""),
                "name": str(getattr(d, "name", "") or ""),
                "type": type(d).__name__,
                "experiment": str(getattr(getattr(d, "experiment", None), "name", "") or ""),
                "filename": str(getattr(d, "filename", "") or ""),
            })
        return result

    def get_dataset_info(self, dataset_index: Optional[int] = None, dataset_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__get(dataset_index=dataset_index, dataset_uid=dataset_uid)
        d, idx = _local_dataset(dataset_index, dataset_uid)
        if d is None:
            return {"ok": False, "error": "dataset not found"}
        return {
            "ok": True,
            "dataset": {
                "index": idx,
                "uid": str(getattr(d, "unique_identifier", "") or ""),
                "name": str(getattr(d, "name", "") or ""),
                "type": type(d).__name__,
                "experiment": str(getattr(getattr(d, "experiment", None), "name", "") or ""),
                "filename": str(getattr(d, "filename", "") or ""),
                "length": int(len(getattr(d, "y", []))) if hasattr(d, "y") else None,
            },
        }

    def get_dataset_curve_data(self, dataset_index: Optional[int] = None, dataset_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__curve_data(dataset_index=dataset_index, dataset_uid=dataset_uid)
        d, _ = _local_dataset(dataset_index, dataset_uid)
        if d is None:
            return {"ok": False, "error": "dataset not found"}
        try:
            result: Dict[str, Any] = {"ok": True}
            x = getattr(d, "x", None)
            if x is not None:
                result["x"] = np.asarray(x, dtype=float).tolist()
            y = getattr(d, "y", None)
            if y is not None:
                result["y"] = np.asarray(y, dtype=float).tolist()
            ex = getattr(d, "ex", None)
            if ex is not None:
                result["ex"] = np.asarray(ex, dtype=float).tolist()
            ey = getattr(d, "ey", None)
            if ey is not None:
                result["ey"] = np.asarray(ey, dtype=float).tolist()
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def load_dataset(
        self,
        experiment_reader: Any = None,
        dataset: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            params: Dict[str, Any] = dict(kwargs)
            if experiment_reader is not None:
                rname = getattr(type(experiment_reader), "__name__", None)
                if rname:
                    params["reader_name"] = rname
                fname = getattr(experiment_reader, "filename", None) or kwargs.get("filename")
                if fname:
                    params["filename"] = str(fname)
                name = getattr(experiment_reader, "name", None) or kwargs.get("name")
                if name:
                    params["name"] = str(name)
                # Extract data arrays from the dataset if available; else
                # try to read them locally so the server has real x/y data.
                if dataset is not None:
                    curve_data = _extract_curve_data(dataset)
                    if curve_data:
                        params["curve_data"] = curve_data
                else:
                    try:
                        local_data = experiment_reader.get_data(**kwargs)
                        curve_data = _extract_curve_data(local_data)
                        if curve_data:
                            params["curve_data"] = curve_data
                    except Exception:
                        pass
            return self.client.call("add_dataset", params)
        from chisurf.macros import core_data
        core_data.add_dataset(
            experiment_reader=experiment_reader,
            dataset=dataset,
            _from_controller=True,
            **kwargs,
        )
        return {"ok": True}

    def remove_datasets(
        self,
        dataset_indices: Optional[List[int]] = None,
        dataset_uids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__remove(dataset_indices=dataset_indices, dataset_uids=dataset_uids)
        import chisurf
        from chisurf.macros import core_data
        indices = list(dataset_indices or [])
        if dataset_uids:
            for i, d in enumerate(getattr(chisurf, "imported_datasets", []) or []):
                if str(getattr(d, "unique_identifier", "")) in dataset_uids:
                    indices.append(i)
        if indices:
            core_data.remove_datasets(dataset_indices=list(set(indices)), _from_controller=True)
        return {"ok": True}

    def clear_datasets(self) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__clear()
        import chisurf
        ds_list = getattr(chisurf, "imported_datasets", None)
        if ds_list is not None:
            ds_list.clear()
        return {"ok": True}

    def group_datasets(
        self,
        dataset_indices: List[int],
        group_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__group(
                dataset_indices=dataset_indices,
                group_name=group_name,
            )
        from chisurf.macros import core_data
        core_data.group_datasets(
            dataset_indices=dataset_indices,
            _from_controller=True,
        )
        return {"ok": True}

    def ungroup_datasets(
        self,
        dataset_indices: List[int],
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.dataset__ungroup(dataset_indices=dataset_indices)
        from chisurf.macros import core_data
        core_data.ungroup_datasets(
            dataset_indices=dataset_indices,
            _from_controller=True,
        )
        return {"ok": True}

    # ── fits ──────────────────────────────────────────────────────

    def list_fits(self) -> List[Dict[str, Any]]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__list()
        import chisurf
        result: List[Dict[str, Any]] = []
        fits = list(getattr(chisurf, "fits", []) or [])
        for idx, f in enumerate(fits):
            chi2 = None
            try:
                chi2 = float(getattr(f, "chi2", float("nan")))
            except Exception:
                pass
            data_name = ""
            try:
                data_name = str(getattr(getattr(f, "data", None), "name", "") or "")
            except Exception:
                pass
            param_count = 0
            try:
                param_count = len(getattr(getattr(f, "model", None), "parameters_all_dict", {}) or {})
            except Exception:
                pass
            result.append({
                "index": idx,
                "uid": str(getattr(f, "unique_identifier", "") or ""),
                "name": str(getattr(f, "name", "") or ""),
                "type": type(f).__name__,
                "chi2": chi2,
                "dataset_uid": str(getattr(getattr(f, "data", None), "unique_identifier", "") or ""),
                "dataset_name": data_name,
                "model_name": str(getattr(getattr(f, "model", None), "name", "") or ""),
                "parameter_count": param_count,
                "data": {
                    "name": str(getattr(getattr(f, "data", None), "name", "") or ""),
                    "uid": str(getattr(getattr(f, "data", None), "unique_identifier", "") or ""),
                    "filename": str(getattr(getattr(f, "data", None), "filename", "") or ""),
                    "experiment": str(getattr(getattr(f, "data", None), "experiment", "") or getattr(getattr(getattr(f, "data", None), "experiment", None), "name", "") or ""),
                } if hasattr(f, "data") and f.data is not None else {},
                "model": {
                    "name": str(getattr(getattr(f, "model", None), "name", "") or ""),
                    "n_points": _safe_n_points(f),
                    "n_free": _safe_n_free(f),
                    "chi2r": _safe_chi2r(f),
                    "parameters_all": _collect_param_list(f, fit_uid=str(getattr(f, "unique_identifier", "") or "")),
                } if hasattr(f, "model") and f.model is not None else {},
            })
        return result

    def get_fit_info(self, fit_index: Optional[int] = None, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__get(fit_index=fit_index, fit_uid=fit_uid)
        fit, idx = _local_fit(fit_index, fit_uid)
        if fit is None:
            return {"ok": False, "error": "fit not found"}
        params = {}
        try:
            pdict = getattr(fit.model, "parameters_all_dict", {}) if hasattr(fit, "model") else {}
            for name, p in pdict.items():
                params[name] = {
                    "value": getattr(p, "value", None),
                    "fixed": bool(getattr(p, "fixed", False)),
                    "bounds": getattr(p, "bounds", None),
                    "bounds_on": bool(getattr(p, "bounds_on", False)),
                    "linked_to": str(getattr(getattr(p, "link", None), "name", "") or ""),
                    "error_estimate": getattr(p, "error_estimate", None),
                }
        except Exception:
            pass
        return {
            "ok": True,
            "fit": {
                "index": idx,
                "uid": str(getattr(fit, "unique_identifier", "") or ""),
                "name": str(getattr(fit, "name", "") or ""),
                "type": type(fit).__name__,
                "chi2": _safe_chi2(fit),
                "chi2r": _safe_chi2r(fit),
                "n_points": _safe_n_points(fit),
                "n_free": _safe_n_free(fit),
                "dataset_uid": str(getattr(getattr(fit, "data", None), "unique_identifier", "") or ""),
                "dataset_name": str(getattr(getattr(fit, "data", None), "name", "") or ""),
                "model_name": str(getattr(getattr(fit, "model", None), "name", "") or ""),
                "parameter_count": len(params),
                "parameters": params,
                "data": {
                    "name": str(getattr(getattr(fit, "data", None), "name", "") or ""),
                    "uid": str(getattr(getattr(fit, "data", None), "unique_identifier", "") or ""),
                    "filename": str(getattr(getattr(fit, "data", None), "filename", "") or ""),
                    "experiment": str(getattr(getattr(fit, "data", None), "experiment", "") or getattr(getattr(getattr(fit, "data", None), "experiment", None), "name", "") or ""),
                } if hasattr(fit, "data") and fit.data is not None else {},
                "model": {
                    "name": str(getattr(getattr(fit, "model", None), "name", "") or ""),
                    "n_points": _safe_n_points(fit),
                    "n_free": _safe_n_free(fit),
                    "chi2r": _safe_chi2r(fit),
                    "parameters_all": _collect_param_list(fit, fit_uid=str(getattr(fit, "unique_identifier", "") or "")),
                } if hasattr(fit, "model") and fit.model is not None else {},
            },
        }

    def run_fit(self, fit_index: Optional[int] = None, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__run(fit_index=fit_index, fit_uid=fit_uid)
        fit, idx = _local_fit(fit_index, fit_uid)
        if fit is None:
            return {"ok": False, "error": "fit not found"}
        try:
            chi2_before = _safe_chi2(fit)
            fit.run()
            chi2_after = _safe_chi2(fit)
            return {
                "ok": True,
                "fit_index": idx,
                "fit_uid": str(getattr(fit, "unique_identifier", "") or ""),
                "chi2_before": chi2_before,
                "chi2_after": chi2_after,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def add_fit(
        self,
        dataset_indices: List[int],
        model_name: Optional[str] = None,
        model_kw: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            call_kw: Dict[str, Any] = {"dataset_indices": list(dataset_indices or [])}
            if model_name is not None:
                call_kw["model_name"] = str(model_name)
            if isinstance(model_kw, dict):
                call_kw["model_kw"] = model_kw
            return self.client.fit__create(**call_kw)
        from chisurf.macros import core_fit
        kwargs: Dict[str, Any] = {"dataset_indices": list(dataset_indices or [])}
        if isinstance(model_kw, dict):
            kwargs["model_kw"] = model_kw
        if model_name is not None:
            kwargs["model_name"] = str(model_name)
        return core_fit.add_fit(**kwargs)

    def remove_fits(
        self,
        fit_indices: Optional[List[int]] = None,
        fit_uids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__remove(fit_indices=fit_indices, fit_uids=fit_uids)
        import chisurf
        fits = list(getattr(chisurf, "fits", []) or [])
        to_remove: set = set()
        if fit_uids:
            for i, f in enumerate(fits):
                if str(getattr(f, "unique_identifier", "")) in fit_uids:
                    to_remove.add(i)
        if fit_indices:
            to_remove.update(int(i) for i in fit_indices if 0 <= int(i) < len(fits))
        if not to_remove:
            return {"ok": False, "error": "no fits specified"}
        kept = [f for i, f in enumerate(fits) if i not in to_remove]
        chisurf.fits[:] = kept
        return {"ok": True, "removed_count": len(to_remove)}

    def clear_fits(self) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__clear()
        import chisurf
        chisurf.fits.clear()
        return {"ok": True}

    def fit_create(self, dataset_index: int = 0, model_name: Optional[str] = None, fit_name: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__create(dataset_index=dataset_index, model_name=model_name, fit_name=fit_name)
        return {"ok": False, "error": "fit.create requires server mode for server-side creation"}

    def fit_update(self, fit_index: Optional[int] = None, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.fit__update(fit_index=fit_index, fit_uid=fit_uid)
        fit, _ = _local_fit(fit_index, fit_uid)
        if fit is None:
            return {"ok": False, "error": "fit not found"}
        try:
            if hasattr(fit, "update"):
                fit.update()
                return {"ok": True}
            return {"ok": False, "error": "fit has no update method"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── parameters ────────────────────────────────────────────────

    def get_parameter(self, parameter_name: str, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.parameter__get(parameter_name=parameter_name, fit_index=fit_index, fit_uid=fit_uid)
        _, p, error = _local_parameter(parameter_name, fit_index, fit_uid)
        if error is not None:
            return error
        return {
            "ok": True,
            "parameter": {
                "name": parameter_name,
                "value": getattr(p, "value", None),
                "fixed": bool(getattr(p, "fixed", False)),
                "bounds": getattr(p, "bounds", None),
                "bounds_on": bool(getattr(p, "bounds_on", False)),
                "error_estimate": getattr(p, "error_estimate", None),
                "linked_to": str(getattr(getattr(p, "link", None), "name", "") or ""),
            },
        }

    def set_parameter_value(self, parameter_name: str, value: float, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.parameter__set_value(parameter_name=parameter_name, value=value, fit_index=fit_index, fit_uid=fit_uid)
        fit, p, error = _local_parameter(parameter_name, fit_index, fit_uid, require_parameters=True)
        if error is not None:
            return error
        try:
            p.value = float(value)
            if hasattr(fit.model, "update_model"):
                fit.model.update_model()
            if hasattr(fit.model, "finalize"):
                fit.model.finalize()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_parameter_fixed(self, parameter_name: str, fixed: bool, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.parameter__set_fixed(parameter_name=parameter_name, fixed=fixed, fit_index=fit_index, fit_uid=fit_uid)
        fit, p, error = _local_parameter(parameter_name, fit_index, fit_uid, require_parameters=True)
        if error is not None:
            return error
        try:
            p.fixed = bool(fixed)
            if hasattr(fit.model, "finalize"):
                fit.model.finalize()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_parameter_bounds(self, parameter_name: str, bounds: tuple, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.parameter__set_bounds(parameter_name=parameter_name, lower=bounds[0], upper=bounds[1], fit_index=fit_index, fit_uid=fit_uid)
        fit, p, error = _local_parameter(parameter_name, fit_index, fit_uid, require_parameters=True)
        if error is not None:
            return error
        try:
            p.bounds = tuple(float(v) for v in bounds)
            if hasattr(fit.model, "finalize"):
                fit.model.finalize()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── model ─────────────────────────────────────────────────────

    def model_finalize(self, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.model__finalize(fit_index=fit_index, fit_uid=fit_uid)
        fit, _ = _local_fit(fit_index, fit_uid)
        if fit is None:
            return {"ok": False, "error": "fit not found"}
        try:
            model = getattr(fit, "model", None)
            if model is None:
                return {"ok": False, "error": "fit has no model"}
            if hasattr(model, "finalize"):
                model.finalize()
                return {"ok": True}
            return {"ok": False, "error": "model has no finalize method"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def model_set_parse_function(self, parse_function: str, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.model__set_parse_function(parse_function=parse_function, fit_index=fit_index, fit_uid=fit_uid)
        fit, _ = _local_fit(fit_index, fit_uid)
        if fit is None:
            return {"ok": False, "error": "fit not found"}
        try:
            model = getattr(fit, "model", None)
            if model is None:
                return {"ok": False, "error": "fit has no model"}
            setattr(model, "parse_function", parse_function)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_parameter_bounds_on(self, parameter_name: str, bounds_on: bool, fit_index: int = 0, fit_uid: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.parameter__set_bounds_on(parameter_name=parameter_name, bounds_on=bounds_on, fit_index=fit_index, fit_uid=fit_uid)
        _, p, error = _local_parameter(parameter_name, fit_index, fit_uid, require_parameters=True)
        if error is not None:
            return error
        try:
            p.bounds_on = bool(bounds_on)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── projects ──────────────────────────────────────────────────

    def get_project_info(self) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.project__info()
        import chisurf
        return {
            "ok": True,
            "project_path": None,
            "fit_count": len(getattr(chisurf, "fits", []) or []),
            "dataset_count": len(getattr(chisurf, "imported_datasets", []) or []),
        }

    def save_project(self, target_path: str, project_name: Optional[str] = None) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.project__save(target_path=target_path, project_name=project_name)
        return {"ok": False, "error": "project.save requires server mode"}

    def load_project(self, project_path: str) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.project__load(project_path=project_path)
        return {"ok": False, "error": "project.load requires server mode"}

    # ── convenience ───────────────────────────────────────────────

    def ping(self) -> Dict[str, Any]:
        if self.client is not None:
            return self.client.meta__ping()
        return {"ok": True, "status": "alive-local"}

    def session_describe(self) -> Dict[str, Any]:
        if self.client is not None:
            return self.client.session__describe()
        import chisurf
        return {
            "ok": True,
            "datasets": self.list_datasets(),
            "fits": self.list_fits(),
            "dataset_count": len(getattr(chisurf, "imported_datasets", []) or []),
            "fit_count": len(getattr(chisurf, "fits", []) or []),
        }

    def session_snapshot(self) -> Dict[str, Any]:
        if self.client is not None:
            return self.client.session__snapshot()
        import chisurf
        return {
            "ok": True,
            "snapshot": {
                "dataset_count": len(getattr(chisurf, "imported_datasets", []) or []),
                "fit_count": len(getattr(chisurf, "fits", []) or []),
            },
        }

    def session_restore(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        if self.client is not None:
            return self.client.session__restore(project_path=project_path)
        import chisurf
        if project_path:
            return {"ok": False, "error": "session restore requires server mode for project loading"}
        getattr(chisurf, "fits", []).clear()
        getattr(chisurf, "imported_datasets", []).clear()
        return {"ok": True, "message": "session cleared locally"}

    @property
    def fit_count(self) -> int:
        fits = self.list_fits()
        return len(fits)

    @property
    def dataset_count(self) -> int:
        datasets = self.list_datasets()
        return len(datasets)

    def subscribe(self, topic: str = "", callback: Optional[Callable] = None) -> Any:
        if self.client is not None:
            return self.client.subscribe(topic, callback)
        return None

    def drain(self) -> None:
        """Process all buffered ZMQ subscriber events from the main thread."""
        if self.client is not None:
            self.client.drain()

    def list_methods(self) -> List[str]:
        if self.client is not None:
            return self.client.meta__methods()
        return []

    def install_proxies(self, force: bool = False) -> None:
        """Replace ``chisurf.fits`` and ``chisurf.imported_datasets`` with proxy objects.

        Only activates in ``server`` mode (or when *force* is true).
        After this call all access to those globals is routed through the
        ZMQ client.
        """
        if not force and self.mode != "server":
            return
        from chisurf.proxy import install_proxies
        install_proxies(self.client)

    # ---- graph ----
    def build_fit_graph(
        self,
        fit_indices: Optional[List[int]] = None,
        fit_uids: Optional[List[str]] = None,
        include_fixed: bool = True,
        connect_fits: bool = False,
    ) -> Dict[str, Any]:
        if self.mode == "server" and self.client is not None:
            return self.client.graph__build(
                fit_indices=fit_indices,
                fit_uids=fit_uids,
                include_fixed=include_fixed,
                connect_fits=connect_fits,
            )
        import chisurf
        fits = list(getattr(chisurf, "fits", []) or [])
        selected = []
        if fit_uids:
            uid_set = set(fit_uids)
            for i, f in enumerate(fits):
                uid = str(getattr(f, "unique_identifier", "") or "")
                if uid in uid_set:
                    selected.append((i, f))
        elif fit_indices:
            for i in fit_indices:
                if 0 <= i < len(fits):
                    selected.append((i, fits[i]))
        else:
            selected = list(enumerate(fits))

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_idx = 0
        fit_node_ids: Dict[int, int] = {}

        for fit_idx, fit in selected:
            try:
                data_filename = str(getattr(getattr(fit, "data", None), "filename", ""))
            except Exception:
                data_filename = ""
            try:
                model_module = getattr(fit.model.__class__, "__module__", "")
                model_class_name = getattr(fit.model.__class__, "__name__", "")
                model_full = f"{model_module}.{model_class_name}"
            except Exception:
                model_full = ""
            node = {
                "node_idx": node_idx,
                "node_type": "fit",
                "name": str(getattr(fit, "name", "fit")),
                "fit_idx": fit_idx,
                "data_filename": data_filename,
                "model": model_full,
            }
            node_id = node_idx
            nodes.append(node)
            fit_node_ids[fit_idx] = node_id
            node_idx += 1
            try:
                parameters = list(getattr(fit.model, "parameters_all", []) or [])
            except Exception:
                parameters = []
            for param in parameters:
                try:
                    fixed = bool(getattr(param, "fixed", False))
                except Exception:
                    fixed = False
                if fixed and not include_fixed:
                    continue
                try:
                    is_linked = bool(getattr(param, "is_linked", False))
                except Exception:
                    is_linked = False
                try:
                    link_name = str(getattr(getattr(param, "link", None), "name", ""))
                except Exception:
                    link_name = ""
                param_node = {
                    "node_idx": node_idx,
                    "node_type": "parameter",
                    "name": str(getattr(param, "name", "param")),
                    "fit_idx": fit_idx,
                    "value": _safe_float(getattr(param, "value", None)),
                    "fixed": fixed,
                    "is_linked": is_linked,
                    "link_name": link_name,
                }
                param_node_id = node_idx
                nodes.append(param_node)
                edges.append({"source": param_node_id, "target": node_id})
                node_idx += 1

        for n in nodes:
            if n["node_type"] != "parameter" or not n.get("is_linked"):
                continue
            link_name = n.get("link_name", "")
            if not link_name:
                continue
            for m in nodes:
                if m["node_type"] != "parameter":
                    continue
                if m["name"] == link_name and m["fit_idx"] != n["fit_idx"]:
                    edges.append({"source": n["node_idx"], "target": m["node_idx"]})

        if connect_fits:
            fit_nodes = [n for n in nodes if n["node_type"] == "fit"]
            for i, a in enumerate(fit_nodes):
                for b in fit_nodes[i + 1:]:
                    edges.append({"source": a["node_idx"], "target": b["node_idx"]})

        return {"ok": True, "graph": {"nodes": nodes, "edges": edges}}


from chisurf.server.services._stats import _safe_chi2, _safe_chi2r, _safe_n_points, _safe_n_free, _collect_param_list
