from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from chisurf.client import ChisurfClient


class DataProxy:
    """Typed wrapper for a dataset's data sub-dict (fit.data). Read-only."""

    def __init__(self, data: dict, client: Optional[ChisurfClient] = None):
        self._data = data
        self._client = client

    @property
    def name(self) -> Optional[str]:
        return self._data.get("name")

    @property
    def uid(self) -> Optional[str]:
        return self._data.get("uid")

    @property
    def filename(self) -> Optional[str]:
        return self._data.get("filename")

    @property
    def experiment(self) -> Optional[str]:
        return self._data.get("experiment")

    def __repr__(self) -> str:
        return f"<DataProxy {self.name}>"


class ParameterProxy:
    """Typed wrapper for a parameter dict with explicit mutation methods."""

    def __init__(self, data: dict, client: Optional[ChisurfClient] = None):
        self._data = data
        self._client = client

    # --- Read-only properties ---

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def fit_uid(self) -> Optional[str]:
        return self._data.get("fit_uid")

    @property
    def value(self) -> Optional[float]:
        return self._data.get("value")

    @property
    def fixed(self) -> bool:
        return bool(self._data.get("fixed", False))

    @property
    def bounds(self) -> Any:
        return self._data.get("bounds")

    @property
    def bounds_on(self) -> bool:
        return bool(self._data.get("bounds_on", False))

    @property
    def is_linked(self) -> bool:
        return bool(self._data.get("is_linked", False))

    @property
    def linked_to(self) -> str:
        return self._data.get("linked_to", "")

    @property
    def error_estimate(self) -> Optional[float]:
        return self._data.get("error_estimate")

    # --- Mutation methods ---

    def set_value(self, value: float) -> dict:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client.call("parameter.set_value", {
            "parameter_name": self.name, "value": value, "fit_uid": self.fit_uid,
        })

    def set_fixed(self, fixed: bool) -> dict:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client.call("parameter.set_fixed", {
            "parameter_name": self.name, "fixed": fixed, "fit_uid": self.fit_uid,
        })

    def set_bounds(self, bounds: tuple) -> dict:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client.call("parameter.set_bounds", {
            "parameter_name": self.name, "bounds": list(bounds), "fit_uid": self.fit_uid,
        })

    def set_bounds_on(self, bounds_on: bool) -> dict:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client.call("parameter.set_bounds_on", {
            "parameter_name": self.name, "bounds_on": bounds_on, "fit_uid": self.fit_uid,
        })

    def link_to(self, target_parameter_name: str, **kw) -> dict:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client.call("parameter.link", {
            "parameter_name": self.name, "target_parameter_name": target_parameter_name,
            "fit_uid": self.fit_uid, **kw,
        })

    def unlink(self) -> dict:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client.call("parameter.unlink", {
            "parameter_name": self.name, "fit_uid": self.fit_uid,
        })

    def __repr__(self) -> str:
        return f"<ParameterProxy {self.name}>"


class ModelProxy:
    """Typed wrapper for the model sub-dict of a fit. Read-only."""

    def __init__(self, data: dict, client: Optional[ChisurfClient] = None):
        self._data = data
        self._client = client

    @property
    def name(self) -> Optional[str]:
        return self._data.get("name")

    @property
    def n_points(self) -> Optional[int]:
        return self._data.get("n_points")

    @property
    def n_free(self) -> Optional[int]:
        return self._data.get("n_free")

    @property
    def chi2r(self) -> Optional[float]:
        return self._data.get("chi2r")

    @property
    def parameters_all(self) -> List[ParameterProxy]:
        plist = self._data.get("parameters_all", [])
        return [ParameterProxy(p, client=self._client) for p in plist]

    @property
    def parameters_all_dict(self) -> Dict[str, ParameterProxy]:
        return {p.name: p for p in self.parameters_all}

    def __repr__(self) -> str:
        return f"<ModelProxy {self.name}>"


class FitProxy:
    """Typed wrapper for a fit response dict with explicit action methods."""

    def __init__(self, data: dict, client: Optional[ChisurfClient] = None):
        self._data = data
        self._client = client

    # --- Read-only properties ---

    @property
    def uid(self) -> Optional[str]:
        return self._data.get("uid")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def type(self) -> str:
        return self._data.get("type", "")

    @property
    def index(self) -> Optional[int]:
        return self._data.get("index")

    @property
    def chi2(self) -> Optional[float]:
        return self._data.get("chi2")

    @property
    def chi2r(self) -> Optional[float]:
        return self._data.get("chi2r")

    @property
    def n_points(self) -> Optional[int]:
        return self._data.get("n_points")

    @property
    def n_free(self) -> Optional[int]:
        return self._data.get("n_free")

    @property
    def dataset_name(self) -> str:
        return self._data.get("dataset_name", "")

    @property
    def dataset_uid(self) -> Optional[str]:
        return self._data.get("dataset_uid")

    @property
    def model_name(self) -> str:
        return self._data.get("model_name", "")

    # --- Sub-object properties ---

    @property
    def data(self) -> Optional[DataProxy]:
        d = self._data.get("data")
        if d:
            return DataProxy(d, client=self._client)
        return None

    @property
    def model(self) -> Optional[ModelProxy]:
        m = self._data.get("model")
        if m:
            return ModelProxy(m, client=self._client)
        return None

    @property
    def parameters_all(self) -> List[ParameterProxy]:
        return self.model.parameters_all if self.model else []

    @property
    def parameters_all_dict(self) -> Dict[str, ParameterProxy]:
        return self.model.parameters_all_dict if self.model else {}

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._data.get("parameters", {})

    # --- RPC action methods ---

    def _require_client(self) -> ChisurfClient:
        if self._client is None:
            raise RuntimeError("No client available")
        return self._client

    def run(self, **kw) -> dict:
        return self._require_client().call("fit.run", {"fit_uid": self.uid, **kw})

    def save(self, filename: str, file_type: str = "csv", **kw) -> dict:
        return self._require_client().call("fit.save", {
            "fit_uid": self.uid, "filename": filename, "file_type": file_type, **kw,
        })

    def update(self, **kw) -> dict:
        return self._require_client().call("fit.update", {"fit_uid": self.uid, **kw})

    def set_result_idx(self, result_idx: int, **kw) -> dict:
        return self._require_client().call("fit.set_result_idx", {
            "fit_uid": self.uid, "result_idx": result_idx, **kw,
        })

    def set_dataset(self, dataset_index: Optional[int] = None, dataset_uid: Optional[str] = None) -> dict:
        return self._require_client().call("fit.set_dataset", {
            "fit_uid": self.uid, "dataset_index": dataset_index, "dataset_uid": dataset_uid,
        })

    def model_finalize(self, **kw) -> dict:
        return self._require_client().call("model.finalize", {"fit_uid": self.uid, **kw})

    def model_set_parse_function(self, function_name: str, **kw) -> dict:
        return self._require_client().call("model.set_parse_function", {
            "fit_uid": self.uid, "function_name": function_name, **kw,
        })

    def __repr__(self) -> str:
        return f"<FitProxy {self.name}>"


class DatasetProxy:
    """Typed wrapper for a dataset response dict."""

    def __init__(self, data: dict, client: Optional[ChisurfClient] = None):
        self._data = data
        self._client = client
        self._curve_cache: Optional[dict] = None

    # --- Read-only properties ---

    @property
    def uid(self) -> Optional[str]:
        return self._data.get("uid")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def type(self) -> str:
        return self._data.get("type", "")

    @property
    def index(self) -> Optional[int]:
        return self._data.get("index")

    @property
    def filename(self) -> str:
        return self._data.get("filename", "")

    @property
    def experiment(self) -> str:
        return self._data.get("experiment", "")

    @property
    def length(self) -> Optional[int]:
        return self._data.get("length")

    # --- Explicit curve data method ---

    def curve_data(self, refresh: bool = False) -> dict:
        if self._client is None:
            return {}
        if refresh or self._curve_cache is None:
            self._curve_cache = self._client.dataset__curve_data(dataset_uid=self.uid) or {}
        return self._curve_cache

    def __repr__(self) -> str:
        return f"<DatasetProxy {self.name}>"


class ProxyList:
    """Base class for list-like proxies backed by a ``ChisurfClient``."""

    def __init__(self, client: ChisurfClient):
        self._client = client
        self._cache: Optional[List[Dict[str, Any]]] = None

    def _fetch(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _invalidate(self) -> None:
        self._cache = None

    def __len__(self) -> int:
        return len(self._fetch())

    def __iter__(self) -> Iterator:
        return iter(self._wrap(d) for d in self._fetch())

    def __getitem__(self, index: int):
        items = self._fetch()
        if index < 0:
            index += len(items)
        return self._wrap(items[index])

    def __contains__(self, item: Any) -> bool:
        uid = getattr(item, "uid", None)
        if uid:
            return any(d.get("uid") == uid for d in self._fetch())
        return False

    def index(self, item: Any, start: int = 0, stop: Optional[int] = None) -> int:
        items = self._fetch()
        uid = getattr(item, "uid", None)
        if isinstance(item, (FitProxy, DatasetProxy)):
            uid = item.uid
        if uid:
            for i, d in enumerate(items):
                if i >= start and d.get("uid") == uid:
                    return i
        raise ValueError(f"{item} is not in list")

    def pop(self, index: int = -1) -> Any:
        items = self._fetch()
        actual = index if index >= 0 else len(items) + index
        if actual < 0 or actual >= len(items):
            raise IndexError("pop index out of range")
        uid = items[actual].get("uid")
        if uid:
            self._remove_items(uids=[uid])
        else:
            self._remove_items(indices=[actual])
        self._invalidate()
        return self._wrap(items[actual])

    def insert(self, index: int, item: Any) -> None:
        self._add_item(item)
        self._invalidate()

    def remove(self, item: Any) -> None:
        uid = getattr(item, "uid", None)
        if isinstance(item, (FitProxy, DatasetProxy)):
            uid = item.uid
        if uid:
            self._remove_items(uids=[uid])
        else:
            try:
                idx = self.index(item)
                self._remove_items(indices=[idx])
            except ValueError:
                raise ValueError(f"{item} is not in list") from None
        self._invalidate()

    def clear(self) -> None:
        self._clear_all()
        self._invalidate()

    def extend(self, items: Any) -> None:
        for item in items:
            self._add_item(item)
        self._invalidate()

    def __delitem__(self, index: Union[int, slice]) -> None:
        if isinstance(index, slice):
            indices = list(range(*index.indices(len(self._fetch()))))
            if indices:
                self._remove_items(indices=indices)
        else:
            actual = index if index >= 0 else len(self._fetch()) + index
            uid = self._fetch()[actual].get("uid")
            if uid:
                self._remove_items(uids=[uid])
            else:
                self._remove_items(indices=[actual])
        self._invalidate()

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        if isinstance(index, slice):
            self._clear_all()
        self._invalidate()

    def _wrap(self, data: dict):
        raise NotImplementedError

    def _add_item(self, item: Any) -> None:
        raise NotImplementedError

    def _remove_items(self, indices=None, uids=None) -> None:
        raise NotImplementedError

    def _clear_all(self) -> None:
        raise NotImplementedError


class ProxyDatasetList(ProxyList):
    """Proxy for ``chisurf.imported_datasets``."""

    def _fetch(self) -> List[Dict[str, Any]]:
        resp = self._client.dataset__list()
        return list(resp) if resp else []

    def _wrap(self, data: dict) -> DatasetProxy:
        return DatasetProxy(data, client=self._client)

    def _add_item(self, item: Any) -> None:
        pass

    def _remove_items(self, indices=None, uids=None) -> None:
        self._client.dataset__remove(dataset_indices=indices or [], dataset_uids=uids or [])

    def _clear_all(self) -> None:
        self._client.dataset__clear()


class ProxyFitList(ProxyList):
    """Proxy for ``chisurf.fits``."""

    def _fetch(self) -> List[Dict[str, Any]]:
        resp = self._client.fit__list()
        return list(resp) if resp else []

    def _wrap(self, data: dict) -> FitProxy:
        return FitProxy(data, client=self._client)

    def _add_item(self, item: Any) -> None:
        pass

    def _remove_items(self, indices=None, uids=None) -> None:
        self._client.fit__remove(fit_indices=indices or [], fit_uids=uids or [])

    def _clear_all(self) -> None:
        self._client.fit__clear()
