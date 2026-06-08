from __future__ import annotations
import chisurf as cs

import threading
from typing import Any, Callable, Dict, List, Optional, Set


class Registry:
    """Central registry for O(1) lookup of persistent entities by UID.

    Provides thread-safe access to datasets, fits, parameters, and windows.
    Supports lifecycle hooks for automatic registration/unregistration.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._datasets: Dict[str, Any] = {}
        self._fits: Dict[str, Any] = {}
        self._parameters: Dict[str, Any] = {}
        self._windows: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable[..., None]]] = {
            "on_dataset_created": [],
            "on_dataset_removed": [],
            "on_fit_created": [],
            "on_fit_removed": [],
            "on_parameter_created": [],
            "on_parameter_removed": [],
        }

    def register_dataset(self, uid: str, obj: Any) -> None:
        with self._lock:
            self._datasets[uid] = obj
            for hook in self._hooks.get("on_dataset_created", []):
                try:
                    hook(uid, obj)
                except Exception:
                    pass

    def unregister_dataset(self, uid: str) -> None:
        with self._lock:
            obj = self._datasets.pop(uid, None)
            for hook in self._hooks.get("on_dataset_removed", []):
                try:
                    hook(uid, obj)
                except Exception:
                    pass

    def get_dataset(self, uid: str) -> Optional[Any]:
        with self._lock:
            return self._datasets.get(uid)

    def list_datasets(self) -> List[str]:
        with self._lock:
            return list(self._datasets.keys())

    def register_fit(self, uid: str, obj: Any) -> None:
        with self._lock:
            self._fits[uid] = obj
            for hook in self._hooks.get("on_fit_created", []):
                try:
                    hook(uid, obj)
                except Exception:
                    pass

    def unregister_fit(self, uid: str) -> None:
        with self._lock:
            obj = self._fits.pop(uid, None)
            for hook in self._hooks.get("on_fit_removed", []):
                try:
                    hook(uid, obj)
                except Exception:
                    pass

    def get_fit(self, uid: str) -> Optional[Any]:
        with self._lock:
            return self._fits.get(uid)

    def list_fits(self) -> List[str]:
        with self._lock:
            return list(self._fits.keys())

    def register_parameter(self, uid: str, obj: Any) -> None:
        with self._lock:
            self._parameters[uid] = obj
            for hook in self._hooks.get("on_parameter_created", []):
                try:
                    hook(uid, obj)
                except Exception:
                    pass

    def unregister_parameter(self, uid: str) -> None:
        with self._lock:
            obj = self._parameters.pop(uid, None)
            for hook in self._hooks.get("on_parameter_removed", []):
                try:
                    hook(uid, obj)
                except Exception:
                    pass

    def get_parameter(self, uid: str) -> Optional[Any]:
        with self._lock:
            return self._parameters.get(uid)

    def list_parameters(self) -> List[str]:
        with self._lock:
            return list(self._parameters.keys())

    def register_window(self, uid: str, obj: Any) -> None:
        with self._lock:
            self._windows[uid] = obj

    def unregister_window(self, uid: str) -> None:
        with self._lock:
            self._windows.pop(uid, None)

    def get_window(self, uid: str) -> Optional[Any]:
        with self._lock:
            return self._windows.get(uid)

    def list_windows(self) -> List[str]:
        with self._lock:
            return list(self._windows.keys())

    def add_hook(self, hook_name: str, callback: Callable[..., None]) -> None:
        with self._lock:
            if hook_name in self._hooks:
                self._hooks[hook_name].append(callback)

    def remove_hook(self, hook_name: str, callback: Callable[..., None]) -> None:
        with self._lock:
            if hook_name in self._hooks:
                try:
                    self._hooks[hook_name].remove(callback)
                except ValueError:
                    pass

    def clear(self) -> None:
        with self._lock:
            self._datasets.clear()
            self._fits.clear()
            self._parameters.clear()
            self._windows.clear()

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "datasets": len(self._datasets),
                "fits": len(self._fits),
                "parameters": len(self._parameters),
                "windows": len(self._windows),
            }


_global_registry: Optional[Registry] = None
_registry_lock = threading.Lock()


def get_registry() -> Registry:
    """Get the global registry instance."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = Registry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    with _registry_lock:
        if _global_registry is not None:
            _global_registry.clear()
        _global_registry = None


def register_dataset(uid: str, obj: Any) -> None:
    """Register a dataset in the global registry."""
    get_registry().register_dataset(uid, obj)


def register_fit(uid: str, obj: Any) -> None:
    """Register a fit in the global registry."""
    get_registry().register_fit(uid, obj)


def register_parameter(uid: str, obj: Any) -> None:
    """Register a parameter in the global registry."""
    get_registry().register_parameter(uid, obj)


def unregister_dataset(uid: str) -> None:
    """Unregister a dataset from the global registry."""
    get_registry().unregister_dataset(uid)


def unregister_fit(uid: str) -> None:
    """Unregister a fit from the global registry."""
    get_registry().unregister_fit(uid)


def unregister_parameter(uid: str) -> None:
    """Unregister a parameter from the global registry."""
    get_registry().unregister_parameter(uid)


def get_dataset(uid: str) -> Optional[Any]:
    """Get a dataset by UID."""
    return get_registry().get_dataset(uid)


def get_fit(uid: str) -> Optional[Any]:
    """Get a fit by UID."""
    return get_registry().get_fit(uid)


def get_parameter(uid: str) -> Optional[Any]:
    """Get a parameter by UID."""
    return get_registry().get_parameter(uid)


def sync_from_runtime() -> None:
    """Sync registry with current runtime state.

    Scans cs.fits and cs.imported_datasets and populates
    the registry with their UIDs.
    """
    reg = get_registry()
    reg.clear()

    for ds in getattr(cs, "imported_datasets", []):
        uid = getattr(ds, "unique_identifier", None)
        if uid:
            reg.register_dataset(uid, ds)

    for fg in getattr(cs, "fits", []):
        uid = getattr(fg, "unique_identifier", None)
        if uid:
            reg.register_fit(uid, fg)
            for local_fit in fg:
                for param in getattr(local_fit, "parameters_all", []):
                    param_uid = getattr(param, "unique_identifier", None)
                    if param_uid:
                        reg.register_parameter(param_uid, param)
