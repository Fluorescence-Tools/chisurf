from __future__ import annotations

from typing import Any, Dict, List, Optional


class FitListAdapter:
    """DTO-based adapter for fit list display.

    Provides a migration path from ``chisurf.fits`` (real objects) to
    ``chisurf.api.list_fits()`` (DTO dicts). Widgets can use this adapter
    to gradually switch from direct object access to DTO-based views.

    Usage (before):
        fit = chisurf.fits[idx]
        name = fit.name
        chi2 = fit.chi2

    Usage (after):
        fits = FitListAdapter(api)
        fit = fits[idx]
        name = fit["name"]
        chi2 = fit["chi2"]
    """

    def __init__(self, api: Any):
        self._api = api

    def refresh(self) -> List[Dict[str, Any]]:
        """Re-fetch the fit list from the API."""
        return self._api.list_fits()

    def __len__(self) -> int:
        return len(self._api.list_fits())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        fits = self._api.list_fits()
        if isinstance(index, slice):
            return fits[index]
        if 0 <= index < len(fits):
            return fits[index]
        raise IndexError(f"fit index {index} out of range (count={len(fits)})")

    def __iter__(self):
        return iter(self._api.list_fits())

    def __contains__(self, uid: str) -> bool:
        return any(f.get("uid") == uid for f in self._api.list_fits())

    def get_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        for f in self._api.list_fits():
            if f.get("uid") == uid:
                return f
        return None

    def fit_count(self) -> int:
        return len(self._api.list_fits())


class DatasetListAdapter:
    """DTO-based adapter for dataset list display."""

    def __init__(self, api: Any):
        self._api = api

    def refresh(self) -> List[Dict[str, Any]]:
        return self._api.list_datasets()

    def __len__(self) -> int:
        return len(self._api.list_datasets())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        datasets = self._api.list_datasets()
        if isinstance(index, slice):
            return datasets[index]
        if 0 <= index < len(datasets):
            return datasets[index]
        raise IndexError(f"dataset index {index} out of range (count={len(datasets)})")

    def __iter__(self):
        return iter(self._api.list_datasets())

    def __contains__(self, uid: str) -> bool:
        return any(d.get("uid") == uid for d in self._api.list_datasets())

    def get_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        for d in self._api.list_datasets():
            if d.get("uid") == uid:
                return d
        return None

    def dataset_count(self) -> int:
        return len(self._api.list_datasets())