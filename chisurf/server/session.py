from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    """Mutable container for the server-side runtime state.

    This is the **single source of truth**. The server owns this state;
    the GUI connects as a client and reads/writes through ZMQ RPC.
    """

    datasets: List[Any] = field(default_factory=list)
    fits: List[Any] = field(default_factory=list)
    experiments: Dict[str, Any] = field(default_factory=dict)
    current_experiment: Optional[str] = None
    current_setup: Optional[str] = None
    current_fit_uid: Optional[str] = None

    def __post_init__(self):
        """Initialise the project registry after dataclass field assignment."""
        from chisurf.project.registry import Registry
        self.registry = Registry()

    # ── mutation helpers ────────────────────────────────────────────

    def add_dataset(self, dataset: Any) -> None:
        """Append a dataset to the session.

        Parameters
        ----------
        dataset : object
            Dataset instance.

        """
        self.datasets.append(dataset)

    def add_fit(self, fit: Any) -> None:
        """Append a fit to the session.

        Parameters
        ----------
        fit : object
            Fit instance.

        """
        self.fits.append(fit)

    @staticmethod
    def _resolve_uid(obj: Any) -> Optional[str]:
        """Extract the unique identifier from an object or dict.

        Parameters
        ----------
        obj : object or dict
            Object with a ``unique_identifier`` attribute or a dict with
            ``"unique_identifier"`` / ``"uid"`` key.

        """
        if isinstance(obj, dict):
            return obj.get("unique_identifier") or obj.get("uid")
        return getattr(obj, "unique_identifier", None)

    def remove_dataset(self, index: Optional[int] = None, uid: Optional[str] = None) -> bool:
        """Remove a dataset by index or uid.

        Parameters
        ----------
        index : int, optional
            Positional index.
        uid : str, optional
            Unique identifier.

        """
        if uid is not None:
            for i, d in enumerate(self.datasets):
                if self._resolve_uid(d) == uid:
                    self.datasets.pop(i)
                    return True
            return False
        if index is not None and 0 <= index < len(self.datasets):
            self.datasets.pop(index)
            return True
        return False

    def remove_fit(self, index: Optional[int] = None, uid: Optional[str] = None) -> bool:
        """Remove a fit by index or uid.

        Parameters
        ----------
        index : int, optional
            Positional index.
        uid : str, optional
            Unique identifier.

        """
        if uid is not None:
            for i, f in enumerate(self.fits):
                if self._resolve_uid(f) == uid:
                    self.fits.pop(i)
                    return True
            return False
        if index is not None and 0 <= index < len(self.fits):
            self.fits.pop(index)
            return True
        return False

    def clear(self) -> None:
        """Reset the session to a clean state."""
        self.datasets.clear()
        self.fits.clear()
        self.experiments.clear()
        self.current_experiment = None
        self.current_setup = None
        self.current_fit_uid = None
        self.registry.clear()

    # ── lookup helpers ──────────────────────────────────────────────

    def find_fit_by_uid(self, uid: str) -> Any:
        """Return a fit by its unique identifier, or ``None``.

        Parameters
        ----------
        uid : str
            Unique identifier.

        """
        for f in self.fits:
            if self._resolve_uid(f) == uid:
                return f
        return None

    def find_dataset_by_uid(self, uid: str) -> Any:
        """Return a dataset by its unique identifier, or ``None``.

        Parameters
        ----------
        uid : str
            Unique identifier.

        """
        for d in self.datasets:
            if self._resolve_uid(d) == uid:
                return d
        return None

    # ── snapshot ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session state to a dictionary."""
        return {
            "dataset_count": len(self.datasets),
            "fit_count": len(self.fits),
            "experiment_names": sorted(self.experiments.keys()),
            "current_experiment": self.current_experiment,
            "current_setup": self.current_setup,
            "current_fit_uid": self.current_fit_uid,
        }
