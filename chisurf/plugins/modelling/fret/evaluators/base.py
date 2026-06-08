"""Base definitions for OLGA-style evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EvaluatorResult:
    """Output of a single evaluator on one structure/frame.

    Parameters
    ----------
    name : str
        Unique name of the evaluated metric.
    value : float
        Numerical value of the evaluated metric.
    unit : str, optional
        Unit of the value (e.g. 'Å', 'Å³', 'rad').
    extra : dict, optional
        Any extra metadata or detail arrays (e.g. histograms).
    """
    name: str
    value: float
    unit: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Abstract base for all OLGA-style evaluators.

    Every evaluator class has a configuration that can be serialized to/from
    dict format for JSON round-trip compatibility in `fps.json`.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        """Evaluate this metric using a pre-computed AV cache and optional bodies.

        Parameters
        ----------
        av_cache : dict
            Maps position name -> AccessibleVolume. Pre-computed by caller.
        bodies : list of RigidBody, optional
            Rigid bodies (needed for geometry calculations).

        Returns
        -------
        result : EvaluatorResult
            The calculated metric.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialise evaluator configuration to a JSON-safe dict."""
        d = {"type": self.__class__.__name__, "name": self.name}
        for k, v in vars(self).items():
            if k != "name":
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Evaluator:
        """Deserialise evaluator from a dict.

        Parameters
        ----------
        d : dict
            JSON dict representing the evaluator config.

        Returns
        -------
        evaluator : Evaluator
            An instantiated evaluator object.
        """
        kwargs = {k: v for k, v in d.items() if k != "type"}
        return cls(**kwargs)


class EvaluationStorage:
    """Accumulates EvaluatorResult objects across frames/structures.

    Used by evaluate.py to collect results from directory or trajectory sweeps.
    """

    def __init__(self) -> None:
        self.filenames: List[str] = []
        self.results: Dict[str, List[float]] = {}

    def add_frame(self, filename: str, frame_results: Dict[str, EvaluatorResult]) -> None:
        """Add results for a single structure/frame.

        Parameters
        ----------
        filename : str
            Filename or frame identifier.
        frame_results : dict
            Map of evaluator name -> EvaluatorResult.
        """
        self.filenames.append(filename)
        for name, res in frame_results.items():
            if name not in self.results:
                self.results[name] = [np.nan] * (len(self.filenames) - 1)
            self.results[name].append(res.value)

        # Fill in NaN for any evaluators that did not report in this frame
        for name in self.results:
            if name not in frame_results:
                self.results[name].append(np.nan)

    def to_dataframe(self) -> Any:
        """Convert accumulated results to a pandas DataFrame."""
        import pandas as pd
        data = {"filename": self.filenames}
        for name, vals in self.results.items():
            data[name] = vals
        return pd.DataFrame(data)

    def to_csv(self, path: str) -> None:
        """Write accumulated results to a CSV file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        import csv
        headers = ["filename"] + list(self.results.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i, fname in enumerate(self.filenames):
                row = [fname]
                for name in self.results:
                    row.append(self.results[name][i])
                writer.writerow(row)
