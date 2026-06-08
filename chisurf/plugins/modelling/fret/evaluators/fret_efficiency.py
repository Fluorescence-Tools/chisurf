"""FRET efficiency evaluator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Evaluator, EvaluatorResult


class FretEfficiencyEvaluator(Evaluator):
    """Evaluates the FRET efficiency <E> between two accessible volumes."""

    def __init__(
        self,
        name: str,
        position1: str,
        position2: str,
        forster_radius: float = 52.0,
    ) -> None:
        super().__init__(name)
        self.position1 = position1
        self.position2 = position2
        self.forster_radius = forster_radius

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        av1 = av_cache.get(self.position1)
        av2 = av_cache.get(self.position2)
        if av1 is None or av2 is None or not av1.has_volume or not av2.has_volume:
            val = 0.0
        else:
            from ..core.distance import _sample_av_distance
            d = _sample_av_distance(av1, av2)
            r = d[:, 0]
            w = d[:, 1]
            e = 1.0 / (1.0 + (r / self.forster_radius) ** 6.0)
            val = float(np.dot(w, e) / w.sum())
        return EvaluatorResult(self.name, val, "")
