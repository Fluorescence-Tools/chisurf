"""Weighted residual evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Evaluator, EvaluatorResult


class WeightedResidualEvaluator(Evaluator):
    """Evaluates the weighted residual (d_model - d_exp) / sigma."""

    def __init__(
        self,
        name: str,
        position1: str,
        position2: str,
        distance: float,
        error_neg: float,
        error_pos: float,
        distance_type: str = "RDAMean",
        forster_radius: float = 52.0,
    ) -> None:
        super().__init__(name)
        self.position1 = position1
        self.position2 = position2
        self.distance = distance
        self.error_neg = error_neg
        self.error_pos = error_pos
        self.distance_type = distance_type
        self.forster_radius = forster_radius

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        av1 = av_cache.get(self.position1)
        av2 = av_cache.get(self.position2)
        if av1 is None or av2 is None or not av1.has_volume or not av2.has_volume:
            return EvaluatorResult(self.name, 0.0, "")

        from ..core.distance import model_distance
        d_model = model_distance(av1, av2, self.distance_type, self.forster_radius)
        delta = d_model - self.distance
        err = self.error_neg if delta < 0 else self.error_pos
        if err <= 0:
            val = 0.0
        else:
            val = delta / err
        return EvaluatorResult(self.name, float(val), "")
