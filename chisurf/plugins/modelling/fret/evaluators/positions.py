"""Positions and AV volume evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Evaluator, EvaluatorResult


class PositionEvaluator(Evaluator):
    """Evaluates the number of points in an accessible volume."""

    def __init__(self, name: str, position_name: str) -> None:
        super().__init__(name)
        self.position_name = position_name

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        av = av_cache.get(self.position_name)
        val = av.n_points if av is not None else 0
        return EvaluatorResult(self.name, float(val), "count")


class AVVolumeEvaluator(Evaluator):
    """Evaluates the physical volume of an accessible volume in Å³."""

    def __init__(self, name: str, position_name: str) -> None:
        super().__init__(name)
        self.position_name = position_name

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        av = av_cache.get(self.position_name)
        if av is None or not av.has_volume:
            val = 0.0
        else:
            val = av.n_points * (av.grid_step ** 3)
        return EvaluatorResult(self.name, float(val), "Å³")
