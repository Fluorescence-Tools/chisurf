"""Distance evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Evaluator, EvaluatorResult


class DistanceEvaluator(Evaluator):
    """Evaluates the model distance between two accessible volumes."""

    def __init__(
        self,
        name: str,
        position1: str,
        position2: str,
        distance_type: str = "RDAMean",
        forster_radius: float = 52.0,
    ) -> None:
        super().__init__(name)
        self.position1 = position1
        self.position2 = position2
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
            val = 0.0
        else:
            from ..core.distance import model_distance
            val = model_distance(av1, av2, self.distance_type, self.forster_radius)
        return EvaluatorResult(self.name, float(val), "Å")


class DistanceDistributionEvaluator(Evaluator):
    """Evaluates the inter-AV distance distribution mean and histogram."""

    def __init__(
        self,
        name: str,
        position1: str,
        position2: str,
        rda_min: float = 1.0,
        rda_max: float = 200.0,
        n_rda_bins: int = 100,
    ) -> None:
        super().__init__(name)
        self.position1 = position1
        self.position2 = position2
        self.rda_min = rda_min
        self.rda_max = rda_max
        self.n_rda_bins = n_rda_bins

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        av1 = av_cache.get(self.position1)
        av2 = av_cache.get(self.position2)
        if av1 is None or av2 is None or not av1.has_volume or not av2.has_volume:
            return EvaluatorResult(self.name, 0.0, "Å", {"histogram": [], "bin_edges": []})

        from ..core.distance import average_distance, histogram_rda
        mean_val = average_distance(av1, av2)
        p, rda_axis = histogram_rda(
            av1,
            av2,
            rda_min=self.rda_min,
            rda_max=self.rda_max,
            n_rda_bins=self.n_rda_bins,
            normalize=True,
        )
        return EvaluatorResult(
            self.name,
            float(mean_val),
            "Å",
            {"histogram": p.tolist(), "bin_edges": rda_axis.tolist()},
        )
