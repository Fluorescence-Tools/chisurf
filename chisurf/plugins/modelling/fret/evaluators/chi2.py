"""Chi2 evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Evaluator, EvaluatorResult


class Chi2Evaluator(Evaluator):
    """Evaluates total chi2 over a set of distance restraints."""

    def __init__(self, name: str, restraints: List[Dict[str, Any]]) -> None:
        super().__init__(name)
        self.restraints = restraints

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        from ..core.distance import model_distance, chi2_score
        total_chi2 = 0.0
        for r in self.restraints:
            pos1 = r["position1"]
            pos2 = r["position2"]
            av1 = av_cache.get(pos1)
            av2 = av_cache.get(pos2)
            if av1 is None or av2 is None or not av1.has_volume or not av2.has_volume:
                continue

            d_type = r.get("distance_type", "RDAMean")
            f_rad = r.get("forster_radius", 52.0)
            d_model = model_distance(av1, av2, d_type, f_rad)

            d_exp = float(r.get("distance", 0.0))
            err_neg = float(r.get("error_neg", 5.0))
            err_pos = float(r.get("error_pos", 5.0))

            total_chi2 += chi2_score(d_model, d_exp, err_neg, err_pos)

        return EvaluatorResult(self.name, total_chi2, "")


class ReducedChi2Evaluator(Evaluator):
    """Evaluates reduced chi2, i.e., chi2 / (n_valid - 1)."""

    def __init__(self, name: str, restraints: List[Dict[str, Any]]) -> None:
        super().__init__(name)
        self.restraints = restraints

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        from ..core.distance import model_distance, chi2_score
        total_chi2 = 0.0
        n_valid = 0
        for r in self.restraints:
            pos1 = r["position1"]
            pos2 = r["position2"]
            av1 = av_cache.get(pos1)
            av2 = av_cache.get(pos2)
            if av1 is None or av2 is None or not av1.has_volume or not av2.has_volume:
                continue

            d_type = r.get("distance_type", "RDAMean")
            f_rad = r.get("forster_radius", 52.0)
            d_model = model_distance(av1, av2, d_type, f_rad)

            d_exp = float(r.get("distance", 0.0))
            err_neg = float(r.get("error_neg", 5.0))
            err_pos = float(r.get("error_pos", 5.0))

            total_chi2 += chi2_score(d_model, d_exp, err_neg, err_pos)
            n_valid += 1

        if n_valid > 1:
            val = total_chi2 / (n_valid - 1)
        else:
            val = total_chi2

        return EvaluatorResult(self.name, val, "")


class Chi2ContributionEvaluator(Evaluator):
    """Evaluates the chi2 contribution for one specific distance."""

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

        from ..core.distance import model_distance, chi2_score
        d_model = model_distance(av1, av2, self.distance_type, self.forster_radius)
        contrib = chi2_score(d_model, self.distance, self.error_neg, self.error_pos)
        return EvaluatorResult(self.name, float(contrib), "")
