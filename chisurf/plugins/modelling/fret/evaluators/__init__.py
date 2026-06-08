"""OLGA-style evaluators package."""

from __future__ import annotations

from typing import Any, Dict

from .base import Evaluator, EvaluatorResult, EvaluationStorage
from .positions import PositionEvaluator, AVVolumeEvaluator
from .distance import DistanceEvaluator, DistanceDistributionEvaluator
from .fret_efficiency import FretEfficiencyEvaluator
from .chi2 import Chi2Evaluator, ReducedChi2Evaluator, Chi2ContributionEvaluator
from .residuals import WeightedResidualEvaluator
from .geometry import EulerAngleEvaluator, TranslationEvaluator, MinDistanceEvaluator
from .av_metrics import AVSizeEvaluator, AVSphereOverlapEvaluator

EVALUATOR_CLASSES = {
    "PositionEvaluator": PositionEvaluator,
    "AVVolumeEvaluator": AVVolumeEvaluator,
    "DistanceEvaluator": DistanceEvaluator,
    "DistanceDistributionEvaluator": DistanceDistributionEvaluator,
    "FretEfficiencyEvaluator": FretEfficiencyEvaluator,
    "Chi2Evaluator": Chi2Evaluator,
    "ReducedChi2Evaluator": ReducedChi2Evaluator,
    "Chi2ContributionEvaluator": Chi2ContributionEvaluator,
    "WeightedResidualEvaluator": WeightedResidualEvaluator,
    "EulerAngleEvaluator": EulerAngleEvaluator,
    "TranslationEvaluator": TranslationEvaluator,
    "MinDistanceEvaluator": MinDistanceEvaluator,
    "AVSizeEvaluator": AVSizeEvaluator,
    "AVSphereOverlapEvaluator": AVSphereOverlapEvaluator,
}


def from_dict(d: Dict[str, Any]) -> Evaluator:
    """Instantiate an Evaluator from its serialized dictionary format.

    Parameters
    ----------
    d : dict
        Serialized representation of the evaluator.

    Returns
    -------
    evaluator : Evaluator
        Instantiated evaluator.
    """
    etype = d.get("type")
    if not etype:
        raise ValueError("Missing 'type' key in evaluator specification")
    cls = EVALUATOR_CLASSES.get(etype)
    if not cls:
        raise ValueError(f"Unknown evaluator type: '{etype}'")
    return cls.from_dict(d)
