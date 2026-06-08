"""Geometry and rigid-body coordinate evaluators."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Evaluator, EvaluatorResult


class EulerAngleEvaluator(Evaluator):
    """Evaluates the ZYZ Euler angles of a rigid body rotation."""

    def __init__(self, name: str, body_id: int, angle_index: int) -> None:
        super().__init__(name)
        self.body_id = body_id
        self.angle_index = angle_index  # 0 for alpha, 1 for beta, 2 for gamma

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        if bodies is None or self.body_id >= len(bodies):
            return EvaluatorResult(self.name, 0.0, "rad")

        body = bodies[self.body_id]
        r = body.rotation

        beta = np.arccos(np.clip(r[2, 2], -1.0, 1.0))
        if np.abs(np.sin(beta)) > 1e-7:
            alpha = np.arctan2(r[1, 2], r[0, 2])
            gamma = np.arctan2(r[2, 1], -r[2, 0])
        else:
            alpha = 0.0
            gamma = np.arctan2(-r[0, 1], r[0, 0])

        angles = [alpha, beta, gamma]
        val = angles[self.angle_index]
        return EvaluatorResult(self.name, float(val), "rad")


class TranslationEvaluator(Evaluator):
    """Evaluates the center-of-mass coordinates of a rigid body."""

    def __init__(self, name: str, body_id: int, coordinate_index: int) -> None:
        super().__init__(name)
        self.body_id = body_id
        self.coordinate_index = coordinate_index  # 0 for x, 1 for y, 2 for z

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        if bodies is None or self.body_id >= len(bodies):
            return EvaluatorResult(self.name, 0.0, "Å")

        body = bodies[self.body_id]
        val = body.com[self.coordinate_index]
        return EvaluatorResult(self.name, float(val), "Å")


class MinDistanceEvaluator(Evaluator):
    """Evaluates the minimum distance between atoms of two rigid bodies."""

    def __init__(self, name: str, body_a_id: int, body_b_id: int) -> None:
        super().__init__(name)
        self.body_a_id = body_a_id
        self.body_b_id = body_b_id

    def evaluate(
        self,
        av_cache: Dict[str, Any],
        bodies: Optional[List[Any]] = None,
    ) -> EvaluatorResult:
        if bodies is None or self.body_a_id >= len(bodies) or self.body_b_id >= len(bodies):
            return EvaluatorResult(self.name, 0.0, "Å")

        from scipy.spatial.distance import cdist
        coords_a = bodies[self.body_a_id].global_coords()
        coords_b = bodies[self.body_b_id].global_coords()
        dists = cdist(coords_a, coords_b)
        val = dists.min()
        return EvaluatorResult(self.name, float(val), "Å")
