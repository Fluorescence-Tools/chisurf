"""Lightweight rigid-body data structures (legacy).

The hand-rolled spring/Verlet docking engine that once lived here has been
removed in favour of :mod:`...core.imp_engine` (IMP + IMP.bff). Only the
plain data classes used by the OLGA-style evaluators (:mod:`...core.evaluate`)
remain: :class:`RigidBody`, :class:`DistanceRestraint`, :class:`SpringParameters`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np



# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class SpringParameters:
    """Simulation parameters mirroring FPS ``FPSParameters``."""
    viscosity_factor: float = 0.85
    time_step_factor: float = 0.005
    max_iterations: int = 50000
    max_force: float = 100.0
    clash_tolerance: float = 0.0
    k_clash: float = 10.0
    rkT: float = 1.0
    E_tolerance: float = 1e-4
    K_tolerance: float = 1e-4
    F_tolerance: float = 1e-4
    T_tolerance: float = 1e-4
    optimize_selected: int = 0  # 0=all, 1=selected, 2=selected-then-all


# ---------------------------------------------------------------------------
# Rigid Body
# ---------------------------------------------------------------------------

@dataclass
class RigidBody:
    """A rigid body that can be translated and rotated as a unit.

    Attributes stored in the *body* (local) frame.
    """
    name: str
    atoms_local: np.ndarray  # (N, 4) xyzr in the local frame (com = 0)
    com: np.ndarray  # (3,) center of mass in global frame
    rotation: np.ndarray  # (3, 3) rotation from local → global
    translation: np.ndarray  # (3,) alias for com (for clarity)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))

    def global_coords(self) -> np.ndarray:
        """Return (N, 3) atom coordinates in the global frame."""
        return self.atoms_local[:, :3] @ self.rotation.T + self.com

    def global_xyzr(self) -> np.ndarray:
        """Return (N, 4) xyzr in the global frame."""
        coords = self.global_coords()
        return np.column_stack([coords, self.atoms_local[:, 3]])

    def apply_force_at_point(
        self,
        force: np.ndarray,
        point: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a force at a point on the body.

        Returns (linear_force_increment, torque_increment).
        """
        r = point - self.com
        torque = np.cross(r, force)
        return force, torque

    def random_shake(self, translation_scale: float = 5.0, rotation_scale: float = 0.5):
        """Randomly perturb position and orientation (for initial clash removal)."""
        self.com += np.random.uniform(-translation_scale, translation_scale, size=3)
        angle = np.random.uniform(-rotation_scale, rotation_scale)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis) + 1e-12
        rot = _rotation_matrix(axis, angle)
        self.rotation = self.rotation @ rot


# ---------------------------------------------------------------------------
# Distance restraint (harmonic spring)
# ---------------------------------------------------------------------------

@dataclass
class DistanceRestraint:
    """A single FRET distance restraint between two AV positions on two bodies.

    Positions are stored as offsets from each body's COM so they move
    with the body during docking.
    """
    name: str
    body_a: int
    offset_a: np.ndarray  # AV mean position relative to body A's COM
    body_b: int
    offset_b: np.ndarray  # AV mean position relative to body B's COM
    distance_exp: float
    error_neg: float
    error_pos: float
    distance_type: str = "RDAMean"
    forster_radius: float = 52.0
    active: bool = True
    position_name_a: str = ""
    position_name_b: str = ""
    sigma_rda: float = 0.0
    convfun: Optional[np.ndarray] = None
    transfer_function_type: str = "Polynomial"

    def global_position_a(self, bodies) -> np.ndarray:
        """Return AV mean position on body A in the global frame."""
        ba = bodies[self.body_a]
        return ba.com + ba.rotation @ self.offset_a

    def global_position_b(self, bodies) -> np.ndarray:
        """Return AV mean position on body B in the global frame."""
        bb = bodies[self.body_b]
        return bb.com + bb.rotation @ self.offset_b

    def get_effective_distance(self, rmp: float) -> float:
        """Apply the transfer function to the Rmp distance.

        Parameters
        ----------
        rmp : float
            Distance between mean positions.

        Returns
        -------
        float
            Effective distance.
        """
        from . import distance as _dist
        if self.distance_type == "Rmp" or self.transfer_function_type == "None":
            return rmp
        elif self.transfer_function_type == "Gaussian":
            if self.sigma_rda > 0.0:
                return float(_dist.gaussian_rmp_to_rda_mean(rmp, self.sigma_rda))
            return rmp
        elif self.transfer_function_type == "Polynomial":
            if self.convfun is not None:
                return float(_dist.polynomial_transfer(rmp, self.convfun))
            if self.sigma_rda > 0.0:
                return float(_dist.gaussian_rmp_to_rda_mean(rmp, self.sigma_rda))
            return rmp
        return rmp


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a (3,3) rotation matrix from an axis-angle pair."""
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a vector by an axis-angle (Rodrigues' formula)."""
    c = np.cos(angle)
    s = np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)
