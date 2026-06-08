from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import clash


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


# ---------------------------------------------------------------------------
# Spring engine
# ---------------------------------------------------------------------------

class SpringEngine:
    """Verlet-spring integrator for FRET-restrained rigid-body docking.

    Mirrors FPS ``SpringEngine``.  Supports:
    - Asymmetric harmonic FRET distance restraints
    - vdW clash detection and forces
    - Viscosity damping
    - Convergence criteria (force / torque / kinetic energy)
    """

    bodies: List[RigidBody]
    restraints: List[DistanceRestraint]
    params: SpringParameters

    def __init__(
        self,
        bodies: List[RigidBody],
        restraints: List[DistanceRestraint],
        params: Optional[SpringParameters] = None,
    ):
        self.bodies = bodies
        self.restraints = restraints
        self.params = params or SpringParameters()
        self._iteration: int = 0
        self._converged: bool = False
        self._energy_history: List[float] = []
        self._force_norms: List[float] = []
        self._torque_norms: List[float] = []

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def iteration(self) -> int:
        return self._iteration

    def _compute_fret_forces(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Compute FRET restraint forces on each body.

        Returns list of (body_index, force, torque) increments.
        """
        n_body = len(self.bodies)
        increments = [(i, np.zeros(3), np.zeros(3)) for i in range(n_body)]

        for rst in self.restraints:
            if not rst.active:
                continue

            pa = rst.global_position_a(self.bodies)
            pb = rst.global_position_b(self.bodies)

            delta = pb - pa
            d = np.linalg.norm(delta)
            if d < 1e-12:
                continue
            direction = delta / d

            d_eff = rst.get_effective_distance(d)
            deviation = d_eff - rst.distance_exp
            k = (2.0 / (rst.error_pos ** 2)) if deviation > 0 else (2.0 / (rst.error_neg ** 2))
            f_mag = min(k * deviation, self.params.max_force)

            ba = self.bodies[rst.body_a]
            bb = self.bodies[rst.body_b]

            # A pulled toward B, B pulled toward A (deviation > 0 = too far apart)
            f_inc_a, t_inc_a = ba.apply_force_at_point(direction * f_mag, pa)
            f_inc_b, t_inc_b = bb.apply_force_at_point(-direction * f_mag, pb)

            i_a, f_a_accum, t_a_accum = increments[rst.body_a]
            i_b, f_b_accum, t_b_accum = increments[rst.body_b]
            increments[rst.body_a] = (i_a, f_a_accum + f_inc_a, t_a_accum + t_inc_a)
            increments[rst.body_b] = (i_b, f_b_accum + f_inc_b, t_b_accum + t_inc_b)

        return increments

    def _compute_clash_forces(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Compute vdW clash forces between all body pairs."""
        n_body = len(self.bodies)
        increments = [(i, np.zeros(3), np.zeros(3)) for i in range(n_body)]

        for i in range(n_body):
            for j in range(i + 1, n_body):
                xyzr_i = self.bodies[i].global_xyzr()
                xyzr_j = self.bodies[j].global_xyzr()
                E, fi, fj = clash.body_clash_energy(xyzr_i, xyzr_j, self.params.k_clash)
                if E == 0:
                    continue
                # Convert per-atom forces to body-level force + torque
                com_i = self.bodies[i].com
                com_j = self.bodies[j].com
                f_i = fi.sum(axis=0)
                f_j = fj.sum(axis=0)

                # Compute per-atom contributions to torque
                t_i = np.cross(xyzr_i[:, :3] - com_i, fi).sum(axis=0)
                t_j = np.cross(xyzr_j[:, :3] - com_j, fj).sum(axis=0)

                # Accumulate
                idx_i, f_acc_i, t_acc_i = increments[i]
                idx_j, f_acc_j, t_acc_j = increments[j]
                increments[i] = (idx_i, f_acc_i + f_i, t_acc_i + t_i)
                increments[j] = (idx_j, f_acc_j + f_j, t_acc_j + t_j)

        return increments

    def _check_convergence(self, forces: np.ndarray, torques: np.ndarray) -> bool:
        """Check force, torque, and kinetic energy convergence.

        Parameters
        ----------
        forces : (n_body, 3) — total force on each body
        torques : (n_body, 3) — total torque on each body
        """
        n_body = len(self.bodies)
        f_norm = np.sqrt(np.mean(np.sum(forces ** 2, axis=1)))
        t_norm = np.sqrt(np.mean(np.sum(torques ** 2, axis=1)))
        ke = 0.5 * sum(
            b.mass * np.dot(b.velocity, b.velocity)
            + np.dot(b.angular_velocity, b.inertia @ b.angular_velocity)
            for b in self.bodies
        )
        self._force_norms.append(f_norm)
        self._torque_norms.append(t_norm)

        if f_norm < self.params.F_tolerance and t_norm < self.params.T_tolerance:
            return True
        if n_body > 0 and ke / n_body < self.params.K_tolerance:
            return True
        return False

    def simulate(self) -> bool:
        """Run the Verlet-spring integration until convergence or max iterations.

        Returns
        -------
        converged : bool
        """
        self._iteration = 0
        self._converged = False
        self._energy_history = []
        self._force_norms = []
        self._torque_norms = []
        n_body = len(self.bodies)

        dt = self.params.time_step_factor
        visc = self.params.viscosity_factor
        n_bodies = len(self.bodies)

        for _ in range(self.params.max_iterations):
            # 1. Compute FRET forces
            fret_incs = self._compute_fret_forces()
            clash_incs = self._compute_clash_forces()

            total_forces = np.zeros((n_body, 3))
            total_torques = np.zeros((n_body, 3))
            for idx, f, t in fret_incs:
                total_forces[idx] += f
                total_torques[idx] += t
            for idx, f, t in clash_incs:
                total_forces[idx] += f
                total_torques[idx] += t

            # 2. Integrate Verlet step
            for i, body in enumerate(self.bodies):
                f = total_forces[i]
                t = total_torques[i]

                # Linear: v *= damping + (F/m) * dt
                body.velocity = body.velocity * visc + (f / max(body.mass, 1e-12)) * dt
                body.com = body.com + body.velocity * dt

                # Angular: w *= damping + (I^{-1} @ torque) * dt
                inv_inertia = np.linalg.inv(body.inertia + np.eye(3) * 1e-12)
                body.angular_velocity = body.angular_velocity * visc + (
                    inv_inertia @ t
                ) * dt
                w = body.angular_velocity
                w_norm = np.linalg.norm(w)
                if w_norm > 1e-14:
                    axis = w / w_norm
                    angle = w_norm * dt
                    body.rotation = body.rotation @ _rotation_matrix(axis, angle)

                # Renormalise rotation (prevent drift)
                u, _, vh = np.linalg.svd(body.rotation)
                body.rotation = u @ vh

            # 3. Update AV attachment point positions (move with body)
            #    In FPS the AV mean positions are recomputed after docking;
            #    during docking the positions are estimated.
            #    Here we track them as vectors in the body frame.
            #    For simplicity, we'll recompute them each iteration from the
            #    original offset * rotation.

            self._iteration += 1
            if self._check_convergence(total_forces, total_torques):
                self._converged = True
                return True

        return False

    def get_energy(self) -> float:
        """Compute the total current energy (clash + restraint)."""
        # Restraint energy
        e_restraint = 0.0
        for rst in self.restraints:
            if not rst.active:
                continue
            pa = rst.global_position_a(self.bodies)
            pb = rst.global_position_b(self.bodies)
            d = np.linalg.norm(pb - pa)
            d_eff = rst.get_effective_distance(d)
            deviation = d_eff - rst.distance_exp
            k = (2.0 / (rst.error_pos ** 2)) if deviation > 0 else (2.0 / (rst.error_neg ** 2))
            e_restraint += 0.5 * k * deviation * deviation

        # Clash energy
        e_clash = 0.0
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                es, _, _ = clash.body_clash_energy(
                    self.bodies[i].global_xyzr(),
                    self.bodies[j].global_xyzr(),
                    self.params.k_clash,
                )
                e_clash += es

        return e_restraint + e_clash

    def get_clash_energy(self) -> float:
        """Compute only the vdW clash energy."""
        e_clash = 0.0
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                es, _, _ = clash.body_clash_energy(
                    self.bodies[i].global_xyzr(),
                    self.bodies[j].global_xyzr(),
                    self.params.k_clash,
                )
                e_clash += es
        return e_clash

    def get_restraint_energy(self) -> float:
        """Compute only the restraint (FRET) energy."""
        e = 0.0
        for rst in self.restraints:
            if not rst.active:
                continue
            pa = rst.global_position_a(self.bodies)
            pb = rst.global_position_b(self.bodies)
            d = np.linalg.norm(pb - pa)
            d_eff = rst.get_effective_distance(d)
            deviation = d_eff - rst.distance_exp
            k = (2.0 / (rst.error_pos ** 2)) if deviation > 0 else (2.0 / (rst.error_neg ** 2))
            e += 0.5 * k * deviation * deviation
        return e
