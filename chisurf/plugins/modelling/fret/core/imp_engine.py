"""IMP + IMP.bff FRET-restrained rigid-body docking engine.

This module is a thin shim around the Integrative Modeling Platform (IMP),
its PMI sampling machinery, and ``IMP.bff`` (the Bayesian-Framework-for-
Fluorescence module that provides accessible-volume decorators and the
``AVNetworkRestraint`` FRET scoring function).

It replaces the legacy hand-rolled spring/Verlet engine
(``engine.py``/``docking.py``/``sampling.py``/``refine.py``/``bootstrap.py``)
with a real IMP model:

* each input PDB (or ``body_id`` group) becomes an :class:`IMP.core.RigidBody`;
* ``IMP.bff.restraints.AVNetworkRestraintWrapper`` reads the ``fps.json`` file
  directly, builds the accessible volumes at the labelling sites, attaches them
  to the rigid bodies, and scores the model's R_DA / R_DA_E / Rmp against the
  experimental distances (asymmetric chi-square);
* ``IMP.core.ExcludedVolumeRestraint`` provides clash repulsion;
* sampling uses the PMI ``ReplicaExchange`` macro driving
  ``IMP.core.RigidBodyMover`` movers (Monte-Carlo, optional simulated
  annealing), which natively writes RMF trajectories and best-scoring PDBs;
* ``ConjugateGradients`` provides local refinement.

The single input format is ``fps.json`` (produced/edited by the
``fps_json_editor`` plugin and parsed by :mod:`...core.io`). ``IMP.bff`` parses
it natively for scoring, so the path is passed straight through.

All public functions raise a clear :class:`RuntimeError` when ``IMP``/``IMP.bff``
is unavailable (see :func:`require_imp`).

References
----------
- ``imp.bff/examples/structure/t4l_pmi.py`` (canonical PMI FRET-docking example)
- Kalinin, S., Peulen, T., et al. (2012) *Nat. Methods* 9, 1218.
- Dimura, M., Peulen, T., et al. (2016) *Nat. Commun.* 7, 10947.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional IMP import (hard requirement for this engine, soft at import time)
# ---------------------------------------------------------------------------

_IMP_IMPORT_ERROR: Optional[BaseException] = None
try:  # pragma: no cover - import guard
    import IMP
    import IMP.algebra
    import IMP.atom
    import IMP.core
    import IMP.container
    import IMP.bff
    import IMP.bff.restraints
    import IMP.rmf
    import IMP.pmi
    import IMP.pmi.tools
    import IMP.pmi.macros
    import RMF

    _HAS_IMP = bool(hasattr(IMP.bff, "AV"))
except Exception as exc:  # pragma: no cover - import guard
    _HAS_IMP = False
    _IMP_IMPORT_ERROR = exc


_REQUIRE_MSG = (
    "FRET docking requires IMP with the bff module (IMP.bff.AV / "
    "AVNetworkRestraint), IMP.pmi and IMP.rmf. Install a conda-forge `imp` "
    ">= 2.23 build that ships bff into the active environment, e.g.\n"
    "    conda install -c conda-forge 'imp>=2.23'\n"
    "and verify with `python -c \"import IMP.bff; IMP.bff.AV\"`."
)


def require_imp() -> None:
    """Raise a helpful :class:`RuntimeError` if IMP/IMP.bff is unavailable.

    Raises
    ------
    RuntimeError
        When IMP, IMP.bff, IMP.pmi or IMP.rmf cannot be imported.
    """
    if not _HAS_IMP:
        raise RuntimeError(f"{_REQUIRE_MSG}\n\nOriginal import error: {_IMP_IMPORT_ERROR!r}")


def has_imp() -> bool:
    """Return ``True`` when the IMP/IMP.bff backend is importable."""
    return _HAS_IMP




def _ensure_output_dir(output_dir: str) -> str:
    """Validate and create ``output_dir``; reject an empty path early.

    ``os.makedirs("")`` raises a cryptic ``FileNotFoundError``; an empty output
    directory almost always means the caller forgot to set it, so fail with a
    clear message instead.
    """
    if not output_dir or not str(output_dir).strip():
        raise ValueError(
            "output_dir is empty — choose an output directory for the "
            "RMF/PDB/CSV results before running."
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


#: Shared FRET transfer-function converters keyed by (Forster radius, sigma).
#: Building one costs ~100 ms (a 128-point scipy lookup), and a typical
#: experiment reuses a handful of (R0, sigma) combinations across dozens of
#: distances — so cache them instead of rebuilding per restraint/pair.
_CONVERTER_CACHE: Dict[Tuple[float, float], object] = {}


def _get_converter(forster_radius: float, sigma: float):
    """Return a cached :class:`IMP.bff.tools.FRETDistanceConverter`.

    The converter's lookup tables depend only on ``forster_radius`` and
    ``sigma``; ``__call__`` mutates only the (synchronously read) centre
    distance, so a shared instance is safe.
    """
    key = (round(float(forster_radius), 3), round(float(sigma), 3))
    dc = _CONVERTER_CACHE.get(key)
    if dc is None:
        dc = IMP.bff.tools.FRETDistanceConverter(
            forster_radius=forster_radius, sigma=sigma,
            distance_range=(1.0, 2.5 * forster_radius))
        _CONVERTER_CACHE[key] = dc
    return dc


#: Converted-on-the-fly fps.json paths, keyed by (source .txt, pdb paths).
_FPS_JSON_CACHE: Dict[Tuple[str, Tuple[str, ...]], str] = {}


def ensure_fps_json(fps_path: str, pdb_paths: Sequence[str]) -> str:
    """Return a path to an ``fps.json``, converting legacy C# files if needed.

    Accepts either a native ``fps.json`` (returned as-is) or the original FPS /
    C# tab-separated labelling file (``LPs*.txt``; the sibling ``Distances.txt``
    is picked up automatically). The C# format is converted to ``fps.json`` once
    and cached, so FPS-native inputs work end-to-end without a manual conversion
    step. ``pdb_paths`` is needed to resolve atom indices to residues.
    """
    p = str(fps_path)
    if p.endswith(".json"):
        return p
    key = (os.path.abspath(p), tuple(str(x) for x in pdb_paths))
    cached = _FPS_JSON_CACHE.get(key)
    if cached and os.path.exists(cached):
        return cached
    import tempfile

    from . import io as _io
    positions, distances, score_sets, extra = _io.read_fps_json(
        p, pdb_paths=list(pdb_paths))
    fd, out = tempfile.mkstemp(suffix=".fps.json", prefix="fps_")
    os.close(fd)
    _io.write_fps_json(out, positions, distances, score_sets or None, extra or None)
    _FPS_JSON_CACHE[key] = out
    return out


#: One representative backbone bead per residue used for the coarse clash term.
_COARSE_BACKBONE = ("CA", "P", "C1'", "C4'")


def _clash_container(model, root, *, coarse=True, bead_radius=2.5):
    """Singleton container for the excluded-volume clash term.

    The atomistic close-pair search dominates the per-iteration cost of
    minimisation docking, so by default we coarse-grain to one (enlarged)
    backbone bead per residue — IMP's excluded volume is rigid-body aware, so
    intra-body bead pairs are skipped and only inter-body clashes are scored.
    """
    leaves = IMP.atom.get_leaves(root)
    if coarse:
        beads = []
        for p in leaves:
            if not IMP.atom.Atom.get_is_setup(p):
                continue
            if IMP.atom.Atom(p).get_atom_type().get_string().strip() in _COARSE_BACKBONE:
                if IMP.core.XYZR.get_is_setup(p):
                    IMP.core.XYZR(p).set_radius(bead_radius)
                beads.append(p)
        if beads:
            leaves = beads
    return IMP.container.ListSingletonContainer(model, leaves)


# ---------------------------------------------------------------------------
# Derivative-enabled mean-distance restraint (drives gradient minimisation)
# ---------------------------------------------------------------------------

if _HAS_IMP:

    class MeanDistanceRestraint(IMP.Restraint):
        """FRET mean-position distance restraint *with* Cartesian derivatives.

        The model distance is the plain point-to-point separation ``d_mp`` of the
        two AV-mean particles; the (cached, hence cheap) FRET transfer function
        converts it to the modelled observable every evaluation and scores it with
        the measurement's asymmetric chi2 — the same scoring as the Monte-Carlo
        path. IMP.bff's ``AVMeanDistanceRestraint`` leaves the derivative
        accumulator untouched (sample-only); this subclass adds gradients so an
        IMP optimiser can dock by minimisation.

        IMP trick for the gradient: the score depends on the coordinates only
        through ``d_mp``, so take ``dScore/d_mp`` numerically (a cheap 1-D central
        difference through the cached transfer function) and map it onto the two
        particles analytically via ``d(d_mp)/dx = ±r̂``, accumulating with
        :meth:`IMP.core.XYZ.add_to_derivatives`.
        """

        def __init__(self, m, measurement, p1, p2, forster_radius, distance_type, sigma):
            IMP.Restraint.__init__(self, m, "MeanDistanceRestraint%1%")
            self.meas = measurement
            self.dtype = int(distance_type)
            # p1/p2 may be IMP.bff.AV decorators or plain XYZ particles.
            self.d1 = IMP.core.XYZ(p1)
            self.d2 = IMP.core.XYZ(p2)
            self._particles = [
                p.get_particle() if hasattr(p, "get_particle") else p for p in (p1, p2)
            ]
            self.dc = _get_converter(forster_radius, sigma)

        def _score(self, d_mp):
            return float(self.meas.score_model(self.dc(d_mp, self.dtype)))

        def unprotected_evaluate(self, da):
            r = self.d1.get_coordinates() - self.d2.get_coordinates()
            d_mp = r.get_magnitude()
            score = self._score(d_mp)
            if da and d_mp > 1e-7:
                eps = 1e-3
                dsc = (self._score(d_mp + eps) - self._score(d_mp - eps)) / (2.0 * eps)
                f = dsc / d_mp
                grad = IMP.algebra.Vector3D(r[0] * f, r[1] * f, r[2] * f)
                self.d1.add_to_derivatives(grad, da)
                self.d2.add_to_derivatives(grad * -1.0, da)
            return score

        def do_get_inputs(self):
            return self._particles

    class _ScoreLogger(IMP.OptimizerState):
        """Append ``frame,score`` rows during minimisation for live plotting."""

        def __init__(self, m, scoring_function, path):
            IMP.OptimizerState.__init__(self, m, "ScoreLogger")
            self._sf = scoring_function
            self._fh = open(path, "w")
            self._fh.write("frame,score\n")
            self._step = 0

        def do_update(self, call_number):
            self._fh.write(f"{self._step},{self._sf.evaluate(False)}\n")
            self._fh.flush()
            self._step += 1

        def close(self):
            try:
                self._fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Parameters and result containers
# ---------------------------------------------------------------------------


@dataclass
class DockingParameters:
    """Control parameters for Monte-Carlo rigid-body docking.

    Attributes
    ----------
    n_frames : int
        Number of PMI frames (outer Monte-Carlo iterations).
    mc_steps : int
        Monte-Carlo steps per frame.
    mc_temperature : float
        Monte-Carlo temperature (kT).
    max_translation, max_rotation : float
        Per-step rigid-body mover amplitudes (Angstrom, radian).
    simulated_annealing : bool
        Enable the PMI simulated-annealing schedule.
    sa_tmin, sa_tmax : float
        Simulated-annealing temperature bounds.
    n_best : int
        Number of best-scoring PDB models PMI writes.
    ev_weight : float
        Weight of the excluded-volume (clash) restraint.
    shuffle_max_translation : float
        Initial random shuffle amplitude (Angstrom); 0 disables shuffling.
    mean_position_restraint : bool
        Use the fast mean-AV-position FRET restraint (recommended for
        sampling): the distance between the rigid-body-attached AV mean
        positions is converted to the modelled observable
        (<R_DA>/R_DA_E/Rmp) with a Gaussian transfer function of width
        ``sigma_da`` before scoring. When ``False`` the AVs are recomputed on
        every evaluation (accurate but slow).
    sigma_da : float
        Width (Angstrom) of the mean-position transfer function used in
        rigid-body mean-distance scoring.
    score_set : str
        Named score set in the ``fps.json`` "chi2" section ("" = all).
    fixed_body : int
        ``body_id`` kept fixed as the reference (others are mobile).
    """

    n_frames: int = 500
    mc_steps: int = 10
    mc_temperature: float = 1.0
    max_translation: float = 4.0
    max_rotation: float = 0.1
    simulated_annealing: bool = False
    sa_tmin: float = 1.0
    sa_tmax: float = 2.5
    n_best: int = 20
    ev_weight: float = 1.0
    shuffle_max_translation: float = 10.0
    mean_position_restraint: bool = True
    sigma_da: float = 6.0
    score_set: str = ""
    fixed_body: int = 0
    #: Minimisation docking only — clash detection uses one coarse bead per
    #: residue (CA/P/C1') instead of all atoms; ~3x cheaper per CG step and the
    #: dominant cost during docking. Set False for full-atom excluded volume.
    coarse_clash: bool = True
    #: Minimisation docking only — number of FPS-style refinement cycles after
    #: docking: re-sample the AVs in the docked context (inter-body occlusion
    #: shifts the mean positions) and re-minimise. 0 disables refinement
    #: (default — refinement re-runs the one-time AV calc and is opt-in).
    refine_av_cycles: int = 0
    #: Minimisation docking only — after docking, compute and export the full
    #: P(R_DA) distance distributions (real AV convolution) to
    #: ``distance_distributions.csv``. Off by default (recomputes AVs).
    save_distributions: bool = False
    #: AV backend for the (optional) distance-distribution recompute:
    #: "auto" | "labellib" | "imp-bff".
    av_backend: str = "auto"


@dataclass
class PairDistance:
    """A single experimental-vs-model distance comparison (FPS-style diagnostics)."""

    name: str
    position1: str
    position2: str
    distance_exp: float
    distance_model: float
    distance_type: str
    forster_radius: float
    error_neg: float = 0.0
    error_pos: float = 0.0

    @property
    def residual(self) -> float:
        """Model minus experimental distance (Angstrom)."""
        return self.distance_model - self.distance_exp

    @property
    def chi2(self) -> float:
        """Asymmetric chi2 contribution of this pair."""
        d = self.residual
        err = self.error_pos if d > 0 else self.error_neg
        if not err:
            return float("nan")
        return (d / err) ** 2

    @staticmethod
    def _efficiency(distance: float, r0: float) -> float:
        if not r0 or distance != distance:
            return float("nan")
        return 1.0 / (1.0 + (distance / r0) ** 6)

    @property
    def efficiency_model(self) -> float:
        return self._efficiency(self.distance_model, self.forster_radius)

    @property
    def efficiency_exp(self) -> float:
        return self._efficiency(self.distance_exp, self.forster_radius)


@dataclass
class DockingResult:
    """Outcome of a docking / scoring / refinement run."""

    score: float
    n_avs: int
    n_distances: int
    pairs: List[PairDistance] = field(default_factory=list)
    output_dir: Optional[str] = None
    rmf_file: Optional[str] = None
    best_pdbs: List[str] = field(default_factory=list)
    score_csv: Optional[str] = None
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return {
            "score": self.score,
            "n_avs": self.n_avs,
            "n_distances": self.n_distances,
            "pairs": [{**vars(p), "residual": p.residual, "chi2": p.chi2,
                       "E_exp": p.efficiency_exp, "E_model": p.efficiency_model}
                      for p in self.pairs],
            "output_dir": self.output_dir,
            "rmf_file": self.rmf_file,
            "best_pdbs": list(self.best_pdbs),
            "score_csv": self.score_csv,
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def _read_positions(fps_json_path: str) -> Tuple[Dict, Dict, List[str]]:
    """Read positions, distances and score-set names from an fps.json file.

    Returns
    -------
    positions, distances : dict
    score_sets : list of str
    """
    with open(fps_json_path) as fh:
        payload = json.load(fh)
    positions = payload.get("Positions", {})
    distances = payload.get("Distances", {})
    score_sets = list(payload.get("χ²", payload.get("chi2", {})).keys())
    return positions, distances, score_sets


@dataclass
class _Assembly:
    """Internal container bundling the assembled IMP model."""

    model: "IMP.Model"
    root: "IMP.atom.Hierarchy"
    rigid_bodies: Dict[int, "IMP.core.RigidBody"]
    fret: "IMP.bff.restraints.AVNetworkRestraintWrapper"
    restraint_set: "IMP.RestraintSet"
    scoring_function: "IMP.core.RestraintsScoringFunction"
    body_of_pdb: List[int]
    mean_position: bool = True
    sigma_da: float = 6.0


def _body_for_pdb_index(idx: int, n_pdb: int, positions: Dict) -> int:
    """Map a PDB index to a ``body_id``.

    When the number of distinct ``body_id`` values equals the number of PDB
    files we assume a one-to-one correspondence in declaration order;
    otherwise PDB ``idx`` maps to body ``idx``.
    """
    body_ids = sorted({int(p.get("body_id", 0)) for p in positions.values()})
    if len(body_ids) == n_pdb:
        return body_ids[idx]
    return idx


def build_assembly(
    pdb_paths: Sequence[str],
    fps_json_path: str,
    *,
    score_set: str = "",
    mean_position_restraint: bool = True,
    ev_weight: float = 1.0,
    sigma_da: float = 6.0,
) -> _Assembly:
    """Assemble the IMP model: rigid bodies + FRET restraint + excluded volume.

    Parameters
    ----------
    pdb_paths : sequence of str
        One or more PDB files. Each becomes a rigid body (keyed by ``body_id``).
    fps_json_path : str
        Path to the fps.json labelling/distance file.
    score_set : str
        Named score set (``""`` uses the default / all distances).
    mean_position_restraint : bool
        Fast mean-AV-position FRET restraint (recommended for sampling).
    ev_weight : float
        Excluded-volume restraint weight.

    Returns
    -------
    _Assembly
    """
    require_imp()
    if not pdb_paths:
        raise ValueError("At least one PDB file is required.")
    for p in pdb_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"PDB file not found: {p}")
    if not os.path.exists(fps_json_path):
        raise FileNotFoundError(f"fps.json not found: {fps_json_path}")

    # Accept native fps.json or the legacy C# LPs/Distances .txt (auto-convert).
    fps_json_path = ensure_fps_json(fps_json_path, pdb_paths)

    positions, _distances, _score_sets = _read_positions(fps_json_path)

    # Chains referenced by each body (used to prune colliding chains when
    # several PDBs are loaded into one hierarchy).
    body_chains: Dict[int, set] = {}
    for p in positions.values():
        b = int(p.get("body_id", 0))
        ch = p.get("chain_identifier")
        if ch:
            body_chains.setdefault(b, set()).add(str(ch))

    model = IMP.Model()
    root = IMP.atom.Hierarchy.setup_particle(IMP.Particle(model, "root"))

    multi = len(pdb_paths) > 1
    rigid_bodies: Dict[int, "IMP.core.RigidBody"] = {}
    body_of_pdb: List[int] = []
    for idx, pdb in enumerate(pdb_paths):
        body_id = _body_for_pdb_index(idx, len(pdb_paths), positions)
        sel = IMP.atom.NonWaterNonHydrogenPDBSelector()
        # With multiple PDBs, keep only the chains this body actually uses so
        # duplicate chain IDs across files do not collide during label lookup.
        chains = body_chains.get(body_id)
        if multi and chains:
            sel = IMP.atom.AndPDBSelector(sel, IMP.atom.ChainPDBSelector(sorted(chains)))
        h = IMP.atom.read_pdb(str(pdb), model, sel)
        root.add_child(h)
        body_of_pdb.append(body_id)
        leaves = IMP.atom.get_leaves(h)
        if not leaves:
            continue
        rb = IMP.atom.create_rigid_body(h)
        rb.set_name(f"body_{body_id}")
        rigid_bodies[body_id] = rb

    model.update()

    # FRET restraint (PMI wrapper around IMP.bff.AVNetworkRestraint). Built
    # after the rigid bodies exist so the AVs attach to their bodies.
    fret = IMP.bff.restraints.AVNetworkRestraintWrapper(
        root,
        fps_json_path,
        mean_position_restraint=mean_position_restraint,
        sigma_DA=sigma_da,
        occupy_volume=True,
        score_set=score_set,
    )
    # add_to_rmf=False: the bff AVNetworkRestraint's RMF RestraintInfo director
    # is incompatible with IMP.rmf.add_restraints; scores are still logged to
    # the PMI stat file via output_objects.
    fret.add_to_model(add_to_rmf=False)

    # Excluded volume (clash) over all structured leaves; IMP is rigid-body aware.
    all_leaves = IMP.atom.get_leaves(root)
    lsc = IMP.container.ListSingletonContainer(model, all_leaves)
    evr = IMP.core.ExcludedVolumeRestraint(lsc, 1.0, 10.0)
    evr.set_name("excluded_volume")
    evr.set_weight(ev_weight)
    # Register with the model so the PMI sampler's scoring function includes it.
    IMP.pmi.tools.add_restraint_to_model(model, evr, add_to_rmf=False)

    # Restraint set + scoring function for direct (reporting) evaluation.
    restraint_set = IMP.RestraintSet(model, "docking")
    restraint_set.add_restraint(fret.rs)
    restraint_set.add_restraint(evr)
    sf = IMP.core.RestraintsScoringFunction([restraint_set])

    return _Assembly(
        model=model,
        root=root,
        rigid_bodies=rigid_bodies,
        fret=fret,
        restraint_set=restraint_set,
        scoring_function=sf,
        body_of_pdb=body_of_pdb,
        mean_position=mean_position_restraint,
        sigma_da=sigma_da,
    )


# ---------------------------------------------------------------------------
# Distance reporting
# ---------------------------------------------------------------------------


def _collect_pairs(asm: "_Assembly") -> List[PairDistance]:
    """Extract model-vs-experiment distances from the FRET restraint.

    In mean-position (rigid-body) mode the model distance is obtained from the
    distance between the AV mean positions converted to the modelled observable
    with the Gaussian transfer function (``IMP.bff.tools.FRETDistanceConverter``).
    In full-AV mode the restraint's own ``get_model_distance`` is used.
    """
    pairs: List[PairDistance] = []
    fret = asm.fret
    rst = fret.av_network_restraint
    used = rst.get_used_distances()
    avs_by_name = {av.get_name(): av for av in rst.get_used_avs()}
    for key in used:
        e = used[key]
        d = json.loads(e.get_json())
        fr = float(d.get("Forster_radius", 52.0))
        dtype_int = int(getattr(e, "distance_type", IMP.bff.DYE_PAIR_DISTANCE_MEAN))
        model_d = float("nan")
        if asm.mean_position:
            av1 = avs_by_name.get(d["position1_name"])
            av2 = avs_by_name.get(d["position2_name"])
            if av1 is not None and av2 is not None:
                try:
                    r = IMP.core.XYZ(av1).get_coordinates() - IMP.core.XYZ(av2).get_coordinates()
                    d_mp = r.get_magnitude()
                    dc = _get_converter(fr, asm.sigma_da)
                    model_d = float(dc(d_mp, dtype_int))
                except Exception:
                    model_d = float("nan")
        else:
            try:
                model_d = float(rst.get_model_distance(
                    d["position1_name"], d["position2_name"], fr, dtype_int,
                ))
            except Exception:
                model_d = float("nan")
        pairs.append(
            PairDistance(
                name=key,
                position1=d.get("position1_name", ""),
                position2=d.get("position2_name", ""),
                distance_exp=float(d.get("distance", 0.0)),
                distance_model=float(model_d),
                distance_type=str(d.get("distance_type", "RDAMean")),
                forster_radius=float(d.get("Forster_radius", 52.0)),
                error_neg=float(d.get("error_neg", 0.0)),
                error_pos=float(d.get("error_pos", 0.0)),
            )
        )
    return pairs


def _write_score_csv(path: str, score: float, pairs: Sequence[PairDistance]) -> None:
    """Write a CSV of the total score and per-pair model/experiment distances."""
    import csv

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# total_score", score])
        w.writerow(["name", "position1", "position2", "distance_type",
                    "forster_radius", "distance_exp", "distance_model",
                    "residual", "chi2", "E_exp", "E_model"])
        for p in pairs:
            w.writerow([p.name, p.position1, p.position2, p.distance_type,
                        p.forster_radius, p.distance_exp, p.distance_model,
                        round(p.residual, 3), round(p.chi2, 3),
                        round(p.efficiency_exp, 4), round(p.efficiency_model, 4)])


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------


def score(
    pdb_paths: Sequence[str],
    fps_json_path: str,
    *,
    score_set: str = "",
    mean_position_restraint: bool = False,
    sigma_da: float = 6.0,
    output_csv: Optional[str] = None,
) -> DockingResult:
    """Score one structure (set of PDBs) against the FRET restraints.

    Uses the accurate (AV-recomputing) restraint by default.

    Parameters
    ----------
    pdb_paths : sequence of str
    fps_json_path : str
    score_set : str
    mean_position_restraint : bool
        Use the fast mean-position approximation instead of full AV recompute.
    output_csv : str, optional
        If given, write the per-pair distance table to this CSV.

    Returns
    -------
    DockingResult
    """
    require_imp()
    asm = build_assembly(
        pdb_paths, fps_json_path,
        score_set=score_set, mean_position_restraint=mean_position_restraint,
        sigma_da=sigma_da,
    )
    total = float(asm.scoring_function.evaluate(False))
    pairs = _collect_pairs(asm)
    if output_csv:
        _write_score_csv(output_csv, total, pairs)
    return DockingResult(
        score=total,
        n_avs=len(asm.fret.av_network_restraint.get_used_avs()),
        n_distances=len(pairs),
        pairs=pairs,
        score_csv=output_csv,
    )


def dock(
    pdb_paths: Sequence[str],
    fps_json_path: str,
    output_dir: str,
    params: Optional[DockingParameters] = None,
) -> DockingResult:
    """Run FRET-restrained Monte-Carlo rigid-body docking.

    Mobile rigid bodies are sampled with the PMI ``ReplicaExchange`` macro,
    which writes an RMF trajectory and the best-scoring PDB models into
    ``output_dir``.

    Parameters
    ----------
    pdb_paths : sequence of str
        One PDB per rigid body to dock.
    fps_json_path : str
    output_dir : str
        Directory for RMF / PDB / CSV output (created if missing).
    params : DockingParameters, optional

    Returns
    -------
    DockingResult
    """
    require_imp()
    params = params or DockingParameters()
    _ensure_output_dir(output_dir)

    asm = build_assembly(
        pdb_paths, fps_json_path,
        score_set=params.score_set,
        mean_position_restraint=params.mean_position_restraint,
        ev_weight=params.ev_weight,
        sigma_da=params.sigma_da,
    )
    model, root = asm.model, asm.root

    # Movers for every mobile rigid body (the fixed body anchors the frame).
    movers = []
    for body_id, rb in asm.rigid_bodies.items():
        if body_id == params.fixed_body:
            continue
        mv = IMP.core.RigidBodyMover(
            model, rb.get_particle_index(),
            params.max_translation, params.max_rotation,
        )
        movers.append(mv)
    if not movers and asm.rigid_bodies:
        # Single body: still allow it to move so docking is meaningful.
        rb = next(iter(asm.rigid_bodies.values()))
        movers.append(IMP.core.RigidBodyMover(
            model, rb.get_particle_index(),
            params.max_translation, params.max_rotation))

    # Initial random shuffle of mobile bodies.
    if params.shuffle_max_translation > 0:
        try:
            IMP.pmi.tools.shuffle_configuration(
                root, max_translation=params.shuffle_max_translation,
            )
        except Exception:
            pass

    out = os.path.join(output_dir, "")
    rex = IMP.pmi.macros.ReplicaExchange(
        model,
        root_hier=root,
        monte_carlo_sample_objects=movers,
        output_objects=[asm.fret],
        monte_carlo_temperature=params.mc_temperature,
        simulated_annealing=params.simulated_annealing,
        simulated_annealing_minimum_temperature=params.sa_tmin,
        simulated_annealing_maximum_temperature=params.sa_tmax,
        monte_carlo_steps=params.mc_steps,
        number_of_best_scoring_models=params.n_best,
        number_of_frames=params.n_frames,
        global_output_directory=out,
        replica_exchange_swap=False,
    )
    rex.execute_macro()

    # Final score of the (in-memory) model after sampling.
    total = float(asm.scoring_function.evaluate(False))
    pairs = _collect_pairs(asm)
    score_csv = os.path.join(output_dir, "scores.csv")
    _write_score_csv(score_csv, total, pairs)

    rmf_file = _find_first(output_dir, (".rmf3", ".rmf"))
    best_pdbs = _find_all(output_dir, (".pdb",))
    return DockingResult(
        score=total,
        n_avs=len(asm.fret.av_network_restraint.get_used_avs()),
        n_distances=len(pairs),
        pairs=pairs,
        output_dir=output_dir,
        rmf_file=rmf_file,
        best_pdbs=best_pdbs,
        score_csv=score_csv,
        extra={"n_frames": params.n_frames, "n_movers": len(movers)},
    )


def dock_minimize(
    pdb_paths: Sequence[str],
    fps_json_path: str,
    output_dir: str,
    params: Optional[DockingParameters] = None,
    stop_check=None,
) -> DockingResult:
    """Dock by FRET-restrained energy *minimisation* (IMP conjugate gradients).

    The fast, deterministic alternative to :func:`dock`'s replica-exchange MC,
    following the FPS approach: AV mean positions ride on the rigid bodies as
    point-member proxies and are pulled toward the experimental distances by
    plain harmonic springs on the point-to-point separation (the transfer
    function is inverted once per pair to set the R_mp target — see
    :func:`_rmp_target_and_weight`), with a coarse excluded-volume clash.
    ``IMP.core.ConjugateGradients`` minimises the combined energy; ``n_frames``
    is reused as the iteration budget. The per-iteration energy is logged to
    ``convergence.csv`` so the GUI can plot the descent live. Optional
    ``refine_av_cycles`` re-sample the AVs in the docked context afterwards.

    Parameters
    ----------
    pdb_paths : sequence of str
    fps_json_path : str
    output_dir : str
    params : DockingParameters, optional

    Returns
    -------
    DockingResult
    """
    require_imp()
    params = params or DockingParameters()
    _ensure_output_dir(output_dir)

    # mean_position_restraint=False keeps IMP.bff from attaching the dye mean
    # positions to the rigid bodies as *body members*: that switches IMP into
    # adjoint-derivative mode, which silently drops the gradients a minimiser
    # needs. We instead attach our own plain XYZ *point-member* proxies (below),
    # whose derivatives propagate to the rigid bodies the classic way.
    asm = build_assembly(
        pdb_paths, fps_json_path,
        score_set=params.score_set,
        mean_position_restraint=False,
        ev_weight=params.ev_weight,
        sigma_da=params.sigma_da,
    )
    model, root = asm.model, asm.root

    # Freeze the reference body; optimise the others.
    mobile = []
    for body_id, rb in asm.rigid_bodies.items():
        opt = body_id != params.fixed_body
        rb.set_coordinates_are_optimized(opt)
        if opt:
            mobile.append(rb)
    if not mobile and asm.rigid_bodies:
        rb = next(iter(asm.rigid_bodies.values()))
        rb.set_coordinates_are_optimized(True)
        mobile.append(rb)

    if params.shuffle_max_translation > 0:
        try:
            IMP.pmi.tools.shuffle_configuration(
                root, max_translation=params.shuffle_max_translation)
        except Exception:
            pass

    rst = asm.fret.av_network_restraint
    used = rst.get_used_distances()
    avs = {av.get_name(): av for av in rst.get_used_avs()}

    # Point-member proxies at the AV mean positions: they ride rigidly with the
    # body (added via the labelling atom's rigid body) and feed gradients back.
    proxies = {}
    for name, av in avs.items():
        av.resample()
        q = IMP.Particle(model, f"avproxy_{name}")
        IMP.core.XYZ.setup_particle(q, IMP.core.XYZ(av).get_coordinates())
        src = av.get_source()
        if IMP.core.RigidBodyMember.get_is_setup(src):
            IMP.core.RigidBodyMember(src).get_rigid_body().add_member(q)
        proxies[name] = q
    model.update()

    # FRET springs on the point-to-point distance between AV-mean proxies: the
    # distance is plain geometry; the cached transfer function converts it to the
    # modelled observable and scores it (asymmetric chi2) every step — cheap, and
    # the same scoring as the Monte-Carlo path.
    rset = IMP.RestraintSet(model, "fret_min")
    meta = []
    for key in used:
        e = used[key]
        d = json.loads(e.get_json())
        q1 = proxies.get(d["position1_name"])
        q2 = proxies.get(d["position2_name"])
        if q1 is None or q2 is None:
            continue
        fr = float(d.get("Forster_radius", 52.0))
        dtype = int(getattr(e, "distance_type", IMP.bff.DYE_PAIR_DISTANCE_MEAN))
        rset.add_restraint(
            MeanDistanceRestraint(model, e, q1, q2, fr, dtype, params.sigma_da))
        meta.append((key, d, q1, q2, fr, dtype))

    clash = _clash_container(model, root, coarse=params.coarse_clash)
    evr = IMP.core.ExcludedVolumeRestraint(clash, 1.0, 10.0)
    evr.set_weight(params.ev_weight)
    sf = IMP.core.RestraintsScoringFunction([rset, evr])

    # Minimise in a single conjugate-gradient run (chunked restarts would reset
    # the conjugate directions and stall), logging the descent via an optimizer
    # state for live plotting.
    conv_csv = os.path.join(output_dir, "convergence.csv")
    cg = IMP.core.ConjugateGradients(model)
    cg.set_scoring_function(sf)
    n_iter = max(1, int(params.n_frames))
    logger = _ScoreLogger(model, sf, conv_csv)
    logger.set_period(max(1, n_iter // 100))
    cg.add_optimizer_state(logger)

    def _optimize(n):
        """Run CG in chunks so a cooperative stop is honoured promptly.

        Cancellation between chunks (plain Python) is reliable; raising from
        inside the IMP optimizer-state callback is not. Chunk ~100 keeps the
        stop responsive while barely affecting the conjugate-gradient descent.
        """
        chunk = min(100, max(20, n // 20))
        done = 0
        while done < n:
            if stop_check is not None and stop_check():
                return True
            cg.optimize(min(chunk, n - done))
            done += chunk
        return False

    # Phase 1 — dock on the isolated-context AV mean positions + transfer fn.
    stopped = _optimize(n_iter)

    # Phase 2 — FPS-style AV refinement: with the partner now docked alongside,
    # re-sample each AV so inter-body occlusion shifts its mean position, move the
    # proxy to the refreshed mean (in the body frame) and re-minimise. The AVs are
    # recomputed only once per cycle, not per CG step.
    for _ in range(max(0, int(params.refine_av_cycles))):
        if stopped:
            break
        for name, av in avs.items():
            av.resample()
            q = proxies[name]
            if IMP.core.RigidBodyMember.get_is_setup(q):
                rbm = IMP.core.RigidBodyMember(q)
                local = (rbm.get_rigid_body().get_reference_frame()
                         .get_transformation_to().get_inverse()
                         .get_transformed(IMP.core.XYZ(av).get_coordinates()))
                rbm.set_internal_coordinates(local)
        model.update()
        stopped = _optimize(n_iter)
    logger.close()

    # The springs already score the asymmetric chi2 through the transfer
    # function, so the scoring function total is directly comparable to dock().
    total = float(sf.evaluate(False))
    pairs = []
    for key, d, q1, q2, fr, dtype in meta:
        dmp = (IMP.core.XYZ(q1).get_coordinates()
               - IMP.core.XYZ(q2).get_coordinates()).get_magnitude()
        model_obs = float(_get_converter(fr, params.sigma_da)(dmp, dtype))
        pairs.append(PairDistance(
            name=key, position1=d.get("position1_name", ""),
            position2=d.get("position2_name", ""),
            distance_exp=float(d.get("distance", 0.0)),
            distance_model=model_obs,
            distance_type=str(d.get("distance_type", "")),
            forster_radius=fr,
            error_neg=float(d.get("error_neg", 0.0)),
            error_pos=float(d.get("error_pos", 0.0)),
        ))

    out_pdb = os.path.join(output_dir, "docked.pdb")
    IMP.atom.write_pdb(root, out_pdb)
    score_csv = os.path.join(output_dir, "scores.csv")
    _write_score_csv(score_csv, total, pairs)

    extra = {"method": "minimize", "iterations": n_iter,
             "n_mobile": len(mobile), "convergence_csv": conv_csv,
             "coarse_clash": bool(params.coarse_clash),
             "refine_av_cycles": int(params.refine_av_cycles),
             "stopped": stopped}
    if params.save_distributions and not stopped:
        try:
            from . import av as _av
            from . import distributions as _distr
            _av.select_backend(params.av_backend)
            positions, dists, _ss = _read_positions(ensure_fps_json(fps_json_path, pdb_paths))
            res = _distr.compute_distance_distributions(
                out_pdb, positions, dists,
                out_csv=os.path.join(output_dir, "distance_distributions.csv"))
            extra["distributions_csv"] = res.get("distributions_csv")
        except Exception:
            pass
    return DockingResult(
        score=total,
        n_avs=len(avs),
        n_distances=len(pairs),
        pairs=pairs,
        output_dir=output_dir,
        best_pdbs=[out_pdb],
        score_csv=score_csv,
        extra=extra,
    )


def refine(
    pdb_paths: Sequence[str],
    fps_json_path: str,
    output_dir: str,
    *,
    score_set: str = "",
    steps: int = 500,
    ev_weight: float = 1.0,
) -> DockingResult:
    """Locally refine a pose with conjugate-gradient minimisation.

    Parameters
    ----------
    pdb_paths : sequence of str
    fps_json_path : str
    output_dir : str
    score_set : str
    steps : int
        Conjugate-gradient steps.
    ev_weight : float

    Returns
    -------
    DockingResult
    """
    require_imp()
    _ensure_output_dir(output_dir)
    asm = build_assembly(
        pdb_paths, fps_json_path,
        score_set=score_set, mean_position_restraint=True, ev_weight=ev_weight,
    )
    cg = IMP.core.ConjugateGradients(asm.model)
    cg.set_scoring_function(asm.scoring_function)
    cg.optimize(steps)

    out_pdb = os.path.join(output_dir, "refined.pdb")
    IMP.atom.write_pdb(asm.root, out_pdb)
    total = float(asm.scoring_function.evaluate(False))
    pairs = _collect_pairs(asm)
    score_csv = os.path.join(output_dir, "scores.csv")
    _write_score_csv(score_csv, total, pairs)
    return DockingResult(
        score=total,
        n_avs=len(asm.fret.av_network_restraint.get_used_avs()),
        n_distances=len(pairs),
        pairs=pairs,
        output_dir=output_dir,
        best_pdbs=[out_pdb],
        score_csv=score_csv,
    )


def screen(
    pdb_inputs: Sequence[str],
    fps_json_path: str,
    *,
    score_set: str = "",
    output_csv: Optional[str] = None,
    mean_position_restraint: bool = False,
) -> List[Tuple[str, float]]:
    """Score a library of structures against the FRET restraints and rank them.

    Parameters
    ----------
    pdb_inputs : sequence of str
        PDB files, or directories that are expanded to their ``*.pdb`` files.
    fps_json_path : str
    score_set : str
    output_csv : str, optional
        Write the ranked ``filename, score`` table here.
    mean_position_restraint : bool

    Returns
    -------
    list of (path, score)
        Sorted ascending by score (best first).
    """
    require_imp()
    pdbs: List[str] = []
    for item in pdb_inputs:
        if os.path.isdir(item):
            pdbs.extend(sorted(
                os.path.join(item, f) for f in os.listdir(item)
                if f.lower().endswith(".pdb")))
        else:
            pdbs.append(item)

    results: List[Tuple[str, float]] = []
    for pdb in pdbs:
        try:
            res = score(
                [pdb], fps_json_path,
                score_set=score_set,
                mean_position_restraint=mean_position_restraint,
            )
            results.append((pdb, res.score))
        except Exception as exc:  # keep screening the rest of the library
            results.append((pdb, float("nan")))
            _ = exc
    results.sort(key=lambda t: (t[1] != t[1], t[1]))  # NaNs last, then ascending

    if output_csv:
        import csv
        with open(output_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pdb", "score"])
            for path, sc in results:
                w.writerow([path, sc])
    return results


def estimate_errors(
    pdb_paths: Sequence[str],
    fps_json_path: str,
    output_dir: str,
    *,
    n_trials: int = 10,
    params: Optional[DockingParameters] = None,
    method: str = "minimize",
    n_workers: Optional[int] = None,
    stop_check=None,
) -> Dict:
    """Repeat docking from independent random starts to estimate spread.

    Trials are independent and run in parallel worker processes by default
    (``n_workers`` = CPU count). Each trial seeds IMP's RNG from its index, so
    results are reproducible and independent of the parallel schedule.

    Parameters
    ----------
    pdb_paths : sequence of str
    fps_json_path : str
    output_dir : str
    n_trials : int
        Number of independent docking trials.
    params : DockingParameters, optional
    method : str
        ``"minimize"`` (conjugate-gradient docking, default) or ``"mc"``
        (replica-exchange Monte-Carlo).
    n_workers : int, optional
        Parallel worker processes. ``None`` -> ``min(n_trials, cpu_count)``;
        ``1`` forces serial. Falls back to serial if a process pool can't start.

    Returns
    -------
    dict
        ``{"scores", "score_mean", "score_std", "n_trials", "trials": [dirs],
        "trial_details": [{trial, score, n_distances, output_dir, best_pdb,
        score_csv, stat_file}], "best_trial", "n_workers"}``.
    """
    require_imp()
    import statistics

    _ensure_output_dir(output_dir)
    params = params or DockingParameters()
    args = [
        (i, os.path.join(output_dir, f"trial_{i:03d}"),
         list(pdb_paths), fps_json_path, params, method)
        for i in range(n_trials)
    ]

    if n_workers is None:
        n_workers = min(int(n_trials), max(1, os.cpu_count() or 1))
    n_workers = max(1, int(n_workers))

    details = None
    if n_workers > 1 and n_trials > 1:
        details = _run_trials_parallel(args, n_workers, stop_check=stop_check)
    used_workers = n_workers if details is not None else 1
    if details is None:  # serial path or parallel fallback
        details = []
        for a in args:
            if stop_check is not None and stop_check():
                break
            details.append(_run_one_trial(*a))

    details.sort(key=lambda d: d["trial"])
    scores = [d["score"] for d in details]
    trial_dirs = [d["output_dir"] for d in details]
    finite = [s for s in scores if s == s]
    best_trial = min(details, key=lambda d: d["score"])["trial"] if finite else None

    # Model precision: superpose the per-run best models on the fixed body and
    # report per-atom RMSF (written to uncertainty.pdb / uncertainty.csv).
    uncertainty = _model_uncertainty(details, pdb_paths, params, output_dir)

    return {
        "scores": scores,
        "score_mean": statistics.fmean(finite) if finite else float("nan"),
        "score_std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "n_trials": n_trials,
        "trials": trial_dirs,
        "trial_details": details,
        "best_trial": best_trial,
        "n_workers": used_workers,
        "uncertainty": uncertainty,
    }


def _model_uncertainty(details, pdb_paths, params, output_dir):
    """Compute FPS-style positional uncertainty across the per-run best models."""
    best_pdbs = [d["best_pdb"] for d in details if d.get("best_pdb")]
    if len(best_pdbs) < 2:
        return None
    try:
        from . import uncertainty as _unc
        fixed_idx = int(params.fixed_body) if int(params.fixed_body) < len(pdb_paths) else 0
        _lines, _xyz, fchains = _unc._read_pdb_atoms(pdb_paths[fixed_idx])
        return _unc.estimate_position_uncertainty(
            best_pdbs, sorted(set(fchains.tolist())),
            out_pdb=os.path.join(output_dir, "uncertainty.pdb"),
            out_csv=os.path.join(output_dir, "uncertainty.csv"))
    except Exception:
        return None


def _run_one_trial(seed, tdir, pdb_paths, fps_json_path, params, method):
    """Run a single docking trial; return a JSON-safe summary dict.

    Module-level and self-contained so it can execute in a forked worker.
    """
    # Seed every RNG the shuffle might draw from so a trial is reproducible and
    # independent of the (serial vs parallel) schedule.
    import random as _random

    import numpy as _np
    s = int(seed) + 1
    _random.seed(s)
    _np.random.seed(s)
    try:
        IMP.random_number_generator.seed(s)
    except Exception:
        pass
    dock_fn = dock_minimize if method == "minimize" else dock
    res = dock_fn(pdb_paths, fps_json_path, tdir, params)
    trace = "convergence.csv" if method == "minimize" else "stat.0.out"
    return {
        "trial": int(seed),
        "score": res.score,
        "n_distances": res.n_distances,
        "output_dir": tdir,
        "best_pdb": res.best_pdbs[0] if res.best_pdbs else None,
        "score_csv": res.score_csv,
        "stat_file": os.path.join(tdir, trace),
    }


def _run_trials_parallel(args, n_workers, stop_check=None):
    """Run trials across forked workers; return ``None`` to fall back to serial.

    Uses the ``fork`` start method so workers inherit the already-imported IMP /
    chisurf modules (``spawn`` would re-import the whole stack per worker and
    erase the speed-up). Polls ``stop_check`` and terminates the pool on cancel,
    returning whatever trials finished. Any failure (no fork, pool error) returns
    ``None``.
    """
    import multiprocessing as mp

    # Children touch only IMP/numpy, never Cocoa — allow fork from a Qt app.
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    try:
        with ctx.Pool(n_workers) as pool:
            async_res = pool.starmap_async(_run_one_trial, args)
            while not async_res.ready():
                if stop_check is not None and stop_check():
                    pool.terminate()
                    return []  # stopped before any trial returned a result
                async_res.wait(0.2)
            return async_res.get()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------


def _find_first(directory: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    """Return the first file under ``directory`` matching any suffix."""
    for dirpath, _dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.lower().endswith(suffixes):
                return os.path.join(dirpath, f)
    return None


def _find_all(directory: str, suffixes: Tuple[str, ...]) -> List[str]:
    """Return all files under ``directory`` matching any suffix."""
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.lower().endswith(suffixes):
                out.append(os.path.join(dirpath, f))
    return out
