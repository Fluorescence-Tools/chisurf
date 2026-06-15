"""Headless ProteinMC model and runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Optional

import numpy as np

import chisurf.core.settings
from chisurf.core.math.rand import mc, weighted_choice
from chisurf.core.structure import ProteinCentroid, Structure, Universe

from .rmf import ProteinMCRmfWriter


ProgressCallback = Callable[["ProteinMCProgress"], None]


@dataclass
class ProteinMCProgress:
    """Progress payload emitted by :class:`ProteinMCRunner`."""

    frame_index: int
    target_frames: int
    iteration: int
    accepted: int
    rejected: int
    energy: float
    labeling_energy: float
    rmsd: list[float] = field(default_factory=list)
    drmsd: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    labeling_energies: list[float] = field(default_factory=list)
    xyz: Optional[np.ndarray] = None
    output_file: Optional[str] = None


@dataclass
class ProteinMCResult:
    """Result returned after a ProteinMC run."""

    output_file: str
    n_frames: int
    accepted: int
    rejected: int
    rmsd: list[float]
    drmsd: list[float]
    energies: list[float]
    labeling_energies: list[float]
    structure: ProteinCentroid | None = None


def load_json(filename: str | Path) -> dict[str, Any]:
    """Load a JSON file and return its object."""
    with open(filename, "r") as fp:
        return json.load(fp)


def normalize_settings(settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return ProteinMC settings with legacy and current key aliases populated."""
    base = dict(chisurf.core.settings.cs_settings.get("mc_settings", {}))
    if settings is None or "potentials" not in settings:
        # Legacy ProteinMC only added potentials after an explicit config load.
        # Keep default CLI/GUI runs focused on the labeling potential.
        base["potentials"] = []
    if settings:
        base.update(settings)
    if "n_out" in base and "pdb_nOut" not in base:
        base["pdb_nOut"] = base["n_out"]
    if "pdb_nOut" in base and "n_out" not in base:
        base["n_out"] = base["pdb_nOut"]
    if "pdbOut" in base and "n_written" not in base:
        base["n_written"] = base["pdbOut"]
    if "n_written" in base and "pdbOut" not in base:
        base["pdbOut"] = base["n_written"]
    if "movemap" in base and "move_map" not in base:
        base["move_map"] = base["movemap"]
    if settings and "movemap" in settings:
        base["move_map"] = settings["movemap"]
    if "move_map" in base and "movemap" not in base:
        base["movemap"] = base["move_map"]
    base.setdefault("mc_mode", "simple")
    base.setdefault("scale", 0.0025)
    base.setdefault("kt", 1.5)
    base.setdefault("n_iter", 10000)
    base.setdefault("pdb_nOut", base.get("n_out", 500))
    base.setdefault("n_written", base.get("pdbOut", 500))
    base.setdefault("pPhi", 0.7)
    base.setdefault("pPsi", 0.3)
    base.setdefault("pOmega", 0.0)
    base.setdefault("pChi", 0.01)
    return base


def load_structure(source: str | Path | ProteinCentroid) -> ProteinCentroid:
    """Load a ProteinCentroid from a path, PDB id, or existing structure."""
    if isinstance(source, ProteinCentroid):
        return source
    source_text = str(source)
    if len(source_text) == 4 and not Path(source_text).exists():
        _patch_rcsb_fetch_url()
    return ProteinCentroid(_pdb2pqr_structure_file(source_text))


def _patch_rcsb_fetch_url() -> None:
    """Patch the legacy RCSB URL to the current HTTPS endpoint."""
    import chisurf.core.fio.structure.coordinates as coordinates

    def fetch_pdb_string(pdb_id: str) -> str:
        import urllib.request

        url = f"https://files.rcsb.org/download/{pdb_id[:4].lower()}.pdb"
        return urllib.request.urlopen(url).read().decode("utf-8")

    coordinates.fetch_pdb_string = fetch_pdb_string


def _pdb2pqr_structure_file(source: str) -> str:
    """Return a pdb2pqr-protonated PQR file."""
    executable = shutil.which("pdb2pqr") or shutil.which("pdb2pqr30")
    if executable is None:
        raise RuntimeError(
            "ProteinMC requires pdb2pqr for structure preparation. "
            "Install the conda-forge pdb2pqr package."
        )
    if len(source) != 4 and not Path(source).exists():
        raise FileNotFoundError(source)
    input_file = _standard_residue_pdb(source)
    handle, output_file = tempfile.mkstemp(suffix="_proteinmc.pqr")
    import os

    os.close(handle)
    cmd = [executable, "--ff=PARSE", input_file, output_file]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        Path(output_file).unlink(missing_ok=True)
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"pdb2pqr failed while preparing ProteinMC input: {message}") from exc
    finally:
        Path(input_file).unlink(missing_ok=True)
    return output_file


def _standard_residue_pdb(source: str) -> str:
    """Write a temporary PDB containing only standard amino-acid residues."""
    structure = Structure(source)
    atoms = structure.atoms
    atoms = atoms[np.array([_as_text(v) in _STANDARD_RESIDUES for v in atoms["res_name"]])]
    keep_masks = []
    names = atoms.dtype.names or ()
    for chain, res_id in _residue_keys(atoms):
        residue_mask = atoms["res_id"] == res_id
        if "chain" in names:
            residue_mask &= np.array([_as_text(v) == chain for v in atoms["chain"]])
        residue_atoms = atoms[residue_mask]
        atom_names = {_as_text(v) for v in residue_atoms["atom_name"]}
        res_name = _as_text(residue_atoms[0]["res_name"])
        required = {"N", "CA", "C", "O"}
        if res_name != "GLY":
            required.add("CB")
        if required.issubset(atom_names):
            keep_masks.append(residue_mask)
    if not keep_masks:
        raise ValueError("No complete standard amino-acid residues found in structure")
    atoms = atoms[np.logical_or.reduce(keep_masks)]
    structure.atoms = atoms
    handle, filename = tempfile.mkstemp(suffix="_proteinmc_standard.pdb")
    import os

    os.close(handle)
    structure.write(filename)
    return filename


def _residue_keys(atoms: np.ndarray) -> list[tuple[str, int]]:
    """Return residue keys in atom-table order."""
    keys = []
    seen = set()
    for atom in atoms:
        chain = _as_text(atom["chain"]) if "chain" in atoms.dtype.names else "A"
        key = (chain, int(atom["res_id"]))
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _residue_index_map(structure: ProteinCentroid) -> dict[tuple[str, int], int]:
    """Map ``(chain_identifier, residue_seq_number)`` to a 0-based residue index.

    The mapping follows the same iteration order as
    :func:`make_residue_lookup_table` so that the returned index corresponds
    to the residue's position in the ``move_map`` array and the
    ``l_residue`` lookup table.
    """
    atoms = structure.atoms
    res_dict = structure.residue_dict
    idx_map: dict[tuple[str, int], int] = {}
    for n, res_id in enumerate(res_dict.keys()):
        mask = atoms["res_id"] == res_id
        if not mask.any():
            continue
        chain = _as_text(atoms[mask][0]["chain"]) if "chain" in atoms.dtype.names else "A"
        idx_map[(chain, int(res_id))] = n
    return idx_map


def build_move_map_from_flexfit(
    structure: ProteinCentroid,
    labeling_file: str | Path,
    flexfit_set: str | None = None,
) -> np.ndarray | None:
    """Build a per-residue move probability map from a FlexFit set.

    Reads the ``"FlexFit"`` section of an FPS JSON file and returns an
    array of length ``structure.n_residues`` where flexible residues from
    the selected set receive weight ``1.0`` and all other residues
    receive ``0.0``.

    Parameters
    ----------
    structure : ProteinCentroid
        The protein structure whose residue ordering defines the
        ``move_map`` indices.
    labeling_file : str | Path
        Path to the FPS JSON labeling file.
    flexfit_set : str or None
        Name of the FlexFit set to use.  When ``None``, the first set
        found in the file is used.

    Returns
    -------
    numpy.ndarray or None
        A 1-D array of length ``n_residues`` with move weights, or
        ``None`` when no FlexFit data is available or no residues
        matched the structure.
    """
    payload = load_json(labeling_file)
    flexfit = payload.get("FlexFit", {}) or {}
    if not isinstance(flexfit, dict) or not flexfit:
        return None

    if flexfit_set is None:
        flexfit_set = next(iter(flexfit), None)
    if not flexfit_set or flexfit_set not in flexfit:
        return None

    entry = flexfit.get(flexfit_set, {})
    if not isinstance(entry, dict):
        return None
    residues = entry.get("Flexible residues", []) or []
    if not residues:
        return None

    idx_map = _residue_index_map(structure)
    move_map = np.zeros(structure.n_residues, dtype=np.float64)

    for r in residues:
        if not isinstance(r, dict):
            continue
        chain = str(r.get("chain_identifier", "")).strip()
        res_num = int(r.get("residue_seq_number", 0))
        idx = idx_map.get((chain, res_num))
        if idx is not None and 0 <= idx < structure.n_residues:
            move_map[idx] = 1.0

    if move_map.sum() == 0.0:
        return None
    return move_map


def list_flexfit_sets(labeling_file: str | Path) -> list[str]:
    """Return the names of FlexFit sets in an FPS JSON file."""
    try:
        payload = load_json(labeling_file)
        flexfit = payload.get("FlexFit", {}) or {}
        if isinstance(flexfit, dict):
            return list(flexfit.keys())
    except Exception:
        pass
    return []


class DirectLabelingPotential:
    """Fast FPS JSON potential based on direct attachment-atom distances."""

    name = "Labeling"

    def __init__(
        self,
        structure: ProteinCentroid,
        labeling_file: str | Path,
        score_set: str = "",
    ) -> None:
        """Create a direct-distance potential from an FPS JSON file.

        Parameters
        ----------
        structure : ProteinCentroid
            Protein structure to evaluate the potential against.
        labeling_file : str | Path
            Path to the fps.json labeling file.
        score_set : str, optional
            Name of the χ² score group to use. When empty or when the
            score set is not found in the file, all distances are used.
        """
        self.structure = structure
        self.labeling_file = str(labeling_file)
        payload = load_json(labeling_file)
        self.all_positions = payload["Positions"]
        all_distances = payload["Distances"]

        # Filter distances by score set
        self.score_set = score_set
        if score_set and "χ²" in payload and score_set in payload["χ²"]:
            group = payload["χ²"][score_set]
            keys = group.get("distances", []) if isinstance(group, dict) else []
            self.distances = {k: all_distances[k] for k in keys if k in all_distances}
        else:
            self.distances = dict(all_distances)

        # Only keep positions referenced by the filtered distances
        used_names: set[str] = set()
        for d in self.distances.values():
            used_names.add(d.get("position1_name", ""))
            used_names.add(d.get("position2_name", ""))
        self.positions = {
            name: self.all_positions[name]
            for name in used_names
            if name in self.all_positions
        }

        self._position_indices = {
            name: self._resolve_position_index(position)
            for name, position in self.positions.items()
        }
        self.last_energy = np.inf

    def _resolve_position_index(self, position: dict[str, Any]) -> int:
        """Resolve a labeling position to an atom index in the current structure."""
        atoms = self.structure.atoms
        if "chain_identifier" in position and "residue_seq_number" in position:
            chain = str(position["chain_identifier"])
            residue = int(position["residue_seq_number"])
            atom_name = str(position.get("atom_name", "CA"))
            mask = np.ones(len(atoms), dtype=bool)
            if "chain" in atoms.dtype.names:
                mask &= np.array([_as_text(v) == chain for v in atoms["chain"]])
            if "res_id" in atoms.dtype.names:
                mask &= atoms["res_id"] == residue
            if "atom_name" in atoms.dtype.names:
                atom_mask = np.array([_as_text(v) == atom_name for v in atoms["atom_name"]])
                fallback_ca = np.array([_as_text(v) == "CA" for v in atoms["atom_name"]])
                selected = np.where(mask & atom_mask)[0]
                if selected.size == 0:
                    selected = np.where(mask & fallback_ca)[0]
            else:
                selected = np.where(mask)[0]
            if selected.size:
                return int(selected[0])
        if "attachment_atom_index" in position:
            index = int(position["attachment_atom_index"])
            if 0 <= index < len(atoms):
                return index
        raise ValueError(f"Cannot resolve labeling position: {position!r}")

    def getEnergy(self) -> float:
        """Return chi-square-like direct-distance energy."""
        xyz = self.structure.xyz
        chi2 = 0.0
        for distance in self.distances.values():
            p1 = self._position_indices[distance["position1_name"]]
            p2 = self._position_indices[distance["position2_name"]]
            model_distance = float(np.linalg.norm(xyz[p1] - xyz[p2]))
            target = float(distance.get("distance", model_distance))
            error_neg = max(float(distance.get("error_neg", 1.0)), 1e-12)
            error_pos = max(float(distance.get("error_pos", 1.0)), 1e-12)
            delta = model_distance - target
            chi2 += (delta / (error_neg if delta < 0 else error_pos)) ** 2
        self.last_energy = float(chi2)
        return self.last_energy


class ProteinMCRunner:
    """Headless ProteinMC Monte Carlo runner."""

    def __init__(
        self,
        structure_source: str | Path | ProteinCentroid,
        *,
        flexfit_set: str | None = None,
        settings: Optional[dict[str, Any]] = None,
        settings_file: str | Path | None = None,
        output_file: str | Path | None = None,
        initial_frames: Optional[list[np.ndarray]] = None,
        progress_callback: ProgressCallback | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the ProteinMC runner.

        Parameters
        ----------
        structure_source : str | Path | ProteinCentroid
        flexfit_set : str or None
            Name of the FlexFit set used to derive the per-residue move
            map.  When ``None`` and a FPS potential with a labeling file is
            configured in *settings*, the first FlexFit set is used automatically.
            An explicit ``move_map`` entry in *settings* takes
            precedence over this parameter.
        settings : dict, optional
        settings_file : str | Path, optional
        output_file : str | Path, optional
        initial_frames : list of ndarray, optional
        progress_callback : ProgressCallback, optional
        verbose : bool
        """
        if settings_file is not None:
            loaded = load_json(settings_file)
            settings = dict(loaded, **(settings or {}))
        self.settings = normalize_settings(settings)
        self.structure = load_structure(structure_source)
        self.flexfit_set = flexfit_set
        self.output_file = str(output_file or _temporary_rmf_filename())
        self.initial_frames = [np.asarray(frame, dtype=float) for frame in (initial_frames or [])]
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.exiting = False
        self.universe = Universe()
        self.rmsd: list[float] = []
        self.drmsd: list[float] = []
        self.energies: list[float] = []
        self.labeling_energies: list[float] = []
        self._reference_xyz: Optional[np.ndarray] = None
        self._previous_xyz: Optional[np.ndarray] = None
        self._eval_intervals: list[int] = []
        self._last_energies: list[float] = []
        self._labeling_potential: DirectLabelingPotential | None = None
        self._configure_potentials()
        self._last_energies = [0.0] * len(self.universe.potentials)
        fps_file = self._find_fps_file()
        if "move_map" not in self.settings and fps_file:
            derived = build_move_map_from_flexfit(
                self.structure, fps_file, flexfit_set
            )
            if derived is not None:
                self.settings["move_map"] = derived

    @property
    def move_map(self) -> np.ndarray:
        """Residue move probability map."""
        move_map = self.settings.get("move_map")
        if move_map is None:
            return np.ones(int(self.structure.n_residues), dtype=np.float64)
        return np.asarray(move_map, dtype=np.float64)

    def stop(self) -> None:
        """Request the MC loop to stop."""
        self.exiting = True

    def _find_fps_file(self) -> str | None:
        """Return the labeling file path from the dye potential, if any."""
        for ps in self.settings.get("potentials", []) or []:
            name = ps.get("name", "")
            if name in ("fps", "dye"):
                return ps.get("settings", {}).get("labeling_file") or None
        return None

    def _configure_potentials(self) -> None:
        """Add configured structural and labeling potentials."""
        for potential_settings in self.settings.get("potentials", []) or []:
            name = potential_settings.get("name")
            weight = potential_settings.get("weight", 1.0)
            interval = max(1, int(potential_settings.get("eval_interval", 1)))
            kwargs = dict(potential_settings.get("settings", {}) or {})
            if name in ("fps", "dye"):
                labeling_file = kwargs.pop("labeling_file", None)
                score_set = kwargs.pop("score_set", "")
                if labeling_file:
                    potential = DirectLabelingPotential(
                        self.structure, str(labeling_file), score_set
                    )
                    self.universe.addPotential(potential, weight)
                    self._eval_intervals.append(interval)
                    self._labeling_potential = potential
                continue
            potential_cls = _potential_classes().get(name)
            if potential_cls is None:
                continue
            potential = potential_cls(structure=self.structure, **kwargs)
            self.universe.addPotential(potential, weight)
            self._eval_intervals.append(interval)

    def run(self) -> ProteinMCResult:
        """Run ProteinMC and write accepted frames to RMF."""
        if self.settings.get("mc_mode", "simple") != "simple":
            raise ValueError("Only ProteinMC mc_mode='simple' is currently supported.")
        writer = ProteinMCRmfWriter(self.output_file, self.structure)
        for frame_index, frame in enumerate(self.initial_frames):
            writer.append(frame, name=str(frame_index))
        self.structure.auto_update = False
        self.structure.update()
        self._reference_xyz = np.array(self.structure.xyz, copy=True)
        self._previous_xyz = np.array(self.structure.xyz, copy=True)

        energy = self._total_energy(iteration=0)
        self._append_frame(writer, energy, self._labeling_energy(), iteration=0, accepted=0, rejected=0)

        move_map = np.asarray(self.move_map, dtype=np.float64)
        n_iter = int(self.settings["n_iter"])
        n_out = max(int(self.settings["pdb_nOut"]), 1)
        n_write = max(int(self.settings["n_written"]), 0)
        kt = float(self.settings["kt"])
        scale = float(self.settings["scale"])
        p_phi = float(self.settings["pPhi"])
        p_psi = float(self.settings["pPsi"])
        p_omega = float(self.settings["pOmega"])
        p_chi = float(self.settings["pChi"])

        c_phi = np.empty_like(self.structure.phi)
        c_psi = np.empty_like(self.structure.psi)
        c_omega = np.empty_like(self.structure.omega)
        c_chi = np.empty_like(self.structure.chi)
        n_chi = max(c_chi.shape[0], 1)
        coord_back = np.empty_like(self.structure.internal_coordinates)
        np.copyto(coord_back, self.structure.internal_coordinates)

        accepted = 0
        rejected = 0
        written = 0
        iteration = 0
        while iteration < n_iter and written < n_write and not self.exiting:
            iteration += 1
            if iteration % 100 == 0:
                time.sleep(0)
            move_phi, move_psi, move_omega, move_chi = (
                np.random.ranf(4) < [p_phi, p_psi, p_omega, p_chi]
            )
            moving_aa = int(np.asarray(weighted_choice(move_map, 1)).flat[0])
            if move_phi and c_phi.size:
                c_phi *= 0.0
                c_phi[moving_aa % c_phi.size] += (np.random.ranf() - 0.5) * scale
                self.structure.phi = self.structure.phi + c_phi
            if move_psi and c_psi.size:
                c_psi *= 0.0
                c_psi[moving_aa % c_psi.size] += (np.random.ranf() - 0.5) * scale
                self.structure.psi = self.structure.psi + c_psi
            if move_omega and c_omega.size:
                c_omega *= 0.0
                c_omega[moving_aa % c_omega.size] += (np.random.ranf() - 0.5) * scale
                self.structure.omega = self.structure.omega + c_omega
            if move_chi and c_chi.size:
                c_chi *= 0.0
                c_chi[moving_aa % n_chi] += (np.random.ranf() - 0.5) * scale
                self.structure.chi = self.structure.chi + c_chi

            self.structure.update()
            new_energy = self._total_energy(iteration=iteration)
            if mc(energy, new_energy, kt):
                energy = new_energy
                accepted += 1
                if accepted % n_out == 0:
                    written += 1
                    self._append_frame(
                        writer,
                        energy,
                        self._labeling_energy(),
                        iteration=iteration,
                        accepted=accepted,
                        rejected=rejected,
                    )
                np.copyto(coord_back, self.structure.internal_coordinates)
            else:
                rejected += 1
                np.copyto(self.structure.internal_coordinates, coord_back)
                self.structure.update()

        try:
            return ProteinMCResult(
                output_file=self.output_file,
                n_frames=len(self.energies),
                accepted=accepted,
                rejected=rejected,
                rmsd=list(self.rmsd),
                drmsd=list(self.drmsd),
                energies=list(self.energies),
                labeling_energies=list(self.labeling_energies),
                structure=self.structure,
            )
        finally:
            writer.close()

    def _append_frame(
        self,
        writer: ProteinMCRmfWriter,
        energy: float,
        labeling_energy: float,
        *,
        iteration: int,
        accepted: int,
        rejected: int,
    ) -> None:
        """Append one frame and emit progress."""
        xyz = np.array(self.structure.xyz, copy=True)
        if self._reference_xyz is None:
            self._reference_xyz = np.array(xyz, copy=True)
        if self._previous_xyz is None:
            self._previous_xyz = np.array(xyz, copy=True)
        rmsd = _rmsd(xyz, self._reference_xyz)
        drmsd = _rmsd(xyz, self._previous_xyz)
        metadata = {
            "Total_Score": float(energy),
            "ProteinMC_Energy": float(energy),
            "ProteinMC_Labeling_Energy": float(labeling_energy),
            "ProteinMC_RMSD": float(rmsd),
            "ProteinMC_dRMSD": float(drmsd),
            "ProteinMC_Iteration": int(iteration),
            "ProteinMC_Accepted": int(accepted),
            "ProteinMC_Rejected": int(rejected),
        }
        writer.append(xyz, name=str(len(self.energies)), metadata=metadata)
        self.rmsd.append(rmsd)
        self.drmsd.append(drmsd)
        self.energies.append(float(energy))
        self.labeling_energies.append(float(labeling_energy))
        self._previous_xyz = np.array(xyz, copy=True)
        self._emit_progress(iteration, accepted, rejected, xyz)

    def _total_energy(self, iteration: int = 0) -> float:
        """Return total weighted energy, optionally skipping evaluations.

        Potentials whose ``eval_interval`` is >1 are only re-evaluated
        every *interval* iterations; the last computed value is reused
        on intermediate steps.
        """
        total = 0.0
        for i, (potential, scale) in enumerate(
            zip(self.universe.potentials, self.universe.scaling)
        ):
            interval = self._eval_intervals[i]
            if interval <= 1 or iteration % interval == 0:
                potential.structure = self.structure
                self._last_energies[i] = float(potential.getEnergy())
            total += scale * self._last_energies[i]
        return float(total)

    def _labeling_energy(self) -> float:
        """Return the latest labeling potential energy."""
        if self._labeling_potential is None:
            return 0.0
        return float(self._labeling_potential.last_energy)

    def _emit_progress(
        self,
        iteration: int,
        accepted: int,
        rejected: int,
        xyz: np.ndarray,
    ) -> None:
        """Emit a progress callback if one was configured."""
        if self.progress_callback is None:
            return
        self.progress_callback(
            ProteinMCProgress(
                frame_index=len(self.energies),
                target_frames=int(self.settings["n_written"]) + 1,
                iteration=iteration,
                accepted=accepted,
                rejected=rejected,
                energy=self.energies[-1],
                labeling_energy=self.labeling_energies[-1],
                rmsd=list(self.rmsd),
                drmsd=list(self.drmsd),
                energies=list(self.energies),
                labeling_energies=list(self.labeling_energies),
                xyz=np.array(xyz, copy=True),
                output_file=self.output_file,
            )
        )


def run_protein_mc(
    pdb_file: str,
    *,
    settings_file: Optional[str] = None,
    labeling_file: Optional[str] = None,
    score_set: str = "",
    flexfit_set: str | None = None,
    output_file: Optional[str] = None,
    verbose: bool = False,
    scale: Optional[float] = None,
    n_iter: Optional[int] = None,
    n_out: Optional[int] = None,
    n_written: Optional[int] = None,
    eval_interval: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Run ProteinMC and return the generated RMF filename."""
    overrides: dict[str, Any] = {}
    if scale is not None:
        overrides["scale"] = scale
    if n_iter is not None:
        overrides["n_iter"] = int(n_iter)
    if n_out is not None:
        overrides["n_out"] = int(n_out)
    if n_written is not None:
        overrides["pdbOut"] = int(n_written)
    if labeling_file:
        fps = {
            "name": "dye",
            "weight": 1.0,
            "eval_interval": max(1, int(eval_interval)),
            "settings": {
                "labeling_file": str(labeling_file),
                "score_set": score_set or "",
            },
        }
        existing = overrides.get("potentials", [])
        overrides["potentials"] = existing + [fps]
    runner = ProteinMCRunner(
        pdb_file,
        flexfit_set=flexfit_set,
        settings=overrides,
        settings_file=settings_file,
        output_file=output_file,
        progress_callback=progress_callback,
        verbose=verbose,
    )
    return runner.run().output_file


def _temporary_rmf_filename() -> str:
    """Return a writable temporary RMF3 filename."""
    handle, filename = tempfile.mkstemp(suffix=".rmf3")
    Path(filename).unlink(missing_ok=True)
    import os

    os.close(handle)
    return filename


def _potential_classes() -> dict[str, Any]:
    """Return structural potential classes, importing compiled code lazily."""
    from chisurf.core.structure.potential import potentials as core_potentials

    return {
        "default": core_potentials.ClashPotential,
        "hbond": core_potentials.HPotential,
        "mj": core_potentials.MJPotential,
        "unres": core_potentials.CEPotential,
        "go": core_potentials.GoPotential,
        "rama": core_potentials.Ramachandran,
        "asa": core_potentials.ASA,
        "rg": core_potentials.RadiusGyration,
        "H-Bond": core_potentials.HPotential,
        "H-Potential": core_potentials.HPotential,
        "Iso-UNRES": core_potentials.CEPotential,
        "Miyazawa-Jernigan": core_potentials.MJPotential,
        "Clash potential": core_potentials.ClashPotential,
        "Go-Potential": core_potentials.GoPotential,
        "Ramachandran": core_potentials.Ramachandran,
        "ASA-Calpha": core_potentials.ASA,
        "Radius of Gyration": core_potentials.RadiusGyration,
    }


def _rmsd(xyz: np.ndarray, reference: np.ndarray) -> float:
    """Return coordinate RMSD without alignment."""
    delta = np.asarray(xyz, dtype=float) - np.asarray(reference, dtype=float)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _as_text(value) -> str:
    """Convert numpy string scalar values to plain text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
