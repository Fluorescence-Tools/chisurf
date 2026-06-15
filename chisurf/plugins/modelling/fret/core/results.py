from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from chisurf.core.fio.structure.coordinates import formats, keys
from chisurf.core.structure.structure import Structure

from . import io


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a single docking simulation."""
    converged: bool = False
    iterations: int = 0
    energy: float = 0.0
    clash_energy: float = 0.0
    restraint_energy: float = 0.0
    translations: List[np.ndarray] = field(default_factory=list)
    rotations: List[np.ndarray] = field(default_factory=list)
    model_distances: Dict[str, float] = field(default_factory=dict)
    rmsd: float = 0.0
    internal_number: int = 0
    force_norms: List[float] = field(default_factory=list)
    torque_norms: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Screening result
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    """Result of screening one structure against FRET restraints."""
    filename: str = ""
    chi2: float = 0.0
    reduced_chi2: float = 0.0
    n_distances: int = 0
    n_valid: int = 0
    n_violations_1sigma: int = 0
    n_violations_2sigma: int = 0
    n_violations_3sigma: int = 0
    n_nan: int = 0
    ref_rmsd: float = 0.0
    model_distances: Dict[str, float] = field(default_factory=dict)
    chi2_contributions: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_docking_results_pdb(
    results: List[SimulationResult],
    atoms_per_body: List[np.ndarray],
    output_dir: str,
    prefix: str = "dock",
) -> List[str]:
    """Write docking results as multi-model PDB files.

    One PDB file per body, each MODEL entry is a docking trial.

    Returns list of written filenames.
    """
    written = []
    for body_idx, atoms_local in enumerate(atoms_per_body):
        path = f"{output_dir}/{prefix}_body_{body_idx}.pdb"
        for trial_idx, sr in enumerate(results):
            if body_idx < len(sr.translations) and body_idx < len(sr.rotations):
                trans = sr.translations[body_idx] if sr.translations else np.zeros(3)
                rot = sr.rotations[body_idx] if sr.rotations else np.eye(3)
                # Combine rotation + translation into homogeneous transform
                tf = np.eye(4)
                tf[:3, :3] = rot
                tf[:3, 3] = trans
            else:
                tf = None
            io.write_pdb(
                atoms_local[:, :3],
                path,
                chain=chr(65 + body_idx),
                res_name="DUM",
                transform=tf,
                model_index=trial_idx,
            )
        written.append(path)
    return written


def _make_coordinates_structure(
    atoms_per_body: List[np.ndarray],
) -> Structure:
    """Create a ChiSurf structure whose chains correspond to docking bodies."""
    atoms_list = []
    for body_idx, atoms_local in enumerate(atoms_per_body):
        atoms = np.zeros(len(atoms_local), dtype={"names": keys, "formats": formats})
        atoms["i"] = np.arange(1, len(atoms_local) + 1)
        atoms["atom_id"] = atoms["i"]
        atoms["atom_name"] = "CA"
        atoms["res_name"] = "UNK"
        atoms["res_id"] = atoms["i"]
        atoms["chain"] = chr(65 + body_idx)
        atoms["element"] = "C"
        atoms["xyz"] = atoms_local[:, :3]
        atoms["radius"] = 1.0
        atoms_list.append(atoms)
    structure = Structure()
    if atoms_list:
        structure.atoms = np.concatenate(atoms_list)
    else:
        structure.atoms = np.zeros(0, dtype={"names": keys, "formats": formats})
    return structure


def write_docking_results_rmf(
    results: List[SimulationResult],
    atoms_per_body: List[np.ndarray],
    output_path: str,
) -> None:
    """Write docking results as a multi-frame PMI-compatible RMF file.

    Requires IMP.rmf. Each frame is one docking trial.
    """
    from chisurf.core.models.structure.rmf import StructureRmfWriter

    structure = _make_coordinates_structure(atoms_per_body)
    with StructureRmfWriter(str(output_path), structure, root_name="FRETDocking") as writer:
        for trial_idx, sr in enumerate(results):
            frames = []
            for body_idx, atoms_local in enumerate(atoms_per_body):
                if trial_idx < len(sr.translations) and trial_idx < len(sr.rotations):
                    trans = sr.translations[trial_idx]
                    rot = sr.rotations[trial_idx]
                    xyz_global = atoms_local[:, :3] @ rot.T + trans
                    frames.append(xyz_global)
                else:
                    frames.append(atoms_local[:, :3])
            coords = np.vstack(frames) if frames else np.zeros((0, 3))
            metadata = {
                "Simulation_Energy": sr.energy,
                "Simulation_Clash_Energy": sr.clash_energy,
                "Simulation_Restraint_Energy": sr.restraint_energy,
                "Simulation_RMSD": sr.rmsd,
                "Simulation_Internal_Number": sr.internal_number,
                "Simulation_n_Translations": len(sr.translations),
                "Simulation_n_Rotations": len(sr.rotations),
            }
            writer.append(coords, name=str(trial_idx), metadata=metadata)


def write_screening_results_csv(
    results: List[ScreeningResult],
    output_path: str,
) -> None:
    """Write screening results as a CSV table."""
    lines = [
        "filename,chi2,chi2_red,n_distances,n_valid,"
        "n_viol_1sigma,n_viol_2sigma,n_viol_3sigma,n_nan,ref_rmsd"
    ]
    for r in results:
        lines.append(
            f"{r.filename},{r.chi2:.4f},{r.reduced_chi2:.4f},"
            f"{r.n_distances},{r.n_valid},"
            f"{r.n_violations_1sigma},{r.n_violations_2sigma},{r.n_violations_3sigma},"
            f"{r.n_nan},{r.ref_rmsd:.4f}"
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_pymol_pml(
    results: List[SimulationResult],
    atoms_per_body: List[np.ndarray],
    output_path: str,
    pdb_prefix: str = "dock",
) -> None:
    """Write PyMOL .pml script loading all docking result PDB files.

    The script contains: load, show, color, and orient commands.

    Parameters
    ----------
    results : list of SimulationResult
        Docking simulation results.
    atoms_per_body : list of ndarray
        Local atom coordinates for each body.
    output_path : str
        Target file path for PyMOL script (.pml).
    pdb_prefix : str
        Prefix of the written docking PDB files.
    """
    lines = []
    for body_idx in range(len(atoms_per_body)):
        filename = f"{pdb_prefix}_body_{body_idx}.pdb"
        obj_name = f"{pdb_prefix}_body_{body_idx}"
        lines.append(f"load {filename}, {obj_name}")
        lines.append(f"show cartoon, {obj_name}")
        lines.append(f"color spectrum, {obj_name}")
    lines.append("orient")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_r_table(
    results: List[SimulationResult],
    distances: Dict,
    output_path: str,
) -> None:
    """Write model distances in R-readable tab-separated format.

    Header: trial <distance_keys...>
    One row per docking trial.

    Parameters
    ----------
    results : list of SimulationResult
        Docking simulation results.
    distances : dict
        fps.json Distances section.
    output_path : str
        Target file path for output table.
    """
    keys = sorted(distances.keys())
    header = "trial\t" + "\t".join(keys)
    lines = [header]
    for i, sr in enumerate(results):
        row = [str(i)]
        for k in keys:
            val = sr.model_distances.get(k, 0.0)
            row.append(f"{val:.4f}")
        lines.append("\t".join(row))
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_chi2_table(
    results: List[SimulationResult],
    distances: Dict,
    experimental_distances: Optional[Dict],
    output_path: str,
) -> None:
    """Write per-distance chi2 contributions for all trials.

    Header: trial <distance_keys...> chi2_total
    One row per docking trial.

    Parameters
    ----------
    results : list of SimulationResult
        Docking simulation results.
    distances : dict
        fps.json Distances section.
    experimental_distances : dict, optional
        Experimental distances (overrides distances if present).
    output_path : str
        Target file path for output table.
    """
    from .distance import chi2_score
    if experimental_distances is None:
        experimental_distances = {}
    keys = sorted(distances.keys())
    header = "trial\t" + "\t".join(keys) + "\tchi2_total"
    lines = [header]
    for i, sr in enumerate(results):
        row = [str(i)]
        chi2_total = 0.0
        for k in keys:
            d_model = sr.model_distances.get(k, 0.0)
            ddef = experimental_distances.get(k, distances.get(k, {}))
            d_exp = float(ddef.get("distance", 0.0))
            err_neg = float(ddef.get("error_neg", 5.0))
            err_pos = float(ddef.get("error_pos", 5.0))
            contrib = chi2_score(d_model, d_exp, err_neg, err_pos)
            chi2_total += contrib
            row.append(f"{contrib:.4f}")
        row.append(f"{chi2_total:.4f}")
        lines.append("\t".join(row))
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

