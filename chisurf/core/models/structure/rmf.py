"""RMF output utilities for ProteinMC."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class RmfWriterError(RuntimeError):
    """Raised when ProteinMC cannot write an RMF trajectory."""


class ProteinMCRmfWriter:
    """Write accepted ProteinMC frames as an IMP-compatible RMF trajectory."""

    def __init__(self, filename: str | Path, structure) -> None:
        """Create an RMF writer for *structure*.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output RMF/RMF3 filename.
        structure : object
            ChiSurf structure with ``atoms`` and ``xyz`` attributes.
        """
        try:
            import IMP
            import IMP.algebra
            import IMP.atom
            import IMP.core
            import IMP.rmf
            import RMF
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RmfWriterError("ProteinMC RMF output requires IMP and RMF.") from exc

        self.IMP = IMP
        self.IMP_atom = IMP.atom
        self.IMP_core = IMP.core
        self.IMP_algebra = IMP.algebra
        self.IMP_rmf = IMP.rmf
        self.RMF = RMF
        self.filename = str(Path(filename))
        self.structure = structure
        self.model = IMP.Model()
        self.particles = []
        self.atom_to_particle: Dict[int, object] = {}
        self.root = self._build_hierarchy(structure)
        self.handle = RMF.create_rmf_file(self.filename)
        IMP.rmf.add_hierarchy(self.handle, self.root)

    def _build_hierarchy(self, structure):
        """Build an IMP hierarchy matching the ChiSurf atom table."""
        atoms = structure.atoms
        root_particle = self.IMP.Particle(self.model, "ProteinMC")
        root = self.IMP_atom.Hierarchy.setup_particle(root_particle)
        chains: Dict[str, object] = {}
        residues: Dict[Tuple[str, int], object] = {}

        for atom_index, atom in enumerate(atoms):
            chain_id = _as_text(atom["chain"]) if "chain" in atoms.dtype.names else "A"
            if not chain_id:
                chain_id = "A"
            res_id = int(atom["res_id"]) if "res_id" in atoms.dtype.names else atom_index + 1
            res_name = _as_text(atom["res_name"]) if "res_name" in atoms.dtype.names else "UNK"
            atom_name = _as_text(atom["atom_name"]) if "atom_name" in atoms.dtype.names else f"A{atom_index}"

            if chain_id not in chains:
                chain_particle = self.IMP.Particle(self.model, chain_id)
                chain = self.IMP_atom.Hierarchy.setup_particle(chain_particle)
                self.IMP_atom.Chain.setup_particle(chain_particle, chain_id)
                root.add_child(chain)
                chains[chain_id] = chain

            residue_key = (chain_id, res_id)
            if residue_key not in residues:
                residue_particle = self.IMP.Particle(self.model, f"{res_name} {res_id}")
                residue = self.IMP_atom.Hierarchy.setup_particle(residue_particle)
                self.IMP_atom.Residue.setup_particle(
                    residue_particle,
                    self.IMP_atom.ResidueType(res_name),
                    res_id,
                )
                chains[chain_id].add_child(residue)
                residues[residue_key] = residue

            particle = self.IMP.Particle(self.model, atom_name)
            hierarchy = self.IMP_atom.Hierarchy.setup_particle(particle)
            self.IMP_atom.Atom.setup_particle(particle, self.IMP_atom.AtomType(atom_name))
            radius = float(atom["radius"]) if "radius" in atoms.dtype.names else 1.5
            xyz = np.asarray(atom["xyz"], dtype=float)
            sphere = self.IMP_algebra.Sphere3D(
                self.IMP_algebra.Vector3D(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                max(radius, 0.1),
            )
            self.IMP_core.XYZR.setup_particle(particle, sphere)
            residues[residue_key].add_child(hierarchy)
            self.particles.append(particle)
            self.atom_to_particle[atom_index] = particle

        return root

    def append(self, xyz: np.ndarray, name: str | None = None) -> None:
        """Append one coordinate frame to the RMF trajectory."""
        coords = np.asarray(xyz, dtype=float)
        if coords.shape != (len(self.particles), 3):
            raise ValueError(
                f"xyz must have shape ({len(self.particles)}, 3), got {coords.shape!r}"
            )
        for particle, coord in zip(self.particles, coords):
            self.IMP_core.XYZ(particle).set_coordinates(
                self.IMP_algebra.Vector3D(float(coord[0]), float(coord[1]), float(coord[2]))
            )
        frame_name = str(name if name is not None else self.handle.get_number_of_frames())
        self.IMP_rmf.save_frame(self.handle, frame_name)


def _as_text(value) -> str:
    """Convert fixed-width numpy string values to plain text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()
