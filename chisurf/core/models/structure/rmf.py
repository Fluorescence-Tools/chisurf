"""PMI-compatible RMF output utilities for ChiSurf structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import gc

import numpy as np


class RmfWriterError(RuntimeError):
    """Raised when ChiSurf cannot write an RMF trajectory."""


class RmfStatWriter:
    """Write per-frame scalar metadata into an RMF ``stat`` category.

    This follows the PMI ``IMP.pmi.output.Output`` pattern: create a ``stat``
    category, create typed RMF keys for output names, save a coordinate frame,
    then set root-node values for that frame.
    """

    def __init__(
        self,
        rmf_handle: Any,
        initial_output: Optional[Mapping[str, Any]] = None,
        *,
        rmf_module: Any | None = None,
    ) -> None:
        """Create a PMI-style RMF stat writer.

        Parameters
        ----------
        rmf_handle :
            RMF file handle returned by ``RMF.create_rmf_file``.
        initial_output : mapping, optional
            Initial output names used to pre-create RMF keys.
        """
        self.handle = rmf_handle
        self.RMF = rmf_module
        if self.RMF is None:
            try:
                import RMF as _RMF
                self.RMF = _RMF
            except Exception:
                self.RMF = getattr(rmf_handle, "RMF", None)
        self.category = None
        self.keys: Dict[str, Any] = {}
        self.enabled = False

        try:
            self.category = rmf_handle.get_category("stat")
            self.enabled = True
        except Exception:
            return

        self._add_keys(initial_output or {})

    def _tag_for(self, value: Any) -> Any:
        """Return the RMF tag type matching a Python scalar value."""
        if isinstance(value, (bool, np.bool_)):
            return self.RMF.int_tag
        if isinstance(value, (int, np.integer)):
            return self.RMF.int_tag
        if isinstance(value, (float, np.floating)):
            return self.RMF.float_tag
        return self.RMF.string_tag

    def _coerce_value(self, value: Any) -> Any:
        """Convert NumPy scalars to Python scalars accepted by RMF."""
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _add_keys(self, values: Mapping[str, Any]) -> None:
        """Create RMF keys for all output names not already present."""
        if not self.enabled:
            return
        for name, value in values.items():
            key_name = str(name)
            if key_name in self.keys:
                continue
            try:
                self.keys[key_name] = self.handle.get_key(
                    self.category,
                    key_name,
                    self._tag_for(value),
                )
            except Exception:
                self.enabled = False
                return

    def write(self, values: Mapping[str, Any]) -> None:
        """Write frame metadata to the RMF root node.

        Parameters
        ----------
        values : mapping
            Per-frame metadata values. Floats, integers, booleans, and strings
            are supported. Other values are converted to strings.
        """
        if not self.enabled:
            return

        self._add_keys(values)
        if not self.enabled:
            return

        root = self.handle.get_root_node()
        frame_index = max(0, self.handle.get_number_of_frames() - 1)
        for name, value in values.items():
            key_name = str(name)
            if key_name not in self.keys:
                continue
            try:
                root.set_value(self.keys[key_name], self._coerce_value(value))
            except Exception:
                continue

        try:
            self.handle.get_root_node().set_value(
                self.keys.setdefault("rmf_frame_index", self.handle.get_key(
                    self.category,
                    "rmf_frame_index",
                    self.RMF.int_tag,
                )),
                frame_index,
            )
        except Exception:
            pass

        try:
            self.handle.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Release this writer's reference to the RMF handle."""
        try:
            del self.handle
        except Exception:
            pass
        try:
            del self.category
        except Exception:
            pass
        try:
            self.keys.clear()
            del self.keys
        except Exception:
            pass
        gc.collect()


class StructureRmfWriter:
    """Write ChiSurf structure frames as PMI-compatible RMF trajectories."""

    def __init__(
        self,
        filename: str | Path,
        structure: Any,
        *,
        stat_output: Optional[Mapping[str, Any]] = None,
        root_name: str = "ProteinMC",
    ) -> None:
        """Create an RMF writer for a ChiSurf structure.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output RMF/RMF3 filename.
        structure : object
            ChiSurf structure with ``atoms`` and ``xyz`` attributes.
        stat_output : mapping, optional
            Initial per-frame metadata keys.
        root_name : str, optional
            Name for the top-level RMF hierarchy node.
        """
        try:
            import IMP
            import IMP.algebra
            import IMP.atom
            import IMP.core
            import IMP.rmf
            import RMF
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RmfWriterError("RMF output requires IMP and RMF.") from exc

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
        self.root = self._build_hierarchy(structure, root_name=root_name)
        self.handle = RMF.create_rmf_file(self.filename)
        self._add_hierarchy()
        self.stat_writer = RmfStatWriter(self.handle, stat_output, rmf_module=RMF)

    def __enter__(self) -> "StructureRmfWriter":
        """Return this writer for ``with``-statement use."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the RMF file when leaving a ``with`` block."""
        self.close()

    @classmethod
    def from_coordinates(
        cls,
        filename: str | Path,
        coords: np.ndarray,
        model_name: str = "structure",
        transform: np.ndarray | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "StructureRmfWriter":
        """Create a writer for raw ``(N, 3)`` or ``(N, 4)`` coordinates.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output RMF/RMF3 filename.
        coords : array_like, shape ``(N, 3)`` or ``(N, 4)``
            Cartesian coordinates. Column 4 is ignored when present.
        model_name : str, optional
            Name used for the generated molecule/chain hierarchy.
        transform : array_like, optional
            Homogeneous transform, rotation matrix, or translation applied to
            ``coords`` before writing.
        metadata : mapping, optional
            Initial per-frame metadata keys.

        Returns
        -------
        StructureRmfWriter
            A configured writer ready for ``append``.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 3:
            raise ValueError(f"Expected (N, 3) or (N, 4) coordinates, got {coords.shape}")

        xyz = coords[:, :3].copy()
        if transform is not None:
            t = np.asarray(transform, dtype=np.float64)
            if t.shape == (4, 4):
                xyz = xyz @ t[:3, :3].T + t[:3, 3]
            elif t.shape == (3, 3):
                xyz = xyz @ t.T
            elif t.shape == (3,):
                xyz = xyz + t
            else:
                raise ValueError(f"Unexpected transform shape {t.shape}")

        from chisurf.core.fio.structure.coordinates import formats, keys
        from chisurf.core.structure.structure import Structure

        atoms = np.zeros(len(xyz), dtype={"names": keys, "formats": formats})
        atoms["i"] = np.arange(1, len(xyz) + 1)
        atoms["atom_id"] = atoms["i"]
        atoms["atom_name"] = "CA"
        atoms["res_name"] = "UNK"
        atoms["res_id"] = atoms["i"]
        atoms["chain"] = "A"
        atoms["element"] = "C"
        atoms["xyz"] = xyz
        atoms["radius"] = 1.0

        structure = Structure()
        structure.atoms = atoms
        return cls(filename, structure, stat_output=metadata, root_name=model_name)

    def _add_hierarchy(self) -> None:
        """Register the root hierarchy with IMP.rmf."""
        if hasattr(self.IMP_rmf, "add_hierarchies"):
            self.IMP_rmf.add_hierarchies(self.handle, [self.root])
        else:
            self.IMP_rmf.add_hierarchy(self.handle, self.root)

    def _build_hierarchy(self, structure: Any, *, root_name: str) -> Any:
        """Build an IMP/PMI-compatible hierarchy matching the atom table."""
        atoms = structure.atoms
        root_particle = self.IMP.Particle(self.model, root_name)
        root = self.IMP_atom.Hierarchy.setup_particle(root_particle)
        try:
            self.IMP_atom.State.setup_particle(root, 0)
        except Exception:
            pass

        chains: Dict[str, object] = {}
        residues: Dict[Tuple[str, int], object] = {}

        for atom_index, atom in enumerate(atoms):
            chain_id = _chain_id(atom)
            res_id = int(atom["res_id"]) if "res_id" in atoms.dtype.names else atom_index + 1
            res_name = _as_text(atom["res_name"]) if "res_name" in atoms.dtype.names else "UNK"
            if not res_name:
                res_name = "UNK"
            atom_name = _as_text(atom["atom_name"]) if "atom_name" in atoms.dtype.names else f"A{atom_index}"
            if not atom_name:
                atom_name = f"A{atom_index}"

            if chain_id not in chains:
                molecule_particle = self.IMP.Particle(self.model, f"{root_name}.{chain_id}")
                molecule = self.IMP_atom.Hierarchy.setup_particle(molecule_particle)
                try:
                    self.IMP_atom.Copy.setup_particle(molecule, 0)
                except Exception:
                    pass
                chain = self.IMP_atom.Chain.setup_particle(molecule, chain_id)
                try:
                    chain.set_sequence("")
                except Exception:
                    pass
                root.add_child(molecule)
                chains[chain_id] = chain

            residue_key = (chain_id, res_id)
            if residue_key not in residues:
                residue_particle = self.IMP.Particle(self.model, f"{res_name} {res_id}")
                residue = self.IMP_atom.Hierarchy.setup_particle(residue_particle)
                self.IMP_atom.Residue.setup_particle(
                    residue_particle,
                    _residue_type(self.IMP_atom, res_name),
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

    def append(
        self,
        xyz: np.ndarray,
        name: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one coordinate frame to the RMF trajectory.

        Parameters
        ----------
        xyz : array_like, shape ``(n_particles, 3)``
            Cartesian coordinates for the frame.
        name : str, optional
            RMF frame name. If omitted, the current frame index is used.
        metadata : mapping, optional
            Per-frame metadata written to the RMF ``stat`` category.
        """
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
        if metadata:
            self.stat_writer.write(metadata)
        else:
            try:
                self.handle.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Flush and release the RMF file handle."""
        try:
            self.handle.flush()
        except Exception:
            pass
        try:
            self.stat_writer.close()
        except Exception:
            pass
        try:
            del self.handle
        except Exception:
            pass
        gc.collect()


# Backwards-compatible alias for existing ProteinMC imports.
ProteinMCRmfWriter = StructureRmfWriter


def _chain_id(atom: np.void) -> str:
    """Return a non-empty chain identifier for an atom."""
    if "chain" in atom.dtype.names:
        chain_id = _as_text(atom["chain"])
        if chain_id:
            return chain_id
    return "A"


def _residue_type(IMP_atom: Any, res_name: str) -> Any:
    """Return an IMP residue type, falling back to UNK for unknown names."""
    try:
        return IMP_atom.ResidueType(res_name)
    except Exception:
        return IMP_atom.ResidueType("UNK")


def _as_text(value: Any) -> str:
    """Convert fixed-width NumPy string values to plain text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


__all__ = [
    "RmfStatWriter",
    "RmfWriterError",
    "StructureRmfWriter",
    "ProteinMCRmfWriter",
]
