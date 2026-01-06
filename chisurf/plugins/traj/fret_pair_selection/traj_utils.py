from __future__ import annotations

import itertools
import re
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mdtraj as md
import numpy as np

import chisurf.fio


@dataclass(frozen=True)
class ResidueSite:
    chain_id: str
    resseq: int
    resname: str
    atom_name: str
    atom_index: int


@dataclass(frozen=True)
class LabelPosition:
    name: str
    atom_index: int
    chain_id: str
    resseq: int
    resname: str
    atom_name: str


def parse_residue_ranges(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"[;,\s]+", text)
    out: List[int] = []
    for part in parts:
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            lo = min(a_i, b_i)
            hi = max(a_i, b_i)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))

    return sorted(set(out))


def _chain_id_of(chain) -> str:
    cid = getattr(chain, "id", None)
    if cid is None:
        cid = getattr(chain, "chain_id", None)
    if cid is None:
        return str(chain.index)
    return str(cid)


def get_chain(top: md.Topology, chain_spec: str):
    chain_spec = (chain_spec or "").strip()
    if not chain_spec:
        return top.chain(0)

    if chain_spec.isdigit():
        return top.chain(int(chain_spec))

    for ch in top.chains:
        if _chain_id_of(ch) == chain_spec:
            return ch
    raise ValueError(f"Chain '{chain_spec}' not found in topology")


def find_residue(chain, resseq: int):
    for res in chain.residues:
        if getattr(res, "resSeq", None) == resseq:
            return res
    return None


def pick_atom_index(residue, atom_name: str, fallback_atom_name: Optional[str] = None) -> Optional[Tuple[int, str]]:
    for at in residue.atoms:
        if at.name == atom_name:
            return at.index, atom_name
    if fallback_atom_name:
        for at in residue.atoms:
            if at.name == fallback_atom_name:
                return at.index, fallback_atom_name
    return None


def build_sites(
    traj: md.Trajectory,
    chain_spec: str,
    residue_numbers: Sequence[int],
    atom_name: str,
    fallback_atom_name: Optional[str] = None,
) -> List[ResidueSite]:
    chain = get_chain(traj.topology, chain_spec)
    chain_id = _chain_id_of(chain)

    sites: List[ResidueSite] = []
    for resseq in residue_numbers:
        res = find_residue(chain, int(resseq))
        if res is None:
            continue
        picked = pick_atom_index(res, atom_name, fallback_atom_name)
        if picked is None:
            continue
        atom_index, used_atom = picked
        sites.append(
            ResidueSite(
                chain_id=chain_id,
                resseq=int(resseq),
                resname=str(res.name),
                atom_name=str(used_atom),
                atom_index=int(atom_index),
            )
        )

    return sites


def build_atom_pairs(sites: Sequence[ResidueSite]) -> Tuple[np.ndarray, List[str]]:
    if len(sites) < 2:
        return np.zeros((0, 2), dtype=np.int32), []

    atom_indices = np.array([s.atom_index for s in sites], dtype=np.int32)
    idx_i, idx_j = np.triu_indices(len(sites), k=1)
    pairs = np.vstack([atom_indices[idx_i], atom_indices[idx_j]]).T.astype(np.int32, copy=False)

    pair_names: List[str] = []
    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        si = sites[i]
        sj = sites[j]
        pair_names.append(
            f"{si.chain_id}:{si.resname}{si.resseq}-{sj.chain_id}:{sj.resname}{sj.resseq}"
        )

    return pairs, pair_names


def compute_efficiencies(
    traj: md.Trajectory,
    atom_pairs: np.ndarray,
    r0_angstrom: float,
) -> np.ndarray:
    if atom_pairs.size == 0:
        return np.zeros((traj.n_frames, 0), dtype=np.float32)

    d_nm = md.compute_distances(traj, atom_pairs, periodic=False, opt=True)
    d_a = d_nm.astype(np.float32, copy=False) * 10.0
    r0 = float(r0_angstrom)
    x = d_a / r0
    return (1.0 / (1.0 + x ** 6)).astype(np.float32, copy=False)


def _select_av_backend() -> str:
    prefer_labellib = sys.platform.startswith('win')
    try:
        from quest.lib.imp_av import HAS_IMP_BFF
    except Exception:
        HAS_IMP_BFF = False

    try:
        from chisurf.structure.av.static import HAS_LABELLIB
    except Exception:
        HAS_LABELLIB = False

    if prefer_labellib:
        if HAS_LABELLIB:
            return 'labellib'
        if HAS_IMP_BFF:
            return 'imp'
    else:
        if HAS_IMP_BFF:
            return 'imp'
        if HAS_LABELLIB:
            return 'labellib'

    raise RuntimeError('Neither IMP.bff nor LabelLib are available to compute accessible volumes.')


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]


def _mean_efficiency_from_points(
    points1: np.ndarray,
    points2: np.ndarray,
    r0_angstrom: float,
    n_samples: int,
) -> float:
    n1 = int(points1.shape[0])
    n2 = int(points2.shape[0])
    if n1 == 0 or n2 == 0:
        return float('nan')

    i1 = np.random.randint(0, n1, size=int(n_samples))
    i2 = np.random.randint(0, n2, size=int(n_samples))
    d = points1[i1] - points2[i2]
    r = np.sqrt((d * d).sum(axis=1))
    x = r / float(r0_angstrom)
    return float(np.mean(1.0 / (1.0 + x ** 6)))


def compute_efficiencies_from_fps_av(
    traj: md.Trajectory,
    fps: Dict,
    pair_position_names: Sequence[Tuple[str, str]],
    r0s_angstrom: np.ndarray,
    n_samples: int = 5000,
    max_points_per_av: int = 20000,
) -> np.ndarray:
    backend = _select_av_backend()

    pos_cfg = fps.get('Positions') or {}
    if not isinstance(pos_cfg, dict):
        raise ValueError("fps.json: 'Positions' must be a dict")

    needed_positions = set()
    for p1, p2 in pair_position_names:
        needed_positions.add(p1)
        needed_positions.add(p2)
    for name in needed_positions:
        if name not in pos_cfg:
            raise ValueError(f"fps.json is missing position '{name}'")

    fd, tmp_pdb = tempfile.mkstemp(suffix='.pdb')
    try:
        effs = np.empty((traj.n_frames, len(pair_position_names)), dtype=np.float32)

        for frame_idx in range(traj.n_frames):
            traj[frame_idx].save_pdb(tmp_pdb)

            av_points: Dict[str, np.ndarray] = {}
            if backend == 'imp':
                from quest.lib.imp_av import build_imp_accessible_volume

                for name in needed_positions:
                    cfg = pos_cfg[name] or {}
                    chain_identifier = str(cfg.get('chain_identifier') or '').strip() or None
                    residue_seq_number = int(cfg.get('residue_seq_number'))
                    atom_name = str(cfg.get('atom_name') or 'CA').strip() or 'CA'
                    linker_length = float(cfg.get('linker_length', 20.0))
                    linker_width = float(cfg.get('linker_width', 0.5))
                    radius1 = float(cfg.get('radius1', 3.5))
                    radius2 = float(cfg.get('radius2', 0.0))
                    radius3 = float(cfg.get('radius3', 0.0))
                    allowed_sphere_radius = float(cfg.get('allowed_sphere_radius', 1.5))
                    simulation_grid_resolution = float(cfg.get('simulation_grid_resolution', 1.5))

                    av = build_imp_accessible_volume(
                        structure=tmp_pdb,
                        residue_seq_number=residue_seq_number,
                        atom_name=atom_name,
                        chain_identifier=chain_identifier,
                        linker_length=linker_length,
                        linker_width=linker_width,
                        radii=(radius1, radius2, radius3),
                        allowed_sphere_radius=allowed_sphere_radius,
                        simulation_grid_resolution=simulation_grid_resolution,
                    )
                    pts = np.asarray(av.points, dtype=np.float64)
                    pts = _downsample_points(pts, max_points_per_av)
                    av_points[name] = pts

            else:
                import chisurf.structure
                import chisurf.structure.av

                structure = chisurf.structure.Structure(tmp_pdb)
                for name in needed_positions:
                    cfg = pos_cfg[name] or {}
                    av_obj = chisurf.structure.av.BasicAV(
                        structure,
                        simulation_grid_resolution=float(cfg.get('simulation_grid_resolution', None) or 1.5),
                        allowed_sphere_radius=float(cfg.get('allowed_sphere_radius', None) or 1.5),
                        radius1=float(cfg.get('radius1', 3.5)),
                        radius2=float(cfg.get('radius2', 4.5)),
                        radius3=float(cfg.get('radius3', 3.5)),
                        linker_width=float(cfg.get('linker_width', 0.5)),
                        linker_length=float(cfg.get('linker_length', 20.0)),
                        simulation_type=str(cfg.get('simulation_type', 'AV1')),
                        chain_identifier=str(cfg.get('chain_identifier') or '').strip() or None,
                        residue_name=str(cfg.get('residue_name') or '').strip() or None,
                        residue_seq_number=int(cfg.get('residue_seq_number')),
                        atom_name=str(cfg.get('atom_name') or 'CA').strip() or 'CA',
                        position_name=str(name),
                    )
                    pts = np.asarray(av_obj.points[:, :3], dtype=np.float64)
                    pts = _downsample_points(pts, max_points_per_av)
                    av_points[name] = pts

            for pair_idx, (p1, p2) in enumerate(pair_position_names):
                e = _mean_efficiency_from_points(
                    av_points[p1],
                    av_points[p2],
                    float(r0s_angstrom[pair_idx]),
                    int(n_samples),
                )
                effs[frame_idx, pair_idx] = np.float32(e)

        return effs

    finally:
        try:
            import os
            os.close(fd)
        except Exception:
            pass
        try:
            import os
            os.remove(tmp_pdb)
        except Exception:
            pass


def rmsd_matrix(traj: md.Trajectory, atom_selection: Optional[str] = None) -> np.ndarray:
    if atom_selection:
        atom_indices = traj.topology.select(atom_selection)
        if atom_indices.size == 0:
            raise ValueError(f"RMSD selection '{atom_selection}' matched 0 atoms")
    else:
        atom_indices = None

    n = traj.n_frames
    out = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        out[:, i] = md.rmsd(traj, traj, frame=i, atom_indices=atom_indices).astype(np.float32) * 10.0
    return out


def load_fps_json(path: Union[str, Path]) -> Dict:
    p = Path(path)
    with chisurf.fio.zipped.open_maybe_zipped(filename=str(p), mode='r') as fp:
        return json.load(fp)


def positions_from_fps_json(traj: md.Trajectory, fps: Dict) -> List[LabelPosition]:
    pos_dict = fps.get("Positions") or {}
    if not isinstance(pos_dict, dict):
        raise ValueError("fps.json: 'Positions' must be a dict")

    positions: List[LabelPosition] = []
    for name, cfg in pos_dict.items():
        if not isinstance(cfg, dict):
            continue

        chain_identifier = str(cfg.get("chain_identifier") or "").strip()
        atom_name = str(cfg.get("atom_name") or "").strip() or "CA"
        resseq = cfg.get("residue_seq_number")
        if resseq is None:
            continue
        resseq = int(resseq)
        resname_cfg = str(cfg.get("residue_name") or "").strip()

        found_atom: Optional[Tuple[int, str, int, str, str]] = None
        for chain in traj.topology.chains:
            chain_id = _chain_id_of(chain)
            if chain_identifier and chain_id != chain_identifier:
                continue
            for res in chain.residues:
                if getattr(res, "resSeq", None) != resseq:
                    continue
                if resname_cfg and str(res.name) != resname_cfg:
                    continue
                picked = pick_atom_index(res, atom_name, fallback_atom_name="CA")
                if picked is None:
                    continue
                atom_index, used_atom_name = picked
                found_atom = (int(atom_index), used_atom_name, int(resseq), str(res.name), chain_id)
                break
            if found_atom is not None:
                break

        if found_atom is None:
            continue

        atom_index, used_atom_name, resseq_found, resname_found, chain_id_found = found_atom
        positions.append(
            LabelPosition(
                name=str(name),
                atom_index=int(atom_index),
                chain_id=str(chain_id_found),
                resseq=int(resseq_found),
                resname=str(resname_found),
                atom_name=str(used_atom_name),
            )
        )

    return positions


def candidate_pairs_from_fps_json(
    positions: Sequence[LabelPosition],
    fps: Dict,
    default_r0_angstrom: float,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[Tuple[str, str]]]:
    pos_by_name = {p.name: p for p in positions}
    if len(pos_by_name) < 2:
        return np.zeros((0, 2), dtype=np.int32), [], np.zeros((0,), dtype=np.float32), []

    effs_section = fps.get("Mean FRET Efficiencies")
    dist_section = fps.get("Distances")

    pairs: List[Tuple[int, int]] = []
    pair_names: List[str] = []
    r0s: List[float] = []
    pair_position_names: List[Tuple[str, str]] = []

    def add_pair(p1_name: str, p2_name: str, r0: float):
        p1 = pos_by_name.get(p1_name)
        p2 = pos_by_name.get(p2_name)
        if p1 is None or p2 is None:
            return
        if p1.atom_index == p2.atom_index:
            return
        pairs.append((int(p1.atom_index), int(p2.atom_index)))
        pair_names.append(f"{p1.name}_{p2.name}")
        r0s.append(float(r0))
        pair_position_names.append((str(p1.name), str(p2.name)))

    if isinstance(effs_section, dict) and effs_section:
        for _, v in effs_section.items():
            if not isinstance(v, dict):
                continue
            p1_name = str(v.get("position1_name") or "").strip()
            p2_name = str(v.get("position2_name") or "").strip()
            if not p1_name or not p2_name:
                continue
            r0 = float(v.get("Forster_radius") or default_r0_angstrom)
            add_pair(p1_name, p2_name, r0)

    else:
        use_distances_as_candidates = False
        if isinstance(dist_section, dict) and dist_section:
            use_distances_as_candidates = True
            for _, v in dist_section.items():
                if not isinstance(v, dict):
                    use_distances_as_candidates = False
                    break

                # If this distance entry looks like an experimental constraint, do not
                # treat it as a candidate pair list.
                if ('rda' in v) or ('prda' in v):
                    use_distances_as_candidates = False
                    break

                # Some templates store placeholders (distance/error = -1) to indicate
                # "not measured yet"; treat those as candidate definitions.
                if ('distance' in v) or ('error_pos' in v) or ('error_neg' in v):
                    d = v.get('distance', None)
                    en = v.get('error_neg', None)
                    ep = v.get('error_pos', None)
                    is_placeholder = True
                    if d is not None and float(d) >= 0.0:
                        is_placeholder = False
                    if en is not None and float(en) >= 0.0:
                        is_placeholder = False
                    if ep is not None and float(ep) >= 0.0:
                        is_placeholder = False
                    if not is_placeholder:
                        use_distances_as_candidates = False
                        break

        if use_distances_as_candidates:
            for _, v in dist_section.items():
                p1_name = str(v.get("position1_name") or "").strip()
                p2_name = str(v.get("position2_name") or "").strip()
                if not p1_name or not p2_name:
                    continue
                r0 = float(v.get("Forster_radius") or default_r0_angstrom)
                add_pair(p1_name, p2_name, r0)

        else:
            # Olga-style: candidate pool = all combinations of labeling positions.
            pos_list = list(positions)
            for i in range(len(pos_list) - 1):
                for j in range(i + 1, len(pos_list)):
                    pairs.append((int(pos_list[i].atom_index), int(pos_list[j].atom_index)))
                    pair_names.append(f"{pos_list[i].name}_{pos_list[j].name}")
                    r0s.append(float(default_r0_angstrom))

    if not pairs:
        return np.zeros((0, 2), dtype=np.int32), [], np.zeros((0,), dtype=np.float32), []

    return (
        np.asarray(pairs, dtype=np.int32),
        pair_names,
        np.asarray(r0s, dtype=np.float32),
        pair_position_names,
    )


def compute_efficiencies_per_pair_r0(
    traj: md.Trajectory,
    atom_pairs: np.ndarray,
    r0s_angstrom: np.ndarray,
) -> np.ndarray:
    if atom_pairs.size == 0:
        return np.zeros((traj.n_frames, 0), dtype=np.float32)
    if r0s_angstrom.shape[0] != atom_pairs.shape[0]:
        raise ValueError("r0s must have same length as atom_pairs")

    d_nm = md.compute_distances(traj, atom_pairs, periodic=False, opt=True)
    d_a = d_nm.astype(np.float32, copy=False) * 10.0
    r0 = r0s_angstrom.astype(np.float32, copy=False)
    x = d_a / r0[None, :]
    return (1.0 / (1.0 + x ** 6)).astype(np.float32, copy=False)
