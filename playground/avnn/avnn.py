#!/usr/bin/env python3
"""
AV-NN: End-to-end pipeline for structure → AV grids → MAE pretraining → embeddings →
supervised head → predictions (fluorescence properties).

Examples
--------
# 1) Simulate one site (writes .npz, optional .dx/.pdb):
python avnn_pipeline.py simulate \
  --site "1ANF:A:160:CA:20" --outdir grids --allowed-sphere-radius 6.0

# 2) Build NPZ dataset (expects grids named PDB_CHAINRES_ATOM.npz in --grids-dir):
python avnn_pipeline.py make-dataset \
  --sites-csv sites.csv --grids-dir grids --out avs.npz \
  --label-cols lifetime_ns --label-cols anisotropy \
  --channel-cols linker_length --dye-col dye --channel-cols dye

# 3) Pretrain MAE unsupervised on grids (saves .mae.pt with config):
python avnn_pipeline.py pretrain --npz avs.npz --epochs 50 --patch 4 --dim 256 --batch-size 4

# 4) Export embeddings:
python avnn_pipeline.py embed --npz avs.npz --weights avs.mae.pt --out avs_emb.npy --rotations 8

# 5) Fit supervised head (uses dye one-hot appended to embeddings):
python avnn_pipeline.py fit-head --npz avs.npz --embeddings avs_emb.npy \
  --labels lifetime_ns --labels anisotropy --epochs 300 --out head.pt

# 6) Predict a single site (simulate → embed → predict):
python avnn_pipeline.py predict-site \
  --pdb-id 1ANF --chain A --resseq 160 --atom CA --dye "ATTO 647N" \
  --mae-weights avs.mae.pt --head head.pt --rotations 8

# 7) Predict many sites from CSV (must have columns: pdb_id,chain,resseq,atom,dye):
python avnn_pipeline.py predict-csv \
  --sites-csv predict_sites.csv --mae-weights avs.mae.pt --head head.pt --out preds.csv

CSV schema notes
----------------
- `sites.csv` for dataset creation must include: `pdb_id, chain, resseq, atom`. Optional columns (strings) are kept;
  numeric label columns are listed via `--label-cols`; constant per-sample channels can be provided via `--channel-cols`.
- If you include a categorical dye column via `--dye-col`, a LUT and integer codes are stored;
  if you ALSO include that column in `--channel-cols`, the dye one-hot is appended as constant channels to the grid.

Dependencies
------------
- numpy, scipy (optional, for KDTree acceleration), click, torch, LabelLib, requests

"""
from __future__ import annotations
import os, csv, json, math, time, glob, random, pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import click

try:
    from click_didyoumean import DYMGroup as _Group
except Exception:  # pragma: no cover
    _Group = click.Group

import math


def _strip_appended_label_channels(g: np.ndarray, meta: dict) -> np.ndarray:
    """
    Zero the appended label channels (value_z, mask) at the END of the channel stack,
    so embeddings don't leak true labels. Shape is preserved.
    """
    if g.ndim != 4:
        return g
    try:
        names = meta.get("appended_label_names", []) or []
        per = int(meta.get("appended_label_channels_per_label", 0))
    except Exception:
        return g
    if not names or per <= 0:
        return g
    L = per * len(names)
    if L <= 0 or L > g.shape[0]:
        return g
    g = g.astype(np.float32, copy=False)
    g[-L:, :, :, :] = 0.0  # zero value_z and mask channels
    return g


def _detect_label_transform(name: str):
    """Return (kind, params) for a label name."""
    n = name.lower()
    if ("lifetime" in n) or ("tau" in n):
        return ("log", {"eps": 1e-6})
    if ("anisotropy" in n) or ("aniso" in n) or (n == "r"):
        # Typical physical upper bound for steady-state r is < 0.5
        return ("logit", {"low": 0.0, "high": 0.5, "eps": 1e-6})
    return ("identity", {})


def _append_inference_scalar_channels(grid: np.ndarray,
                                      linker_length: Optional[float],
                                      appended_label_names: List[str]) -> np.ndarray:
    """
    Append per-grid constant channels expected by MAE that were present during pretraining:
      - linker_length (actual numeric value if provided)
      - for each appended label: [value_z=0.0, mask=0.0]
        (0.0 in z-space corresponds to dataset mean after transform+zscore; mask 0 → unknown)
    """
    scalars: List[float] = []
    # 1) linker_length (if known)
    if linker_length is not None and np.isfinite(linker_length):
        scalars.append(float(linker_length))
    # 2) per-label placeholders: value_z, mask
    for _ in appended_label_names:
        scalars.extend([0.0, 0.0])
    return attach_scalar_channels(grid.astype(np.float32, copy=False), scalars)


def _forward_transform(y: np.ndarray, kind: str, params: dict):
    """Map targets to ℝ for training."""
    if kind == "identity":
        return y
    if kind == "log":
        eps = float(params.get("eps", 1e-6))
        return np.log(np.maximum(y, eps))
    if kind == "logit":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        eps = float(params.get("eps", 1e-6))
        a = (y - low) / max(high - low, 1e-12)
        a = np.clip(a, eps, 1.0 - eps)
        return np.log(a / (1.0 - a))
    raise ValueError(f"Unknown transform kind: {kind}")


def _inverse_transform(z: float, kind: str, params: dict) -> float:
    """Map back to natural units (strictly positive / bounded)."""
    if kind == "identity":
        return float(z)
    if kind == "log":
        return float(np.exp(z))
    if kind == "logit":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        s = 1.0 / (1.0 + np.exp(-z))
        return float(low + s * (high - low))
    raise ValueError(f"Unknown transform kind: {kind}")


# ------------------------------- Physical constants / LUTs --------------------

VDW: Dict[str, float] = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.80, 'S': 1.80, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
    'MG': 1.73, 'ZN': 1.39, 'FE': 1.32, 'NA': 2.27, 'K': 2.75, 'CA': 1.94, 'MN': 1.39, 'CU': 1.40, 'NI': 1.63,
    'CO': 1.67,
}

# Standard 20 AA + common alternates (0 = OTHER/UNKNOWN)
AA2NUM = {
    'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10,
    'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20,
    # common protonation/tautomers map to parent AA
    'HSE': 9, 'HSD': 9, 'HSP': 9, 'MSE': 13, 'SEC': 5, 'PYL': 12,
}

NUM2AA = {v: k for k, v in AA2NUM.items()}


# ----------------------------------- PDB parsing ------------------------------

@dataclass
class Atom:
    x: float;
    y: float;
    z: float
    element: str
    chain: str
    resseq: int
    name: str
    resname: str  # AA/ligand name


def download_pdb_text(pdb_id: str, timeout: int = 30, sleep: float = 0.0) -> str:
    import requests
    pid = pdb_id.strip().lower()
    url = f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pid}.ent"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise click.ClickException(f"Failed to fetch PDB {pdb_id.upper()}: HTTP {r.status_code}")
    if sleep > 0:
        time.sleep(sleep)
    return r.text


def parse_pdb_atoms(text: str) -> List[Atom]:
    atoms: List[Atom] = []
    for line in text.splitlines():
        if not (line.startswith('ATOM  ') or line.startswith('HETATM')):
            continue
        try:
            name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or 'A'
            resseq = int(line[22:26])
            x = float(line[30:38]);
            y = float(line[38:46]);
            z = float(line[46:54])
            element = (line[76:78].strip() or name[0]).upper()
        except Exception:
            continue
        atoms.append(Atom(x, y, z, element, chain, resseq, name, resname))
    return atoms


def atoms_to_arrays(atoms: List[Atom]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    N = len(atoms)
    xyz = np.zeros((3, N), dtype=np.float32);
    vdw = np.zeros((N,), dtype=np.float32)
    resnames: List[str] = []
    for i, a in enumerate(atoms):
        xyz[:, i] = (a.x, a.y, a.z)
        vdw[i] = VDW.get(a.element.upper(), 1.70)
        resnames.append(a.resname)
    return xyz, vdw, resnames


def find_attachment_xyz(atoms: List[Atom], chain: str, resseq: int, atom_name: str) -> np.ndarray:
    chain = chain.strip();
    atom_name = atom_name.strip()
    for a in atoms:
        if a.chain == chain and a.resseq == resseq and a.name.strip() == atom_name:
            return np.array([a.x, a.y, a.z], dtype=np.float32)
    raise click.ClickException(f"Attachment not found: {chain}:{resseq}:{atom_name}")


def find_attachment_index(atoms: List[Atom], chain: str, resseq: int, atom_name: str) -> int:
    chain = chain.strip();
    atom_name = atom_name.strip()
    for i, a in enumerate(atoms):
        if a.chain == chain and a.resseq == resseq and a.name.strip() == atom_name:
            return i
    raise click.ClickException(f"Attachment not found: {chain}:{resseq}:{atom_name}")


# -------------------------------- LabelLib adapter ----------------------------

def av_grid_labellib(
        atoms_xyz: np.ndarray,
        atoms_vdw: np.ndarray,
        atom_resnames: List[str],
        attach_xyz: np.ndarray,
        linker_length: float,
        linker_width: float,
        dye_radius: float,
        spacing: float,
        channels: str = 'aa-coded',
        z_chunk: int = 16,
        slack: float = 2.0,
        aa_none: int = 99,
        exclude_idx: Optional[int] = None,
        allowed_sphere_radius: Optional[float] = None,
) -> np.ndarray:
    """
    Build AV and "color" it per voxel.

    Modes:
      - 'aa-coded'         : int grid (D,H,W). 0 = outside AV; 1..20 = AA type; `aa_none` (e.g., 99) = inside but no AA.
      - 'aa-onehot'        : float grid (C,D,H,W). C = 22: 20 AA (ALA..VAL), 1 'none-AA', 1 inside-AV mask.
                             Outside AV → all zeros.
      - 'density','signed','density+mask' : legacy density modes.

    Geometry controls:
      - `exclude_idx`            : remove the attachment atom from obstacles.
      - `allowed_sphere_radius`  : remove any atom whose CENTER lies within this Å radius from `attach_xyz`.

    Returns float32/uint8/int16 depending on downstream cast; 'aa-onehot' always float32.
    """
    import LabelLib as ll

    # ---- Filter obstacles ----
    N = atoms_xyz.shape[1]
    keep = np.ones(N, dtype=bool)
    if exclude_idx is not None and 0 <= exclude_idx < N:
        keep[exclude_idx] = False
    if allowed_sphere_radius is not None and allowed_sphere_radius > 0.0:
        d2 = np.sum((atoms_xyz - attach_xyz[:, None]) ** 2, axis=0)
        keep &= d2 >= (allowed_sphere_radius ** 2)
    atoms_xyz = atoms_xyz[:, keep]
    atoms_vdw = atoms_vdw[keep]
    atom_resnames = [rn for j, rn in enumerate(atom_resnames) if keep[j]]

    # ---- Run LabelLib to get AV grid in (x,y,z) with .grid (Fortran order) ----
    atoms = np.vstack([atoms_xyz, atoms_vdw[None, :]])
    av1 = ll.dyeDensityAV1(
        atoms,
        attach_xyz.astype(float),
        float(linker_length),
        float(linker_width),
        float(dye_radius),
        float(spacing),
    )
    nx, ny, nz = tuple(av1.shape)  # LabelLib order (x,y,z)
    flat = np.asarray(av1.grid, dtype=np.float32)
    grid_xyz = flat.reshape((nx, ny, nz), order='F')  # (x,y,z)
    mask_xyz = (grid_xyz > 0).astype(np.float32)

    if channels in ('density', 'signed', 'density+mask'):
        grid_zyx = grid_xyz.transpose(2, 1, 0).copy()  # (D=z,H=y,W=x)
        if channels == 'density':
            return np.maximum(grid_zyx, 0.0).astype(np.float32)
        if channels == 'signed':
            return grid_zyx.astype(np.float32)
        if channels == 'density+mask':
            dens = np.maximum(grid_zyx, 0.0).astype(np.float32)
            msk = (grid_zyx > 0).astype(np.float32)
            return np.stack([dens, msk], axis=0)

    if channels not in ('aa-coded', 'aa-onehot'):
        raise click.ClickException("channels must be one of: 'aa-coded','aa-onehot','density','signed','density+mask'")

    # ---- AA assignment (shared for 'aa-coded' and 'aa-onehot') ----
    aa_num_full = np.array([AA2NUM.get(rn.upper(), 0) for rn in atom_resnames], dtype=np.int16)
    is_protein = aa_num_full > 0
    keep_prot = is_protein if np.any(is_protein) else np.ones(atoms_xyz.shape[1], dtype=bool)
    atoms_xyz_p = atoms_xyz[:, keep_prot]
    atoms_vdw_p = atoms_vdw[keep_prot]
    aa_per_atom = aa_num_full[keep_prot]

    xs = (attach_xyz[0] - spacing * (nx - 1) / 2.0) + spacing * np.arange(nx, dtype=np.float32)
    ys = (attach_xyz[1] - spacing * (ny - 1) / 2.0) + spacing * np.arange(ny, dtype=np.float32)
    zs = (attach_xyz[2] - spacing * (nz - 1) / 2.0) + spacing * np.arange(nz, dtype=np.float32)

    cutoff = float(dye_radius) + float(slack)

    # Nearest-neighbor with optional KDTree
    try:
        from scipy.spatial import cKDTree as KDTree
        tree = KDTree(atoms_xyz_p.T.astype(np.float32)) if atoms_xyz_p.shape[1] > 0 else None
    except Exception:
        tree = None
        pts_p = atoms_xyz_p.T.astype(np.float32)

    # Storage for 'aa-coded'
    coded_xyz = None
    if channels == 'aa-coded':
        coded_xyz = np.zeros((nx, ny, nz), dtype=np.int16)  # 0 = outside AV

    # Storage for 'aa-onehot': 20 AA + 1 none + 1 mask; build in xyz then transpose
    if channels == 'aa-onehot':
        C = 22
        onehot_xyz = np.zeros((C, nx, ny, nz), dtype=np.float32)  # last channel C-1 is mask

    for k0 in range(0, nz, z_chunk):
        k1 = min(nz, k0 + z_chunk)
        Zs = zs[k0:k1]
        X, Y, Z = np.meshgrid(xs, ys, Zs, indexing="ij")  # (nx, ny, k)
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        if tree is not None and pts.shape[0] > 0:
            d_center, idx = tree.query(pts, k=1, workers=-1)
        else:
            d_center = np.full((pts.shape[0],), np.inf, dtype=np.float32)
            idx = np.full((pts.shape[0],), -1, dtype=np.int32)
            if atoms_xyz_p.shape[1] > 0:
                BS = 8192
                for a0 in range(0, pts_p.shape[0], BS):
                    a1 = min(pts_p.shape[0], a0 + BS)
                    diff = pts[:, None, :] - pts_p[None, a0:a1, :]
                    dsq = np.einsum('paj,paj->pa', diff, diff, optimize=True)
                    jmin = np.argmin(dsq, axis=1)
                    dmin = np.sqrt(dsq[np.arange(dsq.shape[0]), jmin])
                    better = dmin < d_center
                    d_center[better] = dmin[better]
                    idx[better] = (a0 + jmin[better]).astype(np.int32)

        if atoms_xyz_p.shape[1] == 0:
            aa_block = np.zeros((nx, ny, k1 - k0), dtype=np.int16)
            d_block = np.full((nx, ny, k1 - k0), np.inf, dtype=np.float32)
        else:
            vdw_sel = atoms_vdw_p[idx]
            aa_sel = aa_per_atom[idx].astype(np.int16)
            d_surf = np.maximum(0.0, d_center - vdw_sel).astype(np.float32)
            aa_block = aa_sel.reshape(nx, ny, k1 - k0)
            d_block = d_surf.reshape(nx, ny, k1 - k0)

        mask_block = mask_xyz[:, :, k0:k1] > 0
        assignable = (d_block <= cutoff) & mask_block
        unknown = (aa_block == 0) & mask_block

        if channels == 'aa-coded':
            block_codes = np.full_like(aa_block, 0, dtype=np.int16)  # outside
            block_codes[assignable] = aa_block[assignable]
            # inside but unknown → aa_none
            block_codes[unknown] = int(aa_none)
            block_codes[(~assignable) & mask_block] = int(aa_none)
            coded_xyz[:, :, k0:k1] = np.where(mask_block, block_codes, 0)

        elif channels == 'aa-onehot':
            # Prepare channels: 0..19 → AA 1..20, 20 → none, 21 → mask
            # Start with zeros; fill only inside-AV
            # Map AA codes (1..20) to ch_idx (0..19); unknown/too-far → ch_idx 20
            ch_idx = np.full_like(aa_block, 20, dtype=np.int16)  # default to 'none'
            ch_idx[assignable] = (aa_block[assignable] - 1).clip(0, 19).astype(np.int16)
            ch_idx[unknown] = 20
            ch_idx[(~assignable) & mask_block] = 20  # inside but too far → 'none'

            # One-hot scatter for current z-slab
            # We’ll fill channels 0..20; channel 21 is mask
            for ch in range(21):
                onehot_xyz[ch, :, :, k0:k1] = (mask_block & (ch_idx == ch)).astype(np.float32)
            # Mask channel (C-1)
            onehot_xyz[21, :, :, k0:k1] = mask_block.astype(np.float32)

    if channels == 'aa-coded':
        coded_zyx = coded_xyz.transpose(2, 1, 0)
        return coded_zyx.astype(np.int16, copy=False)

    if channels == 'aa-onehot':
        # Convert from (C, x, y, z) → (C, D=z, H=y, W=x)
        onehot_czyx = onehot_xyz[:, :, :, :].transpose(0, 3, 2, 1).copy()
        return onehot_czyx.astype(np.float32, copy=False)


def _load_head(head_path: str):
    payload = torch.load(head_path, map_location="cpu")
    cfg = payload["config"];
    sd = payload["state_dict"]
    in_dim = int(cfg["in_dim"])
    out_dim = int(cfg.get("out_dim", len(cfg["labels"])))
    hidden = int(cfg.get("hidden", 512))
    drop = float(cfg.get("dropout", 0.1))
    head = RegrHead(in_dim, out_dim, hidden=hidden, drop=drop)
    head.load_state_dict(sd)
    head.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return head, cfg


def _ensure_mae_input_channels(grid: np.ndarray, expected_C: int,
                               dye_name: Optional[str] = None,
                               dye_lut: Optional[dict] = None) -> np.ndarray:
    """
    Make grid's channel count match the MAE expectation.

    If MAE was trained with [AA + K dye channels] and the current grid only has AA,
    this appends the dye one-hot as constant channels when dye_name & dye_lut are provided.
    Otherwise it zero-pads or truncates.
    Returns array shaped (C,D,H,W).
    """
    g = grid.astype(np.float32, copy=False)
    if g.ndim == 3:
        g = g[None, ...]  # (1,D,H,W)

    C, D, H, W = g.shape
    if C == expected_C:
        return g

    if C < expected_C:
        # Exact gap equals number of dyes? append one-hot if possible
        if dye_lut and (dye_name is not None) and (expected_C - C == len(dye_lut)):
            onehot = _onehot_from_name(dye_name, dye_lut).tolist()
            return attach_scalar_channels(g, onehot)
        # Fallback: zero-pad
        pad = expected_C - C
        return np.concatenate([g, np.zeros((pad, D, H, W), dtype=g.dtype)], axis=0)

    # C > expected_C → truncate extra channels
    return g[:expected_C]


# ------------------------------- Exports (OpenDX / PDB) -----------------------

def _write_opendx_scalar(path: str, data_zyx: np.ndarray,
                         spacing: float,
                         origin_xyz: Tuple[float, float, float],
                         field_name: str = "field"):
    nz, ny, nx = data_zyx.shape
    x0, y0, z0 = map(float, origin_xyz)
    flat = data_zyx.astype(np.float32, copy=False).reshape(nz, ny, nx).ravel(order="C")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"object 1 class gridpositions counts {nx} {ny} {nz}\n")
        fh.write(f"origin {x0:.6f} {y0:.6f} {z0:.6f}\n")
        fh.write(f"delta {spacing:.6f} 0 0\n")
        fh.write(f"delta 0 {spacing:.6f} 0\n")
        fh.write(f"delta 0 0 {spacing:.6f}\n")
        fh.write(f"object 2 class gridconnections counts {nx} {ny} {nz}\n")
        fh.write(f"object 3 class array type float rank 0 items {flat.size} data follows\n")
        per_line = 3
        for i in range(0, flat.size, per_line):
            chunk = flat[i:i + per_line]
            fh.write(" ".join(f"{v:.6g}" for v in chunk) + "\n")
        fh.write('attribute "dep" string "positions"\n')
        fh.write(f'object "{field_name}" class field\n')
        fh.write('component "positions" value 1\n')
        fh.write('component "connections" value 2\n')
        fh.write('component "data" value 3\n')


def _export_opendx(out_base_path: str,
                   grid: np.ndarray,
                   spacing: float,
                   attach_xyz: np.ndarray,
                   channels_mode: str):
    if grid.ndim == 3:
        nz, ny, nx = grid.shape
        fields = [(grid, None)]
    elif grid.ndim == 4:
        C, nz, ny, nx = grid.shape
        names = {
            "density+mask": ["density", "mask"],
            "dist+aa": ["dist", "aa"],
            "dist+aa+mask": ["dist", "aa", "mask"],
        }.get(channels_mode, [f"ch{i}" for i in range(C)])
        fields = [(grid[i], names[i] if i < len(names) else f"ch{i}") for i in range(C)]
    else:
        raise ValueError(f"Unexpected grid ndim: {grid.ndim}")

    origin = (
        float(attach_xyz[0]) - spacing * (nx - 1) / 2.0,
        float(attach_xyz[1]) - spacing * (ny - 1) / 2.0,
        float(attach_xyz[2]) - spacing * (nz - 1) / 2.0,
    )

    base_no_ext = os.path.splitext(out_base_path)[0]
    paths = []
    if len(fields) == 1:
        dx_path = base_no_ext + ".dx"
        _write_opendx_scalar(dx_path, fields[0][0], spacing, origin, field_name=channels_mode)
        paths.append(dx_path)
    else:
        for arr, nm in fields:
            dx_path = f"{base_no_ext}_{nm}.dx"
            _write_opendx_scalar(dx_path, arr, spacing, origin, field_name=nm)
            paths.append(dx_path)
    return paths


def _write_pdb_voxels(path: str,
                      grid: np.ndarray,
                      spacing: float,
                      attach_xyz: np.ndarray,
                      channels_mode: str,
                      aa_none: int = 99,
                      chain_id: str = "V"):
    """
    Write a PDB with one pseudo-atom per populated voxel, **inside AV only**.

    Supports:
      - channels_mode == "aa-coded": int grid (D,H,W). 0 = outside; 1..20 = AA; aa_none (e.g., 99) = inside but no AA.
      - channels_mode == "aa-onehot": float grid (C,D,H,W). C>=22 with:
            ch 0..19  → AA 1..20
            ch 20     → 'none-AA' (inside but no residue nearby)
            ch 21     → inside-AV mask (1 inside, 0 outside)
        Outside-AV voxels are not written.

    For other modes, this function skips PDB export (unchanged behavior).
    Voxels assigned to the same residue TYPE share the same residue sequence number:
        AA k → resseq = k (1..20)
        none → resseq = aa_none  (e.g., 99)
    """

    def _atom_line(serial: int, name: str, resname: str, chain: str, resseq: int,
                   x: float, y: float, z: float, occ: float = 1.00, bfac: float = 0.00, element: str = "C") -> str:
        return (f"HETATM{serial:5d} {name:<4s}{' ':1s}{resname:>3s} {chain:1s}"
                f"{resseq:4d}{' ':1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occ:6.2f}{bfac:6.2f}          "
                f"{element:>2s}{' ':2s}\n")

    def _resname_from_code(code: int) -> str:
        if code == aa_none:
            return "UNX"
        return NUM2AA.get(int(code), "UNX")

    def _resseq_from_code(code: int) -> int:
        if code == aa_none:
            return min(int(aa_none), 9999)
        return min(max(int(code), 1), 9999)

    # --------- Extract AA codes (D,H,W) and inside-AV mask (D,H,W) ----------
    aa_codes_zyx = None  # int32, values in {0(outside),1..20,aa_none}
    mask_zyx = None  # bool, True = inside AV

    if grid.ndim == 3 and channels_mode == "aa-coded":
        aa_codes_zyx = grid.astype(np.int32, copy=False)
        mask_zyx = (aa_codes_zyx != 0)

    elif grid.ndim == 4 and channels_mode == "aa-onehot":
        # Expect C>=22: 0..19 AAs, 20 none, 21 mask
        C, D, H, W = grid.shape
        if C < 22:
            print(f"[PDB] aa-onehot expects >=22 channels, got {C}; skipping PDB.")
            return
        mask_zyx = (grid[21] > 0.5)
        # Argmax over channels [0..20] only (AA & none). Map to codes:
        # arg=0..19 → code=1..20; arg=20 → code=aa_none
        logits = grid[0:21]  # (21,D,H,W)
        arg = np.argmax(logits, axis=0).astype(np.int32)  # (D,H,W)
        aa_codes_zyx = np.where(arg < 20, arg + 1, int(aa_none)).astype(np.int32)
        # Outside AV → code 0
        aa_codes_zyx = np.where(mask_zyx, aa_codes_zyx, 0).astype(np.int32)

    else:
        # Try legacy 4D modes with names; otherwise skip
        name_to_idx = {
            "dist+aa": {"aa": 1},
            "dist+aa+mask": {"aa": 1, "mask": 2},
            "density+mask": {"mask": 1},
        }.get(channels_mode, None)
        if name_to_idx and isinstance(grid, np.ndarray) and grid.ndim == 4 and ("aa" in name_to_idx):
            aa_codes_zyx = grid[name_to_idx["aa"]].astype(np.int32, copy=False)
            mask_zyx = (grid[name_to_idx["mask"]] > 0) if ("mask" in name_to_idx) else (aa_codes_zyx != 0)
        else:
            print(f"[PDB] channels_mode '{channels_mode}' not supported for PDB export; skipping.")
            return

    # --------- Geometry / coordinates ----------
    D, H, W = aa_codes_zyx.shape
    x0 = float(attach_xyz[0]) - spacing * (W - 1) / 2.0
    y0 = float(attach_xyz[1]) - spacing * (H - 1) / 2.0
    z0 = float(attach_xyz[2]) - spacing * (D - 1) / 2.0

    # --------- Gather inside-AV voxel indices ----------
    inside_idx = np.argwhere(mask_zyx)
    if inside_idx.size == 0:
        print(f"[PDB] No inside-AV voxels; nothing to write.")
        return

    # --------- Write PDB ----------
    serial = 1
    max_serial = 99999
    wrote = 0
    with open(path, "w", encoding="utf-8") as fh:
        for z, y, x in inside_idx:
            code = int(aa_codes_zyx[z, y, x])
            if channels_mode == "aa-coded" and code == 0:
                continue  # outside
            # Map code to resname/resseq; aa_none → UNX / aa_none
            resname = _resname_from_code(code if code != 0 else aa_none)
            resseq = _resseq_from_code(code if code != 0 else aa_none)

            X = x0 + x * spacing
            Y = y0 + y * spacing
            Z = z0 + z * spacing

            line = _atom_line(
                serial=(serial if serial <= max_serial else max_serial),
                name="V", resname=resname, chain=chain_id, resseq=resseq,
                x=X, y=Y, z=Z, occ=1.00, bfac=0.00, element="C"
            )
            fh.write(line)
            wrote += 1
            serial += 1
        fh.write("END\n")

    if serial - 1 > max_serial:
        print(f"[PDB] WARNING: atom serials exceeded {max_serial}; values were clamped.")
    print(f"[PDB] Wrote {wrote} voxels → {path}")


# --------------------------------- Dataset helpers ----------------------------

def _next_multiple(n: int, k: int) -> int:
    return ((n + k - 1) // k) * k


def _compute_target_shape(grids: list, patch: int) -> Tuple[int, int, int]:
    maxD = max((g.shape[-3] for g in grids))
    maxH = max((g.shape[-2] for g in grids))
    maxW = max((g.shape[-1] for g in grids))
    return (
        _next_multiple(maxD, patch),
        _next_multiple(maxH, patch),
        _next_multiple(maxW, patch),
    )


def read_sites_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        required = {"pdb_id", "chain", "resseq", "atom"}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            raise ValueError(f"sites.csv is missing required columns: {sorted(missing)}")
        for r in rdr:
            r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            r["pdb_id"] = r["pdb_id"].strip()
            r["chain"] = r["chain"].strip()
            r["resseq"] = int(r["resseq"])
            r["atom"] = r["atom"].strip()
            rows.append(r)
    return rows


def _grid_basename(pdb_id: str, chain: str, resseq: int, atom: str) -> str:
    return f"{pdb_id.upper()}_{chain}{resseq}_{atom.upper()}.npz"


def _case_insensitive_lookup(directory: str) -> Dict[str, str]:
    lut = {}
    for p in glob.glob(os.path.join(directory, "*.npz")):
        lut[os.path.basename(p).lower()] = p
    return lut


def _to_float_or_nan(x):
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def attach_scalar_channels(grid: np.ndarray, scalars: List[float]) -> np.ndarray:
    if grid.ndim == 3:
        grid = grid[None, ...]
    C, D, H, W = grid.shape
    extra = []
    for s in scalars:
        v = 0.0 if (s is None or (isinstance(s, float) and math.isnan(s))) else float(s)
        extra.append(np.full((1, D, H, W), v, dtype=grid.dtype))
    if extra:
        grid = np.concatenate([grid] + extra, axis=0)
    return grid


def load_grids_from_dir(
        sites_csv: str,
        grids_dir: str,
        channel_cols: Optional[List[str]] = None
) -> Tuple[List[np.ndarray], List[Dict]]:
    rows = read_sites_csv(sites_csv)
    lut = _case_insensitive_lookup(grids_dir)
    grids: List[np.ndarray] = []
    items: List[Dict] = []
    missing = []

    for r in rows:
        base = _grid_basename(r["pdb_id"], r["chain"], r["resseq"], r["atom"])
        path = os.path.join(grids_dir, base)
        if not os.path.exists(path):
            alt = lut.get(base.lower())
            if alt is None:
                missing.append(base);
                continue
            path = alt
        g = np.load(path)["arr"]
        if g.ndim not in (3, 4):
            raise ValueError(f"Unexpected grid shape for {path}: {g.shape}")
        if channel_cols:
            scalars = [_to_float_or_nan(r.get(c)) for c in channel_cols]
            g = attach_scalar_channels(g.astype(np.float32, copy=False), scalars)
        else:
            g = g.astype(np.float32, copy=False)
        grids.append(g)
        r2 = dict(r);
        r2["grid_path"] = path;
        items.append(r2)

    if missing:
        print(f"[WARN] {len(missing)} grids referenced by CSV were not found in '{grids_dir}':")
        for m in missing[:20]: print("   -", m)
        if len(missing) > 20: print(f"   ... and {len(missing) - 20} more")
    return grids, items


def build_labels_from_rows(rows: List[Dict], label_cols: List[str]):
    if not label_cols:
        return None, None, None
    Y = []
    for r in rows:
        Y.append([_to_float_or_nan(r.get(c)) for c in label_cols])
    y = np.asarray(Y, dtype=np.float32)
    mask = (~np.isnan(y)).astype(np.uint8)
    return y, mask, label_cols


def encode_categorical_column(rows: List[Dict], colname: str):
    values = [r.get(colname, "") or "" for r in rows]
    uniq = sorted({v for v in values if v})
    lut = {v: i for i, v in enumerate(uniq)}
    codes = np.array([lut.get(v, -1) for v in values], dtype=np.int32)
    return codes, lut


def save_npz(out_path: str, grids: List[np.ndarray], meta: Dict, y: Optional[np.ndarray] = None):
    arrays = {f"grid_{i:06d}": g.astype(np.float32) for i, g in enumerate(grids)}
    if y is not None: arrays["y"] = y.astype(np.float32)
    arrays["meta"] = np.array([json.dumps(meta)], dtype=object)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **arrays)


# --------------------------------- Torch / MAE --------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def random_rotate90_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.unsqueeze(0);
        squeeze_back = True
    else:
        squeeze_back = False
    planes = [(-3, -2), (-3, -1), (-2, -1)]
    random.shuffle(planes)
    for dims in planes:
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=dims).contiguous()
    return x.squeeze(0) if squeeze_back else x


@torch.no_grad()
def standardize_per_grid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-grid, per-channel standardization.
    - If a channel is (near-)binary in {0,1}, leave it UNCHANGED (common for aa-onehot + mask).
    - Otherwise z-score it per grid.
    """
    squeeze_back = False
    if x.dim() == 4:
        x = x.unsqueeze(0);
        squeeze_back = True

    x = x.float().contiguous()
    B, C, D, H, W = x.shape
    flat = x.view(B, C, -1)

    # detect near-binary channels per-sample (0/1 within tolerance)
    lo = flat.min(-1).values
    hi = flat.max(-1).values
    frac01 = ((flat == 0).float().mean(-1) + (flat == 1).float().mean(-1)) / 2.0
    near_binary = ((lo >= -1e-6) & (hi <= 1.0 + 1e-6) & (frac01 > 0.98)).view(B, C, 1, 1, 1)

    mean = flat.mean(-1).view(B, C, 1, 1, 1)
    std = flat.std(-1, unbiased=False).view(B, C, 1, 1, 1)

    x_out = torch.where(
        near_binary,  # keep binary channels untouched
        x,
        (x - mean) / (std + eps)  # z-score others
    )
    return x_out.squeeze(0).contiguous() if squeeze_back else x_out


class UnlabeledVoxelDataset(Dataset):
    def __init__(self, grids: List[np.ndarray], target_shape: Tuple[int, int, int]):
        self.grids = grids
        self.target = target_shape

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, i):
        g = self.grids[i]
        g = _pad_crop_center_3d(g, self.target, pad_value=0.0)
        if g.ndim == 3:
            t = torch.from_numpy(g).unsqueeze(0)
        elif g.ndim == 4:
            t = torch.from_numpy(g)
        else:
            raise ValueError("grid must be 3D/4D")
        return standardize_per_grid(t.contiguous())


class Patchify3D(nn.Module):
    def __init__(self, patch_size: Tuple[int, int, int]): super().__init__(); self.ps = patch_size

    def forward(self, x):  # (B,C,D,H,W)
        pd, ph, pw = self.ps;
        B, C, D, H, W = x.shape
        assert D % pd == 0 and H % ph == 0 and W % pw == 0, f"Grid {(D, H, W)} not divisible by patch {self.ps}"
        x = x.unfold(2, pd, pd).unfold(3, ph, ph).unfold(4, pw, pw)
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        B, Dn, Hn, Wn, C, pd, ph, pw = x.shape
        N = Dn * Hn * Wn
        return x.view(B, N, C * pd * ph * pw), (Dn, Hn, Wn)


class MLP(nn.Module):
    def __init__(self, d, h, o, p=0.0): super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                                                                     nn.Dropout(p), nn.Linear(h, o))

    def forward(self, x): return self.net(x)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, p_drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim);
        self.att = nn.MultiheadAttention(dim, heads, dropout=p_drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim);
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, p_drop)

    def forward(self, x):
        h = x;
        x = self.n1(x);
        x, _ = self.att(x, x, x, need_weights=False);
        x = h + x
        h = x;
        x = self.n2(x);
        x = h + self.mlp(x);
        return x


class Encoder3D(nn.Module):
    def __init__(self, patch_dim, dim=256, depth=6, heads=8, drop=0.0):
        super().__init__();
        self.proj = nn.Linear(patch_dim, dim)
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, 4.0, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim);
        self.out_dim = dim

    def forward(self, tokens):
        x = self.proj(tokens)
        for b in self.blocks: x = b(x)
        return self.norm(x)


class Decoder3D(nn.Module):
    def __init__(self, dim, patch_dim, depth=2, heads=8, drop=0.0):
        super().__init__();
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, 4.0, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim);
        self.pred = nn.Linear(dim, patch_dim)

    def forward(self, x):
        for b in self.blocks: x = b(x)
        return self.pred(self.norm(x))


class MAE3D(nn.Module):
    def __init__(self, in_ch, patch_size=(4, 4, 4), dim=256, enc_depth=6, dec_depth=2, heads=8, drop=0.0):
        super().__init__()
        self.patchify = Patchify3D(patch_size)
        P = in_ch * np.prod(patch_size);
        self.patch_dim = P
        self.encoder = Encoder3D(P, dim, enc_depth, heads, drop)
        self.decoder = Decoder3D(dim, P, dec_depth, heads, drop)
        self.pos_embed = None
        self.cfg = {
            "in_ch": int(in_ch),
            "patch_size": tuple(int(x) for x in patch_size),
            "dim": int(dim),
            "enc_depth": int(enc_depth),
            "dec_depth": int(dec_depth),
            "heads": int(heads),
            "drop": float(drop),
        }

    def _pos(self, Np, dim, device):
        if (self.pos_embed is None) or self.pos_embed.size(1) != Np or self.pos_embed.size(2) != dim:
            pe = torch.zeros(1, Np, dim, device=device);
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe)
        return self.pos_embed

    def forward(self, x, mask_ratio=0.7, return_latent=False):
        """
        Inside-aware MAE with CORRECT positional indexing:
        - gather pos embeddings using vis_idx / msk_idx
        - weight reconstruction by inside-AV occupancy (channel 0 > 0)
        """
        B, C, D, H, W = x.shape

        # ----- per-patch weights from occupancy in channel 0 (before std) -----
        with torch.no_grad():
            occ = (x[:, 0] > 0).float()  # (B,D,H,W)
            pd, ph, pw = self.patchify.ps
            occ_p = occ.unfold(1, pd, pd).unfold(2, ph, ph).unfold(3, pw, pw)
            occ_p = occ_p.mean(dim=(4, 5, 6)).contiguous()  # (B,Dn,Hn,Wn)
            Dn, Hn, Wn = occ_p.shape[1:]
            Np = Dn * Hn * Wn
            w = occ_p.view(B, Np)  # (B,Np)
            w = w / (w.mean(dim=1, keepdim=True) + 1e-6)  # normalize

        # ----- patchify & positions -----
        patches, (Dn, Hn, Wn) = self.patchify(x)  # (B,Np,patch_dim)
        Np = patches.size(1)
        pos_all = self._pos(Np, self.encoder.out_dim, x.device)  # (1,Np,dim)

        n_mask = int(mask_ratio * Np)
        vis_idx = []
        msk_idx = []
        for _ in range(B):
            perm = torch.randperm(Np, device=x.device)
            msk_idx.append(perm[:n_mask])
            vis_idx.append(perm[n_mask:])
        msk_idx = torch.stack(msk_idx, 0)  # (B,n_mask)
        vis_idx = torch.stack(vis_idx, 0)  # (B,Np-n_mask)

        # gather visible patches and their positions
        vis_patches = torch.gather(
            patches, 1, vis_idx.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        )  # (B,Nvis,P)
        pos_vis = torch.gather(
            pos_all.expand(B, -1, -1), 1, vis_idx.unsqueeze(-1).expand(-1, -1, pos_all.size(-1))
        )  # (B,Nvis,dim)

        # ----- encoder on visible tokens -----
        vis_tokens = self.encoder.proj(vis_patches) + pos_vis  # (B,Nvis,dim)
        xenc = vis_tokens
        for blk in self.encoder.blocks:
            xenc = blk(xenc)
        lat = self.encoder.norm(xenc)  # (B,Nvis,dim)
        glob = lat.mean(1, keepdim=True)  # (B,1,dim)

        # ----- decoder input for masked tokens: use their OWN positions -----
        pos_msk = torch.gather(
            pos_all.expand(B, -1, -1), 1, msk_idx.unsqueeze(-1).expand(-1, -1, pos_all.size(-1))
        )  # (B,n_mask,dim)
        dec_in = glob.repeat(1, msk_idx.size(1), 1) + pos_msk  # (B,n_mask,dim)
        pred_masked = self.decoder(dec_in)  # (B,n_mask,P)

        # ----- targets & loss (weighted L1 over masked tokens) -----
        tgt_masked = torch.gather(
            patches, 1, msk_idx.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        )  # (B,n_mask,P)
        l_tok = torch.abs(pred_masked - tgt_masked).mean(dim=2)  # (B,n_mask)
        w_sel = torch.gather(w, 1, msk_idx)  # (B,n_mask)
        loss_rec = (l_tok * w_sel).mean()

        if return_latent:
            return loss_rec, pred_masked, tgt_masked, lat
        return loss_rec, pred_masked, tgt_masked

    # DROP-IN replacement (remove @torch.no_grad)
    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward the full (unmasked) token sequence through the encoder and
        return a pooled latent. Differentiable (no @torch.no_grad).
        """
        patches, _ = self.patchify(x)  # (B, Np, patch_dim)
        Np = patches.size(1)
        pos = self._pos(Np, self.encoder.out_dim, patches.device)
        tokens = self.encoder.proj(patches) + pos  # (B, Np, dim)
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder.norm(tokens)
        return tokens.mean(1)  # (B, dim)


def _pad_crop_center_3d(arr: np.ndarray, target: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    if arr.ndim == 3:
        C = None;
        D, H, W = arr.shape;
        x = arr
    elif arr.ndim == 4:
        C, D, H, W = arr.shape;
        x = arr
    else:
        raise ValueError(f"Expected 3D or 4D grid, got shape {arr.shape}")
    tD, tH, tW = target
    out = (np.full((tD, tH, tW), pad_value, dtype=arr.dtype) if C is None else
           np.full((C, tD, tH, tW), pad_value, dtype=arr.dtype))
    sD0 = max(0, (D - tD) // 2);
    sD1 = sD0 + min(D, tD)
    sH0 = max(0, (H - tH) // 2);
    sH1 = sH0 + min(H, tH)
    sW0 = max(0, (W - tW) // 2);
    sW1 = sW0 + min(W, tW)
    dD0 = max(0, (tD - D) // 2);
    dD1 = dD0 + (sD1 - sD0)
    dH0 = max(0, (tH - H) // 2);
    dH1 = dH0 + (sH1 - sH0)
    dW0 = max(0, (tW - W) // 2);
    dW1 = dW0 + (sW1 - sW0)
    if C is None:
        out[dD0:dD1, dH0:dH1, dW0:dW1] = x[sD0:sD1, sH0:sH1, sW0:sW1]
    else:
        out[:, dD0:dD1, dH0:dH1, dW0:dW1] = x[:, sD0:sD1, sH0:sH1, sW0:sW1]
    return out


def _info_nce(lat1: torch.Tensor, lat2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    InfoNCE between two augmented views.
    Runs entirely in float32 to avoid fp16 overflow under AMP.
    """
    # Force FP32 path regardless of outer autocast
    Autocast = getattr(torch.amp, "autocast", None) or torch.cuda.amp.autocast
    with Autocast("cuda", enabled=False):
        z1 = F.normalize(lat1.float(), dim=1)  # (B,D) float32
        z2 = F.normalize(lat2.float(), dim=1)

        B = z1.size(0)
        inv_tau = 1.0 / float(tau)

        logits_11 = (z1 @ z1.t()) * inv_tau  # (B,B) float32
        logits_22 = (z2 @ z2.t()) * inv_tau
        logits_12 = (z1 @ z2.t()) * inv_tau
        logits_21 = (z2 @ z1.t()) * inv_tau

        # Safe diagonal mask value for fp16 (>= -65504), though we stay in fp32 here.
        eye = torch.eye(B, device=z1.device, dtype=torch.bool)
        mask_val = torch.tensor(-1e4, device=z1.device, dtype=logits_11.dtype)
        logits_11 = logits_11.masked_fill(eye, mask_val)
        logits_22 = logits_22.masked_fill(eye, mask_val)

        y = torch.arange(B, device=z1.device)
        loss12 = F.cross_entropy(torch.cat([logits_12, logits_11], dim=1), y)
        loss21 = F.cross_entropy(torch.cat([logits_21, logits_22], dim=1), y)
        return 0.5 * (loss12 + loss21)


@dataclass
class MAEConfig:
    patch_size: Tuple[int, int, int] = (4, 4, 4)
    dim: int = 256;
    enc_depth: int = 6;
    dec_depth: int = 2;
    heads: int = 8;
    drop: float = 0.0
    mask_ratio: float = 0.75
    batch_size: int = 4;
    lr: float = 3e-4;
    wd: float = 1e-4;
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 2
    rot_cons_w: float = 0.1;
    grad_clip: Optional[float] = 1.0
    contr_w: float = 0.5
    use_amp: bool = True
    tf32: bool = True
    prefetch_factor: int = 2


class GridsDataModule:
    def __init__(self, grids: List[np.ndarray], batch_size: int,
                 workers: int, target_shape: Tuple[int, int, int],
                 prefetch_factor: int = 2):
        self.ds = UnlabeledVoxelDataset(grids, target_shape)
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch_factor = prefetch_factor

    def loader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=(self.prefetch_factor if self.workers > 0 else None),
        )


def latent_rot_consistency(mae: MAE3D, x: torch.Tensor, mask_ratio: float, w: float = 0.1):
    _, _, _, lat1 = mae(x, mask_ratio, return_latent=True);
    xr = random_rotate90_3d(x)
    _, _, _, lat2 = mae(xr, mask_ratio, return_latent=True)
    g1 = lat1.mean(1);
    g2 = lat2.mean(1)
    return w * F.mse_loss(g1, g2)


def train_mae3d(unlabeled_grids: List[np.ndarray], in_ch: int = 1, cfg: MAEConfig = MAEConfig()) -> MAE3D:
    # --- speed knobs ---
    if torch.cuda.is_available() and cfg.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    pd, ph, pw = cfg.patch_size
    assert pd == ph == pw, "cubic patch assumed; adjust _compute_target_shape if needed"
    target = _compute_target_shape(unlabeled_grids, pd)
    dm = GridsDataModule(unlabeled_grids, cfg.batch_size, cfg.workers, target, prefetch_factor=cfg.prefetch_factor)
    dl = dm.loader()

    # auto-detect channels from first grid
    sample = unlabeled_grids[0]
    detected_in_ch = (sample.shape[0] if sample.ndim == 4 else 1)
    if detected_in_ch != in_ch:
        print(f"[MAE] Detected in_ch={detected_in_ch} (overriding provided in_ch={in_ch}).")
        in_ch = detected_in_ch

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    mae = MAE3D(in_ch=in_ch, patch_size=cfg.patch_size, dim=cfg.dim,
                enc_depth=cfg.enc_depth, dec_depth=cfg.dec_depth,
                heads=cfg.heads, drop=cfg.drop).to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.use_amp and device.type == "cuda"))
    opt = torch.optim.AdamW(mae.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    for ep in range(1, cfg.epochs + 1):
        mae.train()
        rec_sum = rot_sum = ctr_sum = total_sum = 0.0
        seen = 0

        for x in dl:
            x = x.to(device, non_blocking=True)
            if random.random() < 0.5:
                x = random_rotate90_3d(x)

            with torch.amp.autocast('cuda', enabled=(cfg.use_amp and device.type == "cuda")):
                loss_rec, _, _ = mae(x, mask_ratio=cfg.mask_ratio)

                if cfg.rot_cons_w > 0:
                    loss_rot = latent_rot_consistency(mae, x, cfg.mask_ratio, cfg.rot_cons_w)
                else:
                    loss_rot = torch.zeros((), device=x.device)

                if cfg.contr_w > 0:
                    x1 = random_rotate90_3d(x.clone())
                    x2 = random_rotate90_3d(x.clone())
                    lat1 = mae.encode_full(x1)
                    lat2 = mae.encode_full(x2)
                    loss_ctr = _info_nce(lat1, lat2, tau=0.1)
                    loss = loss_rec + loss_rot + cfg.contr_w * loss_ctr
                else:
                    loss_ctr = torch.zeros((), device=x.device)
                    loss = loss_rec + loss_rot

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(mae.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(mae.parameters(), cfg.grad_clip)
                opt.step()

            bsz = x.size(0)
            rec_sum += loss_rec.detach().item() * bsz
            rot_sum += loss_rot.detach().item() * bsz
            ctr_sum += loss_ctr.detach().item() * bsz
            total_sum += loss.detach().item() * bsz
            seen += bsz

        print(
            f"[MAE] ep {ep:03d}  rec={rec_sum / max(1, seen):.4f}  rot={rot_sum / max(1, seen):.4f}  ctr={ctr_sum / max(1, seen):.4f}  total={total_sum / max(1, seen):.4f}")

    return mae


# ------------------------------ Embedding / Checkpoint IO ---------------------

def _mae_payload(state_dict: dict, model: MAE3D) -> dict:
    return {
        "state_dict": state_dict,
        "config": model.cfg,
    }


def _load_mae_payload(weights_path: str):
    """Return (state_dict, cfg_dict_or_None). Supports old state-dict-only files."""
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        return sd["state_dict"], sd.get("config", None)
    return sd, None


def _build_mae_from_cfg(cfg: Optional[dict], fallback_in_ch: int = 1) -> MAE3D:
    if cfg is None:
        # old checkpoints default
        return MAE3D(in_ch=fallback_in_ch, patch_size=(4, 4, 4), dim=256, enc_depth=6, dec_depth=2, heads=8, drop=0.0)
    return MAE3D(
        in_ch=int(cfg.get("in_ch", fallback_in_ch)),
        patch_size=tuple(cfg.get("patch_size", (4, 4, 4))),
        dim=int(cfg.get("dim", 256)),
        enc_depth=int(cfg.get("enc_depth", 6)),
        dec_depth=int(cfg.get("dec_depth", 2)),
        heads=int(cfg.get("heads", 8)),
        drop=float(cfg.get("drop", 0.0)),
    )


@torch.no_grad()
def _encode_grid_to_emb(mae: MAE3D, grid: np.ndarray, rotations: int = 8) -> np.ndarray:
    """
    Encode a single grid to an embedding.
    Ensures the spatial dims are divisible by the MAE patch size by center pad/crop.
    Applies per-grid standardization and averages over random 90° rotations.
    """
    g = grid.astype(np.float32, copy=False)
    if g.ndim == 3:
        g = g[None, ...]  # (C=1,D,H,W)

    # Pad/crop to multiples of patch size expected by the model
    pd, ph, pw = tuple(mae.cfg.get("patch_size", (4, 4, 4)))
    C, D, H, W = g.shape
    target = (
        _next_multiple(D, pd),
        _next_multiple(H, ph),
        _next_multiple(W, pw),
    )
    g = _pad_crop_center_3d(g, target, pad_value=0.0)

    x = torch.from_numpy(g).unsqueeze(0)  # (1,C,D,H,W)
    x = standardize_per_grid(x.contiguous())
    x = x.to(next(mae.parameters()).device)

    reps = []
    R = max(1, int(rotations))
    for _ in range(R):
        xr = random_rotate90_3d(x.clone())
        reps.append(mae.encode_full(xr))
    e = torch.stack(reps, 0).mean(0).squeeze(0).cpu().numpy()
    return e


# ----------------------------------- Head (MLP) -------------------------------

class RegrHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _masked_mse(pred, target, mask, eps: float = 1e-8):
    diff2 = (pred - target) ** 2
    if mask is None:
        return diff2.mean()
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom


def _compute_norm_stats(y: np.ndarray, mask: np.ndarray):
    K = y.shape[1]
    means = np.zeros((K,), dtype=np.float32)
    stds = np.ones((K,), dtype=np.float32)
    for k in range(K):
        m = mask[:, k].astype(bool) if mask is not None else np.ones((y.shape[0],), bool)
        if m.sum() == 0:
            means[k] = 0.0;
            stds[k] = 1.0
        else:
            vals = y[m, k]
            means[k] = float(vals.mean())
            s = float(vals.std(ddof=0))
            stds[k] = s if s >= 1e-6 else 1.0
    return means, stds


# ------------------------------- Dye helpers ----------------------------------

def _require_dye_metadata(npz_dict) -> tuple:
    if "dye_lut" not in npz_dict or "dye_codes" not in npz_dict:
        raise click.ClickException(
            "This pipeline requires dye identity. Your NPZ lacks 'dye_lut' and/or 'dye_codes'. "
            "Rebuild with: make-dataset --dye-col dye (the CSV must have a 'dye' column)."
        )
    pairs = npz_dict["dye_lut"]
    dye_lut = {str(k): int(v) for k, v in pairs}
    dye_codes = np.asarray(npz_dict["dye_codes"], dtype=np.int32)
    K = len(dye_lut)
    if K == 0:
        raise click.ClickException("dye_lut is empty. Add at least one dye to the dataset.")
    if (dye_codes < 0).any():
        bad = int((dye_codes < 0).sum())
        raise click.ClickException(
            f"{bad} sample(s) have unknown dye codes. All samples must have a valid dye."
        )
    return dye_lut, dye_codes, K


def _onehot_from_code(code: int, K: int) -> np.ndarray:
    if not (0 <= int(code) < int(K)):
        raise click.ClickException(f"Invalid dye code {code} (expected 0..{K - 1}).")
    v = np.zeros((K,), dtype=np.float32)
    v[int(code)] = 1.0
    return v


def _onehot_from_name(name: str, dye_lut: dict) -> np.ndarray:
    if name is None or str(name) == "":
        raise click.ClickException("Dye must be provided (e.g., --dye \"ATTO 647N\").")
    K = len(dye_lut)
    if name not in dye_lut:
        keys = list(dye_lut.keys())
        preview = ", ".join(keys[:8]) + ("..." if len(keys) > 8 else "")
        raise click.ClickException(
            f"Dye '{name}' not found in LUT. Known dyes: {preview}"
        )
    return _onehot_from_code(dye_lut[name], K)


# -------------------------------------- CLI -----------------------------------

@click.group(cls=_Group)
def cli():
    """AV-grid → dataset → MAE → embeddings → head → predictions."""
    pass


# --- SIMULATE -----------------------------------------------------------------

@cli.command('simulate')
@click.option('--sites-csv', type=click.Path(exists=True, dir_okay=False),
              help='CSV columns: pdb_id,chain,resseq,atom,linker_length')
@click.option('--site', 'sites_tok', multiple=True, help="'PDB:CHAIN:RESSEQ:ATOM[:LINKER]' (repeatable)")
@click.option('--pdb-dir', type=click.Path(file_okay=False), default='pdbs', show_default=True)
@click.option('--outdir', type=click.Path(file_okay=False), default='grids', show_default=True)
@click.option('--linker-length', type=float, default=None, help='Default linker length')
@click.option('--linker-width', type=float, default=2.0, show_default=True)
@click.option('--dye-radius', type=float, default=3.5, show_default=True)
@click.option('--spacing', type=float, default=0.9, show_default=True)
@click.option('--channels', type=click.Choice(['aa-coded', 'density', 'signed', 'density+mask', 'aa-onehot']),
              default='aa-coded', show_default=True)
@click.option('--slack', type=float, default=2.0, show_default=True,
              help='Slack added to dye radius for AA assignment cutoff (Å).')
@click.option('--aa-none', type=int, default=99, show_default=True,
              help='Sentinel integer for "no AA" in the AA channel.')
@click.option('--allowed-sphere-radius', type=float, default=2.5, show_default=True,
              help='Ignore atoms whose centers are within this radius (Å) of the attachment point. 0 disables.')
@click.option('--overwrite/--no-overwrite', default=False, show_default=True)
@click.option('--out-dtype', type=click.Choice(['uint8', 'int16', 'float32']), default='uint8', show_default=True,
              help='Data type for saved grid (aa-coded fits in uint8).')
@click.option('--write-opendx/--no-opendx', default=False, show_default=True)
@click.option('--write-pdb/--no-pdb', default=False, show_default=True)
def cmd_simulate(
        sites_csv, sites_tok, pdb_dir, outdir, linker_length, linker_width, dye_radius, spacing,
        channels, slack, aa_none, allowed_sphere_radius, out_dtype, overwrite, write_opendx, write_pdb
):
    if not sites_csv and not sites_tok:
        raise click.ClickException('Provide --sites-csv or --site')
    sites: List[dict] = []
    if sites_csv:
        with open(sites_csv, 'r', encoding='utf-8') as fh:
            rdr = csv.DictReader(fh)
            for row in rdr:
                sites.append({
                    'pdb_id': row['pdb_id'].strip(),
                    'chain': row['chain'].strip(),
                    'resseq': int(row['resseq']),
                    'atom': row['atom'].strip(),
                    'linker_length': float(row.get('linker_length') or 'nan')
                })
    for tok in sites_tok:
        parts = tok.split(':')
        if len(parts) not in (4, 5):
            raise click.ClickException("--site must be 'PDB:CHAIN:RESSEQ:ATOM[:LINKER]'")
        pdb_id, chain, resseq, atom = parts[:4]
        L = float(parts[4]) if len(parts) == 5 else (linker_length if linker_length is not None else float('nan'))
        sites.append({'pdb_id': pdb_id, 'chain': chain, 'resseq': int(resseq), 'atom': atom, 'linker_length': L})

    os.makedirs(pdb_dir, exist_ok=True);
    os.makedirs(outdir, exist_ok=True)

    for s in sites:
        pid = s['pdb_id'].strip().lower()
        pdb_path = os.path.join(pdb_dir, f'{pid}.pdb')
        if not os.path.exists(pdb_path):
            text = download_pdb_text(pid)
            with open(pdb_path, 'w', encoding='utf-8') as fh: fh.write(text)
        with open(pdb_path, 'r', encoding='utf-8') as fh:
            pdb_txt = fh.read()
        atoms = parse_pdb_atoms(pdb_txt)
        if not atoms:
            raise click.ClickException(f'No atoms parsed for {pid.upper()}')
        xyz, vdw, resnames = atoms_to_arrays(atoms)
        attach = find_attachment_xyz(atoms, s['chain'], s['resseq'], s['atom'])
        attach_idx = find_attachment_index(atoms, s['chain'], s['resseq'], s['atom'])
        L = s['linker_length'] if not math.isnan(s['linker_length']) else (
            linker_length if linker_length is not None else 20.0)
        grid = av_grid_labellib(
            xyz, vdw, resnames,
            attach, L, linker_width, dye_radius, spacing,
            channels=channels, slack=slack, aa_none=aa_none,
            exclude_idx=attach_idx,
            allowed_sphere_radius=(allowed_sphere_radius if allowed_sphere_radius > 0 else None),
        )
        tag = f"{pid.upper()}_{s['chain']}{s['resseq']}_{s['atom'].upper()}"
        out = os.path.join(outdir, f'{tag}.npz')
        if os.path.exists(out) and not overwrite:
            click.echo(f'Exists, skipping: {out}')
            continue
        dt = {'uint8': np.uint8, 'int16': np.int16, 'float32': np.float32}[out_dtype]
        np.savez_compressed(out, arr=grid.astype(dt, copy=False))
        dx_files = []
        if write_opendx:
            dx_files = _export_opendx(out, grid, spacing, attach, channels)
        pdb_out_path = None
        if write_pdb:
            pdb_out_path = os.path.splitext(out)[0] + ".pdb"
            _write_pdb_voxels(pdb_out_path, grid, spacing, attach, channels_mode=channels, aa_none=aa_none)
        msg = f"Saved {out} shape={tuple(grid.shape)}"
        if dx_files: msg += f" + {len(dx_files)} .dx"
        if pdb_out_path: msg += " + .pdb"
        if allowed_sphere_radius and allowed_sphere_radius > 0:
            msg += f" (allowed-sphere={allowed_sphere_radius} Å)"
        click.echo(msg)


# --- MAKE DATASET (rich) ------------------------------------------------------

@cli.command("make-dataset")
@click.option("--sites-csv", type=click.Path(exists=True, dir_okay=False), required=True,
              help="CSV with columns: pdb_id,chain,resseq,atom + optional features (e.g., lifetime_ns, anisotropy, dye)")
@click.option("--grids-dir", type=click.Path(file_okay=False), required=True,
              help="Folder with *.npz grids created by 'simulate' (each NPZ contains key 'arr').")
@click.option("--out", "out_npz", type=click.Path(dir_okay=False), required=True,
              help="Output NPZ dataset (e.g., avs.npz)")
@click.option("--label-cols", multiple=True, default=[],
              help="CSV columns to export as supervised labels y (repeatable)")
@click.option("--channel-cols", multiple=True, default=[],
              help="CSV columns to bake into grids as extra constant channels (repeatable). If you also pass --dye-col and include it here, dye will be one-hot appended as channels.")
@click.option("--dye-col", default=None,
              help="Name of categorical dye column in CSV (e.g. 'dye'). Stored as codes+LUT; if also present in --channel-cols, appended as one-hot constant channels.")
# NEW: controls for appending labels as channels for MAE input
@click.option("--append-labels-as-channels/--no-append-labels-as-channels", default=True, show_default=True,
              help="Append selected labels to each grid as constant channels for MAE pretraining (plus a mask channel per label).")
@click.option("--append-which", multiple=True, default=[],
              help="Which labels to append as channels. If omitted, uses --label-cols. Typical: --append-which lifetime_ns --append-which anisotropy.")
def cmd_make_dataset(sites_csv, grids_dir, out_npz, label_cols, channel_cols, dye_col,
                     append_labels_as_channels, append_which):
    rows = read_sites_csv(sites_csv)
    channel_cols = list(channel_cols) or []
    scalar_channel_cols = [c for c in channel_cols if (dye_col is None or c != dye_col)]

    grids, items = load_grids_from_dir(
        sites_csv=sites_csv,
        grids_dir=grids_dir,
        channel_cols=scalar_channel_cols if scalar_channel_cols else None
    )
    if not grids:
        raise click.ClickException("No grids loaded; check --sites-csv and --grids-dir.")

    # Supervised labels (for head training later)
    y, y_mask, y_names = build_labels_from_rows(rows, list(label_cols))
    y_names_list = list(y_names) if y_names is not None else []

    # Dye metadata and optional one-hot as channels
    dye_codes, dye_lut = (None, None)
    if dye_col:
        dye_codes, dye_lut = encode_categorical_column(rows, dye_col)
        if dye_col in channel_cols:
            K = len(dye_lut)
            new_grids = []
            for g, code in zip(grids, dye_codes):
                if K == 0 or code < 0:
                    onehot = [0.0] * max(1, K)
                else:
                    onehot = [1.0 if k == code else 0.0 for k in range(K)]
                g2 = attach_scalar_channels(g, onehot)
                new_grids.append(g2)
            grids = new_grids

    # === NEW: append selected labels as constant channels (+ mask) ===
    appended_label_names = []
    if append_labels_as_channels:
        # Choose which labels to append
        labels_to_append = list(append_which) if append_which else list(label_cols)
        # If user didn't specify, try common names for lifetime/aniso
        if not labels_to_append:
            candidates = [n for n in y_names_list if ("lifetime" in n.lower() or "tau" in n.lower()
                                                      or "anisotropy" in n.lower() or "aniso" in n.lower())]
            labels_to_append = candidates

        # Build per-label transforms and z-scores in transformed space
        if labels_to_append:
            # Map label name -> column index in y
            name_to_idx = {str(n): i for i, n in enumerate(y_names_list)}
            missing = [nm for nm in labels_to_append if nm not in name_to_idx]
            if missing:
                raise click.ClickException(
                    f"--append-which contains labels not present in --label-cols / y: {missing}. "
                    f"Available y labels: {y_names_list}"
                )

            # Prepare transformed values and masks
            transforms = {}
            vals_z = {}  # name -> array (N,)
            masks = {}  # name -> array (N,)
            for nm in labels_to_append:
                col = name_to_idx[nm]
                raw = y[:, col] if y is not None else np.full((len(grids),), np.nan, dtype=np.float32)
                msk = (~np.isnan(raw)).astype(np.float32)
                kind, params = _detect_label_transform(nm)
                zspace = raw.copy()
                if msk.sum() > 0:
                    zspace[msk > 0] = _forward_transform(raw[msk > 0], kind, params)
                # z-score over available entries
                mu = np.nanmean(zspace) if np.any(msk > 0) else 0.0
                sd = np.nanstd(zspace) if np.any(msk > 0) else 1.0
                if not np.isfinite(sd) or sd < 1e-8:
                    sd = 1.0
                z = np.zeros_like(zspace, dtype=np.float32)
                if msk.sum() > 0:
                    z[msk > 0] = (zspace[msk > 0] - mu) / sd
                # store
                transforms[nm] = {"kind": kind, "params": params, "mu": float(mu), "sd": float(sd)}
                vals_z[nm] = z.astype(np.float32)
                masks[nm] = msk.astype(np.float32)

            # Append per-sample channels to each grid: [value_z, mask] for each label
            new_grids = []
            for i, g in enumerate(grids):
                scalars = []
                for nm in labels_to_append:
                    scalars.append(float(vals_z[nm][i]))  # value channel
                    scalars.append(float(masks[nm][i]))  # mask channel
                g2 = attach_scalar_channels(g.astype(np.float32, copy=False), scalars)
                new_grids.append(g2)
            grids = new_grids
            appended_label_names = labels_to_append

    # Prepare NPZ payload
    meta = {
        "items": items,
        "source": "precomputed_AV_grids",
        "in_ch": (grids[0].shape[0] if grids[0].ndim == 4 else 1),
        "label_cols": list(label_cols),
        "channel_cols": list(channel_cols),
        "dye_col": dye_col,
        "dye_lut": dye_lut,
        # NEW meta to document appended label-channels:
        "appended_labels_as_channels": bool(append_labels_as_channels),
        "appended_label_names": appended_label_names,  # order used
        "appended_label_channels_per_label": 2,  # [value_z, mask]
        "appended_label_value_space": "transformed+zscore",  # transformation then z-scored
    }

    arrays = {f"grid_{i:06d}": g.astype(np.float32) for i, g in enumerate(grids)}
    if y is not None:
        arrays["y"] = y.astype(np.float32)
        arrays["y_mask"] = y_mask.astype(np.uint8)
        arrays["y_names"] = np.array(y_names_list, dtype=object)
    if dye_col:
        arrays["dye_codes"] = dye_codes.astype(np.int32)
        if dye_lut is not None:
            arrays["dye_lut"] = np.array(list(dye_lut.items()), dtype=object)
    arrays["meta"] = np.array([json.dumps(meta)], dtype=object)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **arrays)

    in_ch = meta["in_ch"]
    n = len(grids)
    dye_info = f", dye_col={dye_col}, dyes={len(dye_lut) if dye_lut else 0}" if dye_col else ""
    append_info = (f", appended_labels={appended_label_names} (each as [value_z, mask])"
                   if appended_label_names else "")
    click.echo(
        f"Saved {n} grids to {out_npz} (in_ch={in_ch}, y_cols={list(label_cols)}, ch_cols={list(channel_cols)}{dye_info}{append_info})"
    )


# --- BUNDLE GRIDS (simple) ----------------------------------------------------

@cli.command('bundle-grids')
@click.argument('grid_paths', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('--out', type=click.Path(dir_okay=False), required=True)
@click.option('--labels', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--meta', type=str, default=None, help='JSON string or path')
def cmd_bundle_grids(grid_paths, out, labels, meta):
    def _load_grid_file(pth: str) -> np.ndarray:
        # Support .npz saved by `simulate` (with key "arr") and plain .npy
        if pth.lower().endswith(".npz"):
            dd = np.load(pth, allow_pickle=True)
            if "arr" in dd.files:
                return dd["arr"]
            # fallback: first "grid_*" key if present
            grid_keys = [k for k in dd.files if k.startswith("grid_")]
            if len(grid_keys) == 1:
                return dd[grid_keys[0]]
            raise click.ClickException(
                f"{pth} is an NPZ without 'arr' (and ambiguous grid_* keys)."
            )
        return np.load(pth)

    grids = [_load_grid_file(p) for p in grid_paths]
    y = np.load(labels) if labels else None

    # meta can be a JSON string or a path to a JSON file
    if meta and os.path.exists(meta):
        with open(meta, 'r', encoding='utf-8') as fh:
            meta_dict = json.loads(fh.read())
    else:
        meta_dict = json.loads(meta) if meta else {}

    in_ch = grids[0].shape[0] if grids and grids[0].ndim == 4 else 1
    meta_dict.setdefault('in_ch', in_ch)

    save_npz(out, grids, meta_dict, y)
    click.echo(f"Saved dataset with {len(grids)} grids → {out}")


# --- PRETRAIN -----------------------------------------------------------------

@cli.command("pretrain")
@click.option("--npz", "npz_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--epochs", type=int, default=50, show_default=True)
@click.option("--mask-ratio", type=float, default=0.75, show_default=True)
@click.option("--patch", type=int, default=4, show_default=True, help="Cubic patch size (voxels)")
@click.option("--dim", type=int, default=256, show_default=True, help="Transformer width")
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--workers", type=int, default=2, show_default=True, help="DataLoader worker processes")
def cmd_pretrain(npz_path, epochs, mask_ratio, patch, dim, batch_size, workers):
    d = np.load(npz_path, allow_pickle=True)
    grids = [d[k] for k in sorted([k for k in d.files if k.startswith("grid_")])]
    in_ch = (grids[0].shape[0] if grids[0].ndim == 4 else 1)
    mae_cfg = MAEConfig(
        epochs=epochs,
        mask_ratio=mask_ratio,
        patch_size=(patch, patch, patch),
        dim=dim,
        batch_size=batch_size,
        workers=workers,
    )
    mae = train_mae3d(grids, in_ch=in_ch, cfg=mae_cfg)
    payload = _mae_payload(mae.state_dict(), mae)
    out = os.path.splitext(npz_path)[0] + ".mae.pt"
    torch.save(payload, out)
    click.echo(f"Saved MAE weights + config → {out}")


# --- EMBED --------------------------------------------------------------------

@cli.command("embed")
@click.option("--npz", "npz_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="NPZ built by make-dataset/bundle-grids")
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), required=True,
              help="MAE weights (*.mae.pt) from pretrain (supports old state_dict-only files)")
@click.option("--out", "out_npy", type=click.Path(dir_okay=False), required=True,
              help="Output .npy with per-grid embeddings (N, dim)")
@click.option("--rotations", type=int, default=8, show_default=True,
              help="Average over random 90° rotations for SO(3)-robust embeddings")
@click.option("--strip-appended-label-channels/--keep-appended-label-channels", default=True, show_default=True,
              help="Zero appended label channels (value_z,mask) before encoding to prevent leakage.")
def cmd_embed(npz_path, weights, out_npy, rotations, strip_appended_label_channels):
    d = np.load(npz_path, allow_pickle=True)
    # Parse meta (JSON string in single-element object array)
    meta = {}
    if "meta" in d.files:
        try:
            meta = json.loads(str(d["meta"][0]))
        except Exception:
            meta = {}

    # Load grids
    grid_keys = sorted([k for k in d.files if k.startswith("grid_")])
    grids = [d[k] for k in grid_keys]

    # Optionally strip appended label channels to match inference-time inputs
    if strip_appended_label_channels:
        grids = [_strip_appended_label_channels(g, meta) for g in grids]

    # Load model from payload or fallback
    sd, cfg = _load_mae_payload(weights)
    fallback_in_ch = (grids[0].shape[0] if grids[0].ndim == 4 else 1)
    mae = _build_mae_from_cfg(cfg, fallback_in_ch=fallback_in_ch)

    # Optional dye metadata (if present in NPZ)
    dye_lut = None;
    dye_codes = None;
    K = 0
    if "dye_lut" in d.files and "dye_codes" in d.files:
        pairs = d["dye_lut"];
        dye_lut = {str(k): int(v) for k, v in pairs};
        K = len(dye_lut)
        dye_codes = np.asarray(d["dye_codes"], dtype=np.int32)

    exp_C = int(mae.cfg.get("in_ch", (grids[0].shape[0] if grids[0].ndim == 4 else 1)))

    msd = mae.state_dict()
    sd_filtered = {k: v for k, v in sd.items() if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
    if len(sd_filtered) != len(sd):
        print(f"[MAE] Dropping {len(sd) - len(sd_filtered)} mismatched params from checkpoint (shape mismatch).")
    mae.load_state_dict(sd_filtered, strict=False)
    mae.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu";
    mae.to(device)

    embs = []
    with torch.no_grad():
        for i, g in enumerate(grids):
            # Ensure expected channel count (append dye one-hot if MAE expects it)
            cur_C = (1 if g.ndim == 3 else g.shape[0])
            if cur_C != exp_C:
                if (dye_lut is not None) and (dye_codes is not None) and (exp_C - cur_C == K) and (
                        i < len(dye_codes)) and (dye_codes[i] >= 0):
                    onehot = np.zeros((K,), dtype=np.float32);
                    onehot[int(dye_codes[i])] = 1.0
                    g = attach_scalar_channels(g, onehot.tolist())
                else:
                    g = _ensure_mae_input_channels(g, exp_C, None, None)

            x = torch.from_numpy(g)
            if x.ndim == 3: x = x.unsqueeze(0)
            x = x.unsqueeze(0)
            x = standardize_per_grid(x.contiguous()).to(device)

            reps = []
            for _ in range(max(1, rotations)):
                xr = random_rotate90_3d(x.clone())
                reps.append(mae.encode_full(xr))
            e = torch.stack(reps, 0).mean(0).squeeze(0).cpu().numpy()
            embs.append(e)

    embs = np.stack(embs, 0).astype(np.float32)
    np.save(out_npy, embs)
    print(f"[EMBED] strip_labels={strip_appended_label_channels} → Saved embeddings {embs.shape} to {out_npy}")


# --- FIT HEAD -----------------------------------------------------------------

@cli.command("fit-head")
@click.option("--npz", "npz_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="NPZ from make-dataset (must include y, y_names, dye_lut, dye_codes)")
@click.option("--embeddings", type=click.Path(exists=True, dir_okay=False), required=True,
              help="N×D .npy from 'embed' (embeddings computed from grids; dye is added here as one-hot)")
@click.option("--labels", multiple=True, required=True,
              help="Targets to predict (subset of y_names), e.g. --labels lifetime_ns --labels anisotropy")
@click.option("--epochs", type=int, default=300, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=5e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--hidden", type=int, default=512, show_default=True)
@click.option("--dropout", type=float, default=0.1, show_default=True)
@click.option("--out", "out_head", type=click.Path(dir_okay=False), required=True,
              help="Output .pt (weights + config)")
def cmd_fit_head(npz_path, embeddings, labels, epochs, batch_size, lr, weight_decay, hidden, dropout, out_head):
    X = np.load(embeddings).astype(np.float32)
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"] if "y" in d.files else None
    yn = list(d["y_names"]) if "y_names" in d.files else None
    if y is None or yn is None:
        raise click.ClickException("NPZ lacks y and y_names. Rebuild with --label-cols.")
    dye_lut, dye_codes, K = _require_dye_metadata(d)

    y_names = list(map(str, yn))
    label_list = [str(l) for l in labels]
    try:
        cols = [y_names.index(l) for l in label_list]
    except ValueError as e:
        raise click.ClickException(f"labels must be subset of y_names. Available: {y_names}") from e

    # Select target columns
    Y = y[:, cols].astype(np.float32)
    M = (~np.isnan(Y)).astype(np.float32)

    # --- NEW: per-label transforms ---
    transforms = []
    Yt = Y.copy()
    for k, nm in enumerate(label_list):
        kind, params = _detect_label_transform(nm)
        transforms.append((kind, params))
        # apply only where mask is valid
        mk = M[:, k] > 0
        if mk.any():
            Yt[mk, k] = _forward_transform(Y[mk, k], kind, params)
        # if not valid, leave zeros (will be masked anyway)

    # Clean numerics
    badX = ~np.isfinite(X)
    if badX.any():
        X = X.copy();
        X[badX] = 0.0
    badY = ~np.isfinite(Yt)
    if badY.any():
        Yt = Yt.copy();
        M = M.copy()
        Yt[badY] = 0.0;
        M[badY] = 0.0

    # --- Normalize in transformed space ---
    mu, sd = _compute_norm_stats(Yt, M)
    Yn = np.zeros_like(Yt, dtype=np.float32)
    for k in range(Yt.shape[1]):
        mk = M[:, k] > 0
        if mk.any():
            Yn[mk, k] = (Yt[mk, k] - mu[k]) / sd[k]

    badX = ~np.isfinite(X)
    if badX.any():
        X = X.copy();
        X[badX] = 0.0
    badY = ~np.isfinite(Y)
    if badY.any():
        Y = Y.copy();
        M = M.copy();
        Y[badY] = 0.0;
        M[badY] = 0.0

    N, D = X.shape
    onehots = np.zeros((N, K), dtype=np.float32)
    for i, c in enumerate(dye_codes):
        onehots[i] = _onehot_from_code(int(c), K)
    X_aug = np.concatenate([X, onehots], axis=1).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_t = torch.from_numpy(X_aug)
    Y_t = torch.from_numpy(Yn)
    M_t = torch.from_numpy(M)

    head = RegrHead(in_dim=X_aug.shape[1], out_dim=Y.shape[1], hidden=hidden, drop=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    idx = np.arange(N)
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx);
        run = 0.0;
        seen = 0
        for s in range(0, N, batch_size):
            j = idx[s:s + batch_size]
            xb = X_t[j].to(device)
            yb = Y_t[j].to(device)
            mb = M_t[j].to(device)
            xb = torch.nan_to_num(xb);
            yb = torch.nan_to_num(yb);
            mb = torch.nan_to_num(mb)
            pred = head(xb)
            loss = _masked_mse(pred, yb, mb)
            opt.zero_grad(set_to_none=True);
            loss.backward();
            nn.utils.clip_grad_norm_(head.parameters(), 1.0);
            opt.step()
            run += float(loss.item()) * xb.size(0);
            seen += xb.size(0)
        print(f"[HEAD] ep {ep:04d}  loss {run / max(1, seen):.6f}")

    cfg = {
        "in_dim": int(X_aug.shape[1]),
        "emb_dim": int(D),
        "dye_K": int(K),
        "labels": label_list,
        "hidden": int(hidden),
        "dropout": float(dropout),
        "y_mean": mu.tolist(),
        "y_std": sd.tolist(),
        "dye_lut": dye_lut,
        "uses_dye": True,
        "input_mode": "emb_plus_dye",
        "out_dim": int(Y.shape[1]),
        "notes": "Head input = [embedding || dye_onehot]; targets de-normalized on output.",
        "label_transforms": [{"kind": k, "params": p} for (k, p) in transforms],
    }
    payload = {"state_dict": head.state_dict(), "config": cfg}
    os.makedirs(os.path.dirname(out_head) or ".", exist_ok=True)
    torch.save(payload, out_head)
    click.echo(f"Saved head → {out_head} (labels={label_list}, emb_dim={D}, dye_K={K})")


# --- PREDICT ------------------------------------------------------------------

@cli.command("predict-site")
@click.option("--pdb-id", required=True)
@click.option("--chain", required=True)
@click.option("--resseq", type=int, required=True)
@click.option("--atom", required=True)
@click.option("--dye", required=True, help="MUST match training LUT (case-sensitive)")
@click.option("--mae-weights", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--head", "head_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--rotations", type=int, default=8, show_default=True)
@click.option("--linker-length", type=float, default=20.0, show_default=True)
@click.option("--linker-width", type=float, default=2.0, show_default=True)
@click.option("--dye-radius", type=float, default=3.5, show_default=True)
@click.option("--spacing", type=float, default=0.9, show_default=True)
@click.option("--slack", type=float, default=2.0, show_default=True)
@click.option("--allowed-sphere-radius", type=float, default=2.5, show_default=True)
def cmd_predict_site(pdb_id, chain, resseq, atom, dye, mae_weights, head_path, rotations,
                     linker_length, linker_width, dye_radius, spacing, slack, allowed_sphere_radius):
    head, cfg = _load_head(head_path)

    if not cfg.get("uses_dye", False) or cfg.get("input_mode") != "emb_plus_dye":
        raise click.ClickException("Head was not trained with dye features. Re-train with 'fit-head'.")
    dye_lut = cfg.get("dye_lut", {})
    K = int(cfg.get("dye_K", 0))
    if K <= 0 or not dye_lut:
        raise click.ClickException("Head config missing dye LUT. Re-train head with dye.")

    # Labels used by head (assumed also appended as channels during MAE pretraining)
    appended_label_names = list(cfg.get("labels", []))

    # Simulate grid
    text = download_pdb_text(pdb_id)
    atoms = parse_pdb_atoms(text)
    if not atoms:
        raise click.ClickException(f"No atoms parsed for {pdb_id}")
    xyz, vdw, resnames = atoms_to_arrays(atoms)
    attach = find_attachment_xyz(atoms, chain, resseq, atom)
    attach_idx = find_attachment_index(atoms, chain, resseq, atom)
    grid = av_grid_labellib(
        xyz, vdw, resnames, attach,
        linker_length, linker_width, dye_radius, spacing,
        channels='aa-coded', slack=slack, aa_none=99,
        exclude_idx=attach_idx,
        allowed_sphere_radius=(
            allowed_sphere_radius if (allowed_sphere_radius and allowed_sphere_radius > 0) else None),
    )

    # Load MAE
    sd, mcfg = _load_mae_payload(mae_weights)
    mae = _build_mae_from_cfg(mcfg, fallback_in_ch=1)
    msd = mae.state_dict();
    sd_filtered = {k: v for k, v in sd.items() if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
    mae.load_state_dict(sd_filtered, strict=False)
    mae.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate constant channels MAE saw at pretrain time:
    #   linker_length (actual), plus per-label placeholders [value_z=0, mask=0]
    grid = _append_inference_scalar_channels(grid, linker_length, appended_label_names)

    # Ensure dye one-hot presence / channel alignment
    exp_C = int(mae.cfg.get("in_ch", 1))
    grid = _ensure_mae_input_channels(grid, exp_C, dye_name=dye, dye_lut=dye_lut)

    emb = _encode_grid_to_emb(mae, grid, rotations=rotations)

    dvec = _onehot_from_name(dye, dye_lut)
    x_aug = np.concatenate([emb, dvec], axis=0).astype(np.float32)
    x = torch.from_numpy(x_aug[None, :]).to(next(head.parameters()).device)
    with torch.no_grad():
        yhat_n = head(x).cpu().numpy().squeeze(0)

    mu = np.asarray(cfg["y_mean"], dtype=np.float32)
    sd = np.asarray(cfg["y_std"], dtype=np.float32)
    z = (yhat_n * sd + mu)

    tf_specs = cfg.get("label_transforms", [{"kind": "identity", "params": {}}] * len(cfg["labels"]))
    out_vals = []
    for i, z_i in enumerate(z.tolist()):
        spec = tf_specs[i] if i < len(tf_specs) else {"kind": "identity", "params": {}}
        val = _inverse_transform(z_i, spec.get("kind", "identity"), spec.get("params", {}))
        out_vals.append(float(val))

    result = {name: val for name, val in zip(cfg["labels"], out_vals)}
    units = {name: ("ns" if ("time" in name or "tau" in name or "lifetime" in name) else "-") for name in cfg["labels"]}

    click.echo(json.dumps({
        "site": f"{pdb_id.upper()}:{chain}:{resseq}:{atom}",
        "dye": dye,
        "pred": result,
        "units": units,
        "rotations": rotations,
        "notes": "MAE input matched to training: [AA || linker_length || per-label(value_z,mask) || dye_onehot].",
    }, indent=2))


# --- Predict CSV --------------------------------------------------------------

@cli.command("predict-csv")
@click.option("--sites-csv", type=click.Path(exists=True, dir_okay=False), required=True,
              help="CSV with columns: pdb_id,chain,resseq,atom,dye[,linker_length]]")
@click.option("--mae-weights", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--head", "head_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", type=click.Path(dir_okay=False), required=True, help="Output CSV with predictions")
@click.option("--rotations", type=int, default=4, show_default=True)
@click.option("--linker-width", type=float, default=2.0, show_default=True)
@click.option("--dye-radius", type=float, default=3.5, show_default=True)
@click.option("--spacing", type=float, default=0.9, show_default=True)
@click.option("--slack", type=float, default=2.0, show_default=True)
@click.option("--allowed-sphere-radius", type=float, default=2.5, show_default=True)
def cmd_predict_csv(sites_csv, mae_weights, head_path, out, rotations,
                    linker_width, dye_radius, spacing, slack, allowed_sphere_radius):
    import csv as _csv
    head, cfg = _load_head(head_path)

    if not cfg.get("uses_dye", False) or cfg.get("input_mode") != "emb_plus_dye":
        raise click.ClickException("Head was not trained with dye features. Re-train with 'fit-head'.")
    dye_lut = cfg.get("dye_lut", {})
    K = int(cfg.get("dye_K", 0))
    if K <= 0 or not dye_lut:
        raise click.ClickException("Head config missing dye LUT. Re-train head with dye.")

    appended_label_names = list(cfg.get("labels", []))

    with open(sites_csv, "r", encoding="utf-8") as fh:
        rdr = _csv.DictReader(fh)
        rows = list(rdr)
    for r in rows:
        if "dye" not in r or not r["dye"].strip():
            raise click.ClickException("Input CSV must contain a non-empty 'dye' column for every row.")

    sd, mcfg = _load_mae_payload(mae_weights)
    mae = _build_mae_from_cfg(mcfg, fallback_in_ch=1)
    msd = mae.state_dict();
    sd_filtered = {k: v for k, v in sd.items() if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
    mae.load_state_dict(sd_filtered, strict=False)
    mae.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    preds_out = []
    dev = next(head.parameters()).device

    for r in rows:
        pdb_id = r["pdb_id"].strip();
        chain = r["chain"].strip()
        resseq = int(r["resseq"]);
        atom = r["atom"].strip()
        dye = r["dye"].strip()
        # Use per-row linker_length if present, else NaN (will be ignored)
        L = float(r["linker_length"]) if ("linker_length" in r and r["linker_length"]) else float("nan")

        text = download_pdb_text(pdb_id)
        atoms = parse_pdb_atoms(text)
        if not atoms:
            raise click.ClickException(f"No atoms parsed for {pdb_id}")
        xyz, vdw, resnames = atoms_to_arrays(atoms)
        attach = find_attachment_xyz(atoms, chain, resseq, atom)
        attach_idx = find_attachment_index(atoms, chain, resseq, atom)

        grid = av_grid_labellib(
            xyz, vdw, resnames, attach,
            (L if np.isfinite(L) else 20.0), linker_width, dye_radius, spacing,
            channels='aa-coded', slack=slack, aa_none=99,
            exclude_idx=attach_idx,
            allowed_sphere_radius=(
                allowed_sphere_radius if (allowed_sphere_radius and allowed_sphere_radius > 0) else None),
        )

        # Recreate constant channels seen during pretraining
        grid = _append_inference_scalar_channels(grid, (L if np.isfinite(L) else None), appended_label_names)

        # Ensure dye one-hot presence / channel alignment
        exp_C = int(mae.cfg.get("in_ch", 1))
        grid = _ensure_mae_input_channels(grid, exp_C, dye_name=dye, dye_lut=dye_lut)

        emb = _encode_grid_to_emb(mae, grid, rotations=rotations)

        dvec = _onehot_from_name(dye, dye_lut)
        x_aug = np.concatenate([emb, dvec], axis=0).astype(np.float32)
        with torch.no_grad():
            yhat_n = head(torch.from_numpy(x_aug[None, :]).to(dev)).cpu().numpy().squeeze(0)

        mu = np.asarray(cfg["y_mean"], dtype=np.float32)
        sdv = np.asarray(cfg["y_std"], dtype=np.float32)
        z = (yhat_n * sdv + mu)

        tf_specs = cfg.get("label_transforms", [{"kind": "identity", "params": {}}] * len(cfg["labels"]))
        yhat_physical = []
        for i, z_i in enumerate(z.tolist()):
            spec = tf_specs[i] if i < len(tf_specs) else {"kind": "identity", "params": {}}
            yhat_physical.append(_inverse_transform(z_i, spec.get("kind", "identity"), spec.get("params", {})))

        out_row = dict(r)
        for name, val in zip(cfg["labels"], yhat_physical):
            out_row[name] = float(val)

        preds_out.append(out_row)

    out_fields = list(rows[0].keys()) + cfg["labels"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader();
        w.writerows(preds_out)
    click.echo(f"Wrote predictions → {out}  (rows={len(preds_out)})")


# ------------------------------- Utils ----------------------------------------


@cli.command("debug-emb")
@click.option("--npz", "npz_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="NPZ used for labels and metadata.")
@click.option("--embeddings", type=click.Path(exists=True, dir_okay=False), required=True,
              help="N×D .npy from 'embed'.")
def cmd_debug_emb(npz_path, embeddings):
    """
    Diagnostics for embedding spread & identifiability.
    Prints per-dim std, overall variance, pairwise cosine stats, and nearest neighbors.
    If embeddings are collapsed, std and pairwise distances will be tiny.
    """
    import numpy as _np
    X = _np.load(embeddings).astype(_np.float32)  # (N,D)
    d = _np.load(npz_path, allow_pickle=True)
    items = []
    if "meta" in d.files:
        try:
            meta = json.loads(str(d["meta"][0]))
            items = meta.get("items", [])
        except Exception:
            items = []
    N, D = X.shape
    std_per_dim = X.std(axis=0)
    overall_std = std_per_dim.mean()

    # cosine similarities
    def _cos(a, b):
        na = (a * a).sum() ** 0.5;
        nb = (b * b).sum() ** 0.5
        if na == 0 or nb == 0: return 0.0
        return float((a @ b) / (na * nb))

    cosims = []
    for i in range(min(N, 200)):
        for j in range(i + 1, min(N, 200)):
            cosims.append(_cos(X[i], X[j]))
    cosims = _np.array(cosims, dtype=_np.float32) if cosims else _np.zeros((0,), _np.float32)

    print(f"[DEBUG-EMB] N={N}, D={D}")
    print(
        f"[DEBUG-EMB] mean per-dim std = {overall_std:.6f}  (min={std_per_dim.min():.6f}, max={std_per_dim.max():.6f})")
    if cosims.size:
        print(
            f"[DEBUG-EMB] cosine: mean={cosims.mean():.6f}, std={cosims.std():.6f}, min={cosims.min():.6f}, max={cosims.max():.6f}")
    else:
        print("[DEBUG-EMB] cosine: not enough samples")

    # Nearest neighbor preview for first few items
    K = min(N, 5)
    for i in range(K):
        sims = []
        for j in range(N):
            if i == j: continue
            sims.append((_cos(X[i], X[j]), j))
        sims.sort(reverse=True)  # highest cosine = closest
        nn = sims[0] if sims else (None, None)
        tag = (f"{items[i]['pdb_id'].upper()}:{items[i]['chain']}:{items[i]['resseq']}:{items[i]['atom']}"
               if i < len(items) else f"idx{i}")
        nn_tag = (
            f"{items[nn[1]]['pdb_id'].upper()}:{items[nn[1]]['chain']}:{items[nn[1]]['resseq']}:{items[nn[1]]['atom']}"
            if (nn[1] is not None and nn[1] < len(items)) else str(nn[1]))
        print(f"[DEBUG-EMB] row {i:03d} {tag}  nearest cos={nn[0]:.6f} → {nn_tag}")


@cli.command("fit-head-ridge")
@click.option("--npz", "npz_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="NPZ with y, dye metadata, and optional linker_length channel-col.")
@click.option("--embeddings", type=click.Path(exists=True, dir_okay=False), required=True,
              help="N×D .npy from 'embed'.")
@click.option("--labels", multiple=True, required=True,
              help="Targets to predict (subset of y_names), e.g. --labels lifetime_ns --labels anisotropy")
@click.option("--alpha", type=float, default=1.0, show_default=True,
              help="Ridge regularization strength (L2).")
@click.option("--out", "out_head", type=click.Path(dir_okay=False), required=True,
              help="Output .pt (weights + config) compatible with predict-site (uses same RegrHead interface).")
def cmd_fit_head_ridge(npz_path, embeddings, labels, alpha, out_head):
    """
    Label-efficient linear probe: ridge regression on [emb || dye_onehot || linker_length].
    Trains closed-form per target in transformed space, stores stats to invert back.
    """
    import numpy as _np
    X = _np.load(embeddings).astype(_np.float32)  # (N,D)
    d = _np.load(npz_path, allow_pickle=True)

    # y handling
    if "y" not in d.files or "y_names" not in d.files:
        raise click.ClickException("NPZ lacks y or y_names.")
    y_all = d["y"].astype(_np.float32)
    y_names = [str(s) for s in list(d["y_names"])]
    cols = []
    for nm in labels:
        if nm not in y_names:
            raise click.ClickException(f"Label '{nm}' not in y_names: {y_names}")
        cols.append(y_names.index(nm))
    Y = y_all[:, cols].astype(_np.float32)  # (N,K)

    # dye metadata (optional but recommended)
    dye_lut = None;
    dye_codes = None;
    Kd = 0
    if "dye_lut" in d.files and "dye_codes" in d.files:
        pairs = d["dye_lut"]
        dye_lut = {str(k): int(v) for k, v in pairs}
        dye_codes = _np.asarray(d["dye_codes"], dtype=_np.int32)
        Kd = len(dye_lut) if dye_lut else 0
    # linker_length optional (if present in channel_cols)
    link = None
    if "meta" in d.files:
        try:
            meta = json.loads(str(d["meta"][0]))
            if "channel_cols" in meta and "linker_length" in meta["channel_cols"]:
                # constant channels are appended in make-dataset order; find the index if needed.
                # But simpler: pull from CSV mirror in meta items if present.
                items = meta.get("items", [])
                if items and ("linker_length" in items[0]):
                    link = _np.array([float(it.get("linker_length", _np.nan)) for it in items], dtype=_np.float32)
        except Exception:
            pass

    # build design matrix: [emb, dye_onehot?, linker_length?]
    parts = [X]
    if Kd > 0 and dye_codes is not None:
        onehot = _np.zeros((X.shape[0], Kd), dtype=_np.float32)
        for i, c in enumerate(dye_codes):
            if 0 <= c < Kd: onehot[i, int(c)] = 1.0
        parts.append(onehot)
    if link is not None and _np.isfinite(link).any():
        parts.append(link.reshape(-1, 1))
    X_aug = _np.concatenate(parts, axis=1).astype(_np.float32)

    # transforms → z space
    Kt = Y.shape[1]
    t_specs = []
    Z = _np.zeros_like(Y, dtype=_np.float32)
    M = (~_np.isnan(Y)).astype(_np.float32)
    for k in range(Kt):
        nm = labels[k]
        kind, params = _detect_label_transform(nm)
        t_specs.append((kind, params))
        mk = M[:, k] > 0
        if mk.any():
            Z[mk, k] = _forward_transform(Y[mk, k], kind, params)

    # standardize features (important for ridge)
    muX = X_aug.mean(axis=0, keepdims=True)
    sdX = X_aug.std(axis=0, keepdims=True) + 1e-6
    Xn = (X_aug - muX) / sdX

    # closed-form ridge per target in z-space: w = (X^T X + alpha I)^(-1) X^T y
    W = _np.zeros((Xn.shape[1], Kt), dtype=_np.float32)
    b = _np.zeros((Kt,), dtype=_np.float32)
    for k in range(Kt):
        mk = M[:, k] > 0
        if not mk.any():
            continue
        Xk = Xn[mk]
        zk = Z[mk, k]
        Xkb = _np.concatenate([Xk, _np.ones((Xk.shape[0], 1), dtype=_np.float32)], axis=1)
        I = _np.eye(Xkb.shape[1], dtype=_np.float32);
        I[-1, -1] = 0.0  # no reg on bias
        A = Xkb.T @ Xkb + alpha * I
        w_full = _np.linalg.solve(A, Xkb.T @ zk)
        W[:, k] = w_full[:-1]
        b[k] = w_full[-1]

    # pack into a small torch head so predict-site stays unchanged
    cfg = {
        "in_dim": int(X_aug.shape[1]),
        "emb_dim": int(X.shape[1]),
        "dye_K": int(Kd),
        "labels": list(labels),
        "hidden": 0,  # not used
        "dropout": 0.0,  # not used
        "y_mean": [0.0] * Kt,  # not used (we store transform-only)
        "y_std": [1.0] * Kt,  # not used
        "dye_lut": dye_lut,
        "uses_dye": (Kd > 0),
        "input_mode": "emb_plus_dye",  # for predict-site compatibility
        "out_dim": int(Kt),
        "label_transforms": [{"kind": k, "params": p} for (k, p) in t_specs],
        # feature normalization for inference:
        "feat_mu": muX.squeeze(0).tolist(),
        "feat_sd": sdX.squeeze(0).tolist(),
        # linear weights:
        "ridge_W": W.tolist(),
        "ridge_b": b.tolist(),
        "notes": "Ridge linear probe in transformed space with standardized features.",
    }

    class _LinearPredictor(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            W = torch.tensor(cfg["ridge_W"], dtype=torch.float32)
            b = torch.tensor(cfg["ridge_b"], dtype=torch.float32)
            self.W = nn.Parameter(W, requires_grad=False)
            self.b = nn.Parameter(b, requires_grad=False)
            self.mu = torch.tensor(cfg["feat_mu"], dtype=torch.float32)
            self.sd = torch.tensor(cfg["feat_sd"], dtype=torch.float32)

        def forward(self, x):
            x = (x - self.mu) / (self.sd + 1e-6)
            return x @ self.W + self.b

    head = _LinearPredictor(cfg)
    payload = {"state_dict": head.state_dict(), "config": cfg}
    os.makedirs(os.path.dirname(out_head) or ".", exist_ok=True)
    torch.save(payload, out_head)
    click.echo(
        f"[Ridge] Saved linear probe → {out_head}  (alpha={alpha}, in_dim={cfg['in_dim']}, labels={list(labels)})")


if __name__ == '__main__':
    cli()
