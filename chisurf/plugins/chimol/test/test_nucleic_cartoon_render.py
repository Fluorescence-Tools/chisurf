"""Headless verification of the DNA/RNA (nucleic) cartoon geometry.

This test exercises ``_generate_nucleic_cartoon_arrays`` directly (the mesh
builder used by ``MolView`` for DNA/RNA) without a live OpenGL context or a Qt
widget, so it runs anywhere numpy + matplotlib are available.

It:

* loads ``cartoon.py`` (and its sibling ``ambient.py``) by file path so the
  Qt-importing ``chimol`` package ``__init__`` is not pulled in;
* builds the same inputs ``MolView`` passes (full atom array, scaled/centered
  coordinates, per-residue trace ids, per-residue colors) from the FRET docking
  example ``fps_hiv_rt/dna.pdb`` (DA/DC/DG/DT + modified 2DA, mixed ' / * sugar
  naming);
* renders before/after PNGs with an offscreen matplotlib renderer for visual
  inspection; and
* asserts the mesh is non-degenerate, that the backbone tube honours
  ``backbone_radius`` (guards against the historical double-scale bug), and that
  smoothing reduces backbone curvature.

Run just this file:  ``pytest chisurf/plugins/chimol/test/test_nucleic_cartoon_render.py -q``
Produce the PNGs:     ``python chisurf/plugins/chimol/test/test_nucleic_cartoon_render.py``
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

# --- Paths -----------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_GEOM = _HERE.parent / "chimol" / "geometry"
_DNA_PDB = (
    _HERE.parent.parent
    / "modelling"
    / "fret"
    / "examples"
    / "fps_hiv_rt"
    / "dna.pdb"
)
_RENDERS = _HERE / "renders"


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_cartoon_module():
    """Load cartoon.py standalone and inject the AO helper it expects."""
    ambient = _load_by_path("_chimol_ambient_standalone", _GEOM / "ambient.py")
    cartoon = _load_by_path("_chimol_cartoon_standalone", _GEOM / "cartoon.py")
    # cartoon.py's ``from .ambient import ...`` fails under file-path loading and
    # falls back to None; wire the real implementation back in.
    cartoon._estimate_ambient_occlusion = ambient._estimate_ambient_occlusion
    return cartoon


def _build_inputs(scale: float = 10.0):
    """Return (atoms, coords_all, res_ids, chain_ids, colors) like MolView."""
    from chisurf.core.structure import Structure

    struct = Structure(str(_DNA_PDB))
    atoms = struct.atoms
    xyz = np.asarray(atoms["xyz"], dtype=float)
    center = xyz.mean(axis=0)
    coords_all = (xyz - center) * scale

    # Per-residue trace in file order (unique (res_id, chain)).
    res_ids: list[int] = []
    chain_ids: list[str] = []
    seen: set[tuple] = set()
    rid_all = np.asarray(atoms["res_id"])
    ch_all = np.asarray(atoms["chain"]).astype(str)
    for k in range(len(atoms)):
        key = (int(rid_all[k]), ch_all[k].strip())
        if key not in seen:
            seen.add(key)
            res_ids.append(int(rid_all[k]))
            chain_ids.append(ch_all[k].strip())
    res_ids_arr = np.asarray(res_ids)
    chain_ids_arr = np.asarray(chain_ids, dtype=object)

    # Per-residue colour gradient (Nx4 RGBA).
    n = len(res_ids_arr)
    t = np.linspace(0.0, 1.0, max(n, 1))
    colors = np.column_stack([t, 0.4 + 0.5 * (1.0 - t), 1.0 - t, np.ones(n)])
    return atoms, coords_all, res_ids_arr, chain_ids_arr, colors, scale


# Config that reproduces the OLD (pre-fix) appearance: bloated double-scaled
# backbone (0.1*scale^2 == 1.0*scale after the fix), P-first trace, no smoothing,
# no tension, no AO.
_BEFORE_CFG = {
    "backbone_radius": 1.0,  # -> 10 scene units, the historical bloated tube
    "nucleic_trace_atoms": ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "C1*"],
    "backbone_smooth_cycles": 0,
    "spline_tension": 0.0,
    "nucleic_ao_strength": 0.0,
    "ring_style": "filled",  # old flat plates (no rounded rim)
}


def _tri_area_sum(verts: np.ndarray, faces: np.ndarray) -> float:
    tris = verts[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _render_png(cartoon, cfg, inputs, png_path: pathlib.Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    atoms, coords_all, res_ids, chain_ids, colors, scale = inputs
    v, n, f, c = cartoon._generate_nucleic_cartoon_arrays(
        atoms, coords_all, res_ids, chain_ids, colors,
        config={**cfg, "coordinate_scale": scale},
    )
    tris = v[f]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
    lam = np.clip(fn @ np.array([0.3, 0.4, 0.85]), 0.0, 1.0)
    base = c[f[:, 0]][:, :3] if c is not None else np.full((len(f), 3), 0.7)
    facecol = np.clip(base * (0.25 + 0.75 * lam)[:, None], 0.0, 1.0)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tris, facecolors=facecol, edgecolors="none")
    ax.add_collection3d(coll)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    ctr = (lo + hi) / 2.0
    rad = float((hi - lo).max()) / 2.0 or 1.0
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-62)
    ax.set_axis_off()
    ax.set_title(title)
    _RENDERS.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return v, n, f, c


@pytest.mark.skipif(not _DNA_PDB.is_file(), reason="dna.pdb example not present")
def test_nucleic_cartoon_before_after_and_asserts():
    pytest.importorskip("matplotlib")
    cartoon = _load_cartoon_module()
    inputs = _build_inputs()

    before_png = _RENDERS / "nucleic_before.png"
    after_png = _RENDERS / "nucleic_after.png"
    _render_png(cartoon, _BEFORE_CFG, inputs, before_png, "DNA cartoon — before")
    v, n, f, c = _render_png(cartoon, {}, inputs, after_png, "DNA cartoon — after")

    # --- Non-degeneracy ---
    assert v.shape[0] > 0 and f.shape[0] > 0
    assert int(f.max()) < len(v)
    assert np.isfinite(v).all() and np.isfinite(n).all()
    nn = np.linalg.norm(n, axis=1)
    nz = nn > 1e-6
    assert np.allclose(nn[nz], 1.0, atol=1e-3)

    # --- Radius honoured (guards the double-scale regression) ---
    atoms, coords_all, res_ids, chain_ids, colors, scale = inputs
    areas = []
    for r in (0.1, 0.4, 0.8):
        vv, _, ff, _ = cartoon._generate_nucleic_cartoon_arrays(
            atoms, coords_all, res_ids, chain_ids, colors,
            config={"coordinate_scale": scale, "backbone_radius": r,
                    "nucleic_ao_strength": 0.0},
        )
        areas.append(_tri_area_sum(vv, ff))
    assert areas[0] < areas[1] < areas[2], f"tube area not monotone in radius: {areas}"

    # --- Smoothing reduces backbone curvature ---
    # Use the trace points the builder would collect (first available trace atom).
    trace = []
    rid_all = np.asarray(atoms["res_id"])
    ch_all = np.asarray(atoms["chain"]).astype(str)
    names = np.char.upper(np.char.strip(np.asarray(atoms["atom_name"]).astype(str)))
    prio = ["C4'", "C4*", "C3'", "C3*", "C5'", "C5*", "O5'", "O5*",
            "P", "O3'", "O3*", "C1'", "C1*"]
    for i, rid in enumerate(res_ids):
        m = (rid_all == rid) & (np.char.strip(ch_all) == str(chain_ids[i]).strip())
        if not np.any(m):
            continue
        lut = {nm: co for nm, co in zip(names[m], coords_all[m])}
        for cand in prio:
            if cand in lut:
                trace.append(lut[cand])
                break
    trace = np.asarray(trace, dtype=float)
    assert trace.shape[0] >= 4

    def _turning(p):
        d = np.diff(p, axis=0)
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        cosang = np.clip((d[:-1] * d[1:]).sum(axis=1), -1.0, 1.0)
        return float(np.arccos(cosang).sum())

    raw = _turning(cartoon._smooth_backbone_points(trace, cycles=0))
    smooth = _turning(cartoon._smooth_backbone_points(trace, cycles=2))
    assert smooth <= raw + 1e-6, f"smoothing increased curvature: {smooth} > {raw}"

    print(f"\nnucleic cartoon renders:\n  {before_png}\n  {after_png}")
    print(f"areas(r=0.1,0.4,0.8) = {areas}")
    print(f"turning raw={raw:.3f} smoothed={smooth:.3f}")


if __name__ == "__main__":
    _cartoon = _load_cartoon_module()
    _inputs = _build_inputs()
    _render_png(_cartoon, _BEFORE_CFG, _inputs, _RENDERS / "nucleic_before.png",
                "DNA cartoon — before")
    _render_png(_cartoon, {}, _inputs, _RENDERS / "nucleic_after.png",
                "DNA cartoon — after")
    print("wrote:", _RENDERS / "nucleic_before.png", _RENDERS / "nucleic_after.png")
