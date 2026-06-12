# Chimol Render Upgrade Plan

## Goal
Upgrade Chimol render quality to match/exceed PyMOL while keeping ChiSurf working, with minimal Qt/PyQt coupling in the geometry/model/scene layers to allow a future C++/ImGui port.

## Architecture Principles
- **Geometry, representation builders, scene description, and material model** must be free of Qt imports.
- **Qt is only used** in the concrete render backend (`qtgl.py`), the widget controller (`view.py`), and the renderer base class (`base.py`).
- Each representation (cartoon, sticks, spheres, surface) produces backend-neutral `Geometry` + `SceneObject` with optional `Material`.
- Render passes: opaque → transparent (sorted) → overlay → labels.

## Current State (Before)

| Feature | Chimol | PyMOL |
|---------|--------|-------|
| Sticks | GL_LINES, no lighting | Cylinder mesh with per-vertex colors (half-bond), Phong-lit |
| Spheres | Fixed low-res UV sphere (10×20), merged mesh | Instanced/impostor spheres with proper normals |
| Cartoon | Tube/ribbon mesh, Catmull-Rom spline | Extrude + CGO, similar quality |
| Surface | Gaussian metaballs → marching cubes OR AO-shaded points | Mesh surface with proper SES/SAS |
| Render passes | All geometry rendered in submission order | Opaque → transparent sorted → overlay |
| Materials | Per-object dict passed to shader | Full material system (ambient/diffuse/spec/transparency) |
| Export | Screen capture only | Ray-traced / high-res offscreen |

## Milestone Plan

### Milestone 1: Materials + Render Passes [DONE]
**Files changed:** `scene.py`, `qtgl.py`

- [x] Add `Material` dataclass to `scene.py` with fields: `ambient`, `diffuse`, `specular`, `shininess`, `opacity`, `metallic`, `roughness`, `rim_strength`, `rim_power`
- [x] Make `SceneObject.material` typed as `Optional[Material]` instead of `Optional[Dict]`
- [x] Add render pass sorting in `QtGLRenderer.paintGL`: collect calls by `render_mode`, draw opaque first, then transparent (back-to-front sorted by depth), then overlay
- [x] Enhance fragment shader to accept per-material uniforms (already partially done; ensure `material` dict is always applied)
- [x] Add `blending_mode` field to material for future use

### Milestone 2: High-Quality Spheres [DONE]
**Files changed:** `primitives.py`, `geometry/__init__.py`, `view.py`

- [x] Increase sphere mesh resolution and cache it
- [x] Add instanced sphere rendering path (GL instancing or merged mesh)
- [x] Keep fallback to point-sprite impostor for very large atom counts

### Milestone 3: Cylinder Sticks [DONE]
**Files changed:** `geometry/__init__.py`, `view.py`

- [x] `_build_stick_mesh` already exists in `primitives.py` — export it from `geometry/__init__.py`
- [x] In `view.py:_update_sticks`, replace GL_LINES path with cylinder mesh path using `_build_stick_mesh`
- [x] Keep lines as a performance fallback for >X bonds

### Milestone 4: Cartoon Quality [LATER]
**Files changed:** `cartoon.py`, `view.py`
for that look at pymol code: /Users/tpeulen/dev/pymol-open-source

- [ ] Split cartoon builder into sampler → profile → extruder stages (like PyMOL's `RepCartoon`)
- [ ] Add proper arrow heads for beta strands
- [ ] Improve smooth interpolation at SS transitions

### Milestone 5: Mesh Surface [DONE]
**Files changed:** `surface.py`, `view.py`

- [x] ~~Current `_update_surface` returns points — change to mesh when possible~~ (Marching cubes mesh generated from Gaussian density is now implemented)
- [x] Add proper SAS/SES computation (EDT-based Solvent Surface using scipy EDT and marching cubes)
- [x] ~~Keep Gaussian metaballs as optional fallback~~ (Metaballs are now fully rendered as meshes with proper normal calculations and ambient occlusion)

### Milestone 6: PyMOL Compat Widget [LATER]
**Files changed:** `MolView.py` (legacy), `view.py`

- [ ] Replace `chisurf/gui/plots/molview/MolView.py` with Chimol compat wrapper

### Milestone 7: High-Quality Export [LATER]
**Files changed:** `qtgl.py`, `view.py`

- [ ] Offscreen FBO rendering at arbitrary resolution with MSAA
- [ ] PNG/TIFF export with transparent background option


## Detailed File-by-File Plan

### `scene.py` — Material dataclass
```python
@dataclass
class Material:
    ambient: float = 0.4
    diffuse: float = 0.6
    specular: float = 0.3
    shininess: float = 40.0
    opacity: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5
    rim_strength: float = 0.2
    rim_power: float = 2.0
    blending: str = "normal"  # "normal" | "additive"
```

### `qtgl.py` — Render pass ordering
```python
def _render_pass(self, calls, mode):
    """mode is 'opaque', 'transparent', or 'overlay'"""
    filtered = [c for c in calls if c.render_mode == mode]
    if mode == 'transparent':
        filtered.sort(key=lambda c: c.depth, reverse=True)  # back-to-front
    for call in filtered:
        self._draw_call(call)
```

### `view.py:_update_sticks` — Cylinder mesh
Replace the GL_LINES block (lines 2633-2646) with a call to `_build_stick_mesh`:

```python
mesh = _build_stick_mesh(bonds, pts_all, atom_colors, radius=sticks_radius)
if mesh:
    verts, norms, faces, cols = mesh
    geom = Geometry(kind="mesh", positions=verts, indices=faces, normals=norms, colors=cols)
    return [SceneObject(id="sticks", geometry=geom, render_mode="opaque")]
```

## Implementation Order

1. `scene.py` — Add `Material` dataclass
2. `geometry/__init__.py` — Export `_build_stick_mesh`
3. `view.py` — Wire cylinder sticks into `_update_sticks`
4. `qtgl.py` — Add render pass sorting
5. Test: `python -m chisurf` and verify sticks show as cylinders

## Testing
- No automated tests for visual output; manual inspection
- Run `python -m chisurf` to load a PDB and toggle `show_sticks`
- All existing call sites (`proteinmc.py`, `proteinMC.py`) must keep working
