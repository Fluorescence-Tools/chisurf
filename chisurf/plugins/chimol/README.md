# Chimol (MolView) protein viewer

Chimol is the integrated ChiSurf 3D protein viewer (class name `MolView`).
It provides fast cartoon/backbone rendering, simple selections, and
sequence/structure synchronization inside the ChiSurf GUI or as a
standalone Qt application.

## Features

- Cartoon, trace, atoms, sticks, surface, and dots render modes with
  ambient occlusion and configurable profiles.
- Integrated sequence panel for residue selection, coloring, and alignment.
- Basic selections: click-pick residue/atom, rectangular drag-select.
- Trajectory support (mdtraj), MRC map loading, and RMF frame loading
  when optional dependencies are installed.
- Secondary-structure assignment using a lightweight DSSP-like algorithm
  (no external DSSP binary required).
- Runs inside ChiSurf or as a standalone Qt app (`python -m chisurf.plugins.chimol`).

## Quickstart (ChiSurf)

1. Launch ChiSurf and open **Plugins → Structure → Structure → ChiMOL**.
2. Click **Open** and choose a PDB/mmCIF/trajectory.
3. Toggle representations (Cartoon/Atoms/Sticks/Trace/Dots/Surface) from the toolbar.
4. Color by secondary structure or sequence gradient using the color buttons.
5. Use the sequence panel to select residues; selections sync to 3D.

## Quickstart (standalone)

```bash
python -m chisurf.plugins.chimol
```

## Dependencies

- Required: Python 3.9+, Qt (via `qtpy`), NumPy.
- Optional:
  - `mdtraj` for trajectory loading and DSSP comparison tests.
  - `PyOpenGL` for explicit GL entry points (QtGL renderer).
  - `numba` to speed up secondary-structure assignment (falls back to NumPy).
  - `IMP` (via `chisurf.fio.structure.coordinates`) for structure IO when available.

## Configuration

- Display parameters (cartoon thickness, colors, AO strength, etc.) are read from
  `chimol_display.json`. Preferred location: the ChiSurf settings directory
  (`chisurf.settings.get_path("settings")`). Fallback: a JSON file next to
  `config.py`. Legacy `molview_display.json` or `protview_display.json` filenames
  are also accepted for backward compatibility.

## Developer notes

- Tests live under `chisurf/plugins/chimol/tests/`. To run a quick subset:
  ```bash
  pytest chisurf/plugins/chimol/tests -k chimol
  ```
  The `mdtraj`-based DSSP comparison test is skipped automatically if `mdtraj`
  or an external DSSP binary is unavailable.
- Key modules:
  - `chimol/app/molview_main_window.py`: UI wiring (Qt docks, sequence, object list).
  - `chimol/renderer/qtgl.py`: Qt native OpenGL renderer.
  - `chimol/geometry/cartoon.py`: cartoon geometry generation.
  - `chimol/config.py`: display configuration loading and defaults.
- For standalone development, ensure `QT_API` is set (e.g., `PySide6` or `PyQt5`).
- Keep user-visible strings using the Chimol name; class names remain `MolView`
  for API compatibility.
