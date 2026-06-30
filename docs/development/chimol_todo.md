# Chimol TODO & Command Parity Tracker

This document tracks general development tasks, command parity, structural, and lifecycle TODOs for Chimol that are separate from the render-quality/visual upgrade plan.

## Command Parity (vs. PyMOL)

Track the status of PyMOL-style commands and utilities. For a detailed design plan of the command subsystem, see [CMD_PLAN.md](../../notes/CMD_PLAN.md) (local-only dev note).

### Completed Commands
- [x] `split_chains` - Split structures into per-chain objects (moved from rendering mixin to lifecycle mixin, using `viewer.split_chains`).
- [x] `distance`, `angle`, `dihedral` - Resolution of selections and drawing overlay objects.
- [x] Camera verbs - `center`, `orient`, `zoom`, `reset`.
- [x] Color commands - `byelement`, `bychain`, `spectrum`, and named colors.

### Pending Parity Tasks
- [ ] Add `group` / `ungroup` commands to organize objects hierarchically (Tier 3).
- [ ] Implement read-write versions of `alter`, `alter_state`, and `iterate` commands to support residue/atom property manipulation.
- [ ] Add support for `map_new`, `isomesh`, and `isosurface` for volumetric density maps (Tier 4).
- [ ] Implement chemical editing commands (`bond`, `unbond`, `h_add`) in coordination with backend topology.

## General TODOs

- [ ] Complete the migration to Dear ImGui as the primary UI shell for standalone mode (see Phase 9.0+ roadmap in `CMD_PLAN.md`).
- [ ] Implement undo/redo queues for structural editing.
- [ ] Fix mouse interaction mode switch for middle-mouse panning. A setting was added (`camera.mouse_mode` in `chimol_display.json`, default `"pymol"`) and a toolbar toggle was wired up in `ControlsToolbar` / `MolViewPluginWindow`. The left-drag rotation inversion works as intended, but middle-drag panning still does not reliably switch between PyMOL-style (object follows cursor) and Chimol-style (camera/plane follows cursor). Attempts tried:
  - Inverting pan deltas in `_pan_from_delta()` via a mode multiplier. This produced opposite cursor behavior but did not match the expected "camera vs. object" semantics in the running viewer.
  - Reverting to shared pan behavior and then re-introducing a separate `_pan_delta_multiplier()`. The setting reads correctly and the multiplier returns the expected `1.0`/`-1.0`, yet the on-screen pan direction does not consistently reflect the selected mode.
  - Verified against PyMOL open-source (`layer1/SceneMouse.cpp`, `layer5/PyMOL.cpp`, `layer1/SceneView.cpp`): PyMOL's 3-Button Viewing maps middle button to `cButModeTransXY`, which translates `m_pos` in camera space so the object follows the cursor. Chimol's existing `_pan_from_delta()` is mathematically equivalent when no multiplier is applied, so the remaining issue is likely in how the view matrix / pan offset is constructed or how the mode is propagated to the renderer at runtime.
  - Next step when revisiting: instrument `_pan_from_delta()` and `_build_matrices()` with the current mode/delta/center values, or refactor panning to use a separate camera-space offset (analogous to PyMOL's `m_pos`) instead of folding it into the scene center.

## Architectural Principles

### IMP-Inspired Design Patterns
- Follow IMP-like hierarchical model architecture: Particle → Decorator → Constraint hierarchy
- Use tree-based node structures for molecular representations (similar to IMP::Algebraics)
- Implement bounding box caching at each hierarchy level for efficient culling
- Separate data model from visualization representation (model-view pattern)

### Extensibility
- Plugin architecture for new representation types (cartoon, surface, volume)
- Strategy pattern for field functions (Gaussian, Wyvill, custom kernels)
- Observer pattern for configuration changes and view updates
