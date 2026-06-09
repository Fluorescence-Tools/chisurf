# Chimol TODO & Command Parity Tracker

This document tracks general development tasks, command parity, structural, and lifecycle TODOs for Chimol that are separate from the render-quality/visual upgrade plan.

## Command Parity (vs. PyMOL)

Track the status of PyMOL-style commands and utilities. For a detailed design plan of the command subsystem, see [CMD_PLAN.md](../../chisurf/plugins/chimol/CMD_PLAN.md).

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
