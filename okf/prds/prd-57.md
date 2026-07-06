---
type: PRD
prd: "57"
title: "PRD-57: ChiMOL Command Parity and Renderer Migration"
description: Grow the ChiMOL molecular viewer's cmd surface toward parity with a reference molecular-graphics command language, and migrate its renderer to an immediate-mode GUI backend while holding a Qt-free controller/scene contract.
status: draft
phase: "feature track"
resource: chisurf/plugins/chimol/
tags: [prd, plugins, structure, viewer]
timestamp: '2026-07-06T00:00:00Z'
---

# Summary
ChiMOL (formerly moview/ProtView) provides an in-tree molecular viewer with a
command line modelled on the de-facto molecular-graphics `cmd` API. This PRD grows
that command surface incrementally toward parity with that reference language, and
sets the long-term direction for the renderer: swap the hand-rolled Qt/GL label and
console machinery for an immediate-mode GUI backend, while keeping the controller
and scene Qt-free so both backends consume one shared contract. Current-state
plugin facts live in the [ChiMOL profile](/plugins/profiles/chimol.md); this
concept is the forward roadmap.

# Status
Draft / in-progress. Tiers 0–3 of the command roadmap have landed (distinct camera
verbs; `byelement`/`bychain`/`spectrum`/named colors; a full recursive-descent
selection grammar in `cmd/sele_parser.py`; per-atom representation masks; persistent
3D measurement overlays). The renderer already separates a Qt-free scene from
swappable backends (`renderer/scene.py`, `base.py`, `qtgl.py`, `raytracer.py`); no
immediate-mode backend exists yet. Tiers 4–5 and the renderer migration phases below
are future work.

# Command parity roadmap (tiers)

- **Tier 0 — baseline (done).** Command shell + `@script` execution, IO
  (`help`/`load`/`open`/`fetch`/`objects`/`get_names`), viewer toggles, color
  pipeline, residue-range selections and geometric measurements. Feature-complete;
  polish/bugfix only.
- **Tier 1 — low-effort polish (done).** Reference-aligned help/usage text; color
  synonyms and named-color table; distinct `center`/`orient`/`zoom`/`reset`
  semantics with a saved view; measurement overlays instead of console-only output.
- **Tier 2 — selections & per-object control (done for the grammar).** A tokenizer/
  parser/evaluator supporting `all`/`none`/`resi`/`resn`/`name`/`chain`/`elem`/
  `and`/`or`/`not`/`within`/`around`/`byres`/`expand` and `/object/chain/resi/name`
  macros; per-selection `show`/`hide`/`color`; named-selection state; broader
  `set`/`get` bridging remains.
- **Tier 3 — representation depth & chemistry-light editing.** Per-atom visual
  overrides as a first-class channel; full representation mapping (lines/sticks/
  spheres/dots/mesh/surface/cartoon/ribbon) with per-object toggles; read-only
  `alter`/`iterate` label workflows; pseudoatoms; object grouping. Partially landed
  (per-atom masks, overlays).
- **Tier 4 — structural editing & new objects (future).** Chemistry edits
  (`bond`/`unbond`/`h_add`/`rebond`) behind a dedicated backend with undo/redo;
  volume/map objects (`map_new`/`isomesh`/`isosurface`); sandboxed `alter`-style
  expression evaluation.
- **Tier 5 — movies, wizards, analysis (long-term).** Timeline/keyframe animation;
  guided-workflow plugin framework; structural analysis (`align`/`super`/`cealign`/
  `rms`/`intra_fit`).

Keep unimplemented command names **registered** so the CLI emits a friendly "not
yet implemented" message instead of `KeyError`. When porting a command, re-read the
reference implementation's block first to understand hidden side effects.

# Renderer migration (immediate-mode GUI backend)

Long-term goal: an immediate-mode GUI backend implementing the `Renderer` ABC, which
removes the fragile `QPainter.begin(self)`-inside-`paintGL` pattern, enables headless
rendering/testing without a `QApplication`, and replaces the hand-rolled label
system and `QLineEdit` command bar. A pre-built cross-platform binding that can
coexist with Qt in-process is the intended vehicle; a windowing/GL-context library
drives the standalone shell.

Phased plan (renderer):

| Phase | Goal | Status |
|---|---|---|
| 9.0 | Controller code Qt-free; `scene.py` Qt-free | done by design |
| 9.1 | `renderer/imgui_renderer.py` implementing `Renderer` via the GL-context library | future |
| 9.2 | Immediate-mode command console replaces the `QLineEdit` bar | future |
| 9.3 | Object-panel sidebar (tree node per object, color swatch) | future |
| 9.4 | World-space overlays via the backend draw list (replace `QPainter` labels) | future |
| 9.5 | Extract a `CameraState` dataclass; route mouse via the backend IO | future |
| 9.6 | Optional: embed the GL window in the Qt shell | future |

# Design constraints that apply now

These hold during all current ChiMOL work so the migration stays cheap (also in the
[ChiMOL profile](/plugins/profiles/chimol.md)):

1. No Qt in controller code (`MolView` and its mixins import without a
   `QApplication`).
2. `scene.py` stays Qt-free — `Geometry`/`SceneObject`/`Scene` are plain dataclasses,
   the shared contract for every backend.
3. All overlays go through `SceneObject(kind="text", ...)`; both the Qt/GL and
   offline raytracer backends render from the same scene.
4. `qtgl.py` is a thin translator — camera math, shader uniforms, event handling
   only; no business logic.
5. Avoid new `QPainter.begin(self)` uses (removed in phase 9.4).
6. New camera attributes go on simple fields for later extraction into a
   `CameraState` dataclass (phase 9.5).

# Relationships
- Forward roadmap for the plugin documented in the
  [ChiMOL profile](/plugins/profiles/chimol.md); the viewer is exposed as a
  structure/molecular-viewer tool in [Modelling plugins](/plugins/modelling.md) and
  scripted via `chimol-cli` ([Macros, CLI & scripting](/subsystems/macros-cli.md)).
- Manifest category drift is tracked as [assessment INC-07](/specs/assessment.md).
