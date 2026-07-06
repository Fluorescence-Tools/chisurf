---
type: Subsystem
title: GUI & AutoForm
description: The Qt application and the data-driven AutoForm UI framework that renders view.json schemes.
resource: chisurf/gui/
tags: [gui, qt, autoform, viewspec]
timestamp: '2026-07-05T00:00:00Z'
---

# GUI

`chisurf/gui/` is the Qt application: the main window, widgets, plots
(pyqtgraph), resources, and GUI startup helpers. It is launched via
`python -m chisurf` (`pixi run chisurf`).

# AutoForm

`chisurf/gui/autoform/` renders UI declaratively from JSON view schemes
(`*.view.json`) instead of hand-built Qt widgets. This is part of the PRD-40
model/UI split, backed by data specs in `chisurf/core/dataspec/`.

Notable section types the framework supports include `image` (3D stacks with
click-pick / markers / ROI), `waterfall` (RGB time-vs-µtime images),
`path_list` (drag-drop file/folder lists), and `wizard`/`info`/`embed`.
Sections take a `description` field mapped to widget tooltips, and manifest
`rpc_methods` can be rendered as forms via `AutoForm.from_rpc_method`.

When touching GUI code, prefer porting hand-built widgets to AutoForm + a
JSON view scheme.

# Node editor widget

`chisurf/gui/widgets/node_editor/` is a self-contained, dependency-light node
editor (PyQt + optional `networkx`) with model/view/editor layering. It has a
**versioned JSON graph schema** (`{version, nodes:[{id,type,title,inputs,outputs,
config,pos,collapsed}], edges:[{source,source_port,target,target_port}]}`),
validate-before-mutate loading (`NodeGraphValidationError`, no partial scene on
failure), non-GUI file/dict serialization, a **node-type registry** replacing
hardcoded type lists, and DAG utilities (`is_directed_acyclic`, cycle
highlighting, optional `enforce_acyclic` edge guard). Round-trip and cycle
behaviour are covered by tests under its `tests/` directory. It is the substrate
the visual burst-programming canvas builds on (PRD-29).

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)
