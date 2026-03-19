# MVC Action Architecture (KISS Contract)

This document defines the minimum architecture contract for state-changing flows.

## Contract

- UI/plugins/macros should trigger state changes via
  `chisurf.action_controller.execute(...)`.
- Structured history must be emitted via the runtime action layer
  (`record_action(...)`), not directly from callsites.
- Lifecycle operations (project/dataset/fit/setup) should not rely on
  string-command routing (`chisurf.run("...")`).
- MCP automation must use the same action surface as GUI paths.

## Layering

- `ActionController`: route action name -> handler.
- `controllers/services/*`: small domain services implementing behavior.
- `runtime/actions.py`: registry, validation, dispatch, history recording.
- UI/plugins: thin adapters only.

## Core Action Surface

- Project: `project.save`, `project.load`, `project.close`
- Dataset: `dataset.add`, `dataset.remove`, `dataset.group`, `dataset.ungroup`
- Fit: `fit.add`, `fit.close`, `fit.set_dataset`, `fit.run`
- Setup/experiment: `experiment.set`, `setup.select`, `setup.params.set`
- Tooling: `action.catalog.export`

## Review Gate

- New state mutation path must add/route through an action.
- New action should include schema in `build_default_registry()`.
- History payload should be deterministic and replay-friendly.
