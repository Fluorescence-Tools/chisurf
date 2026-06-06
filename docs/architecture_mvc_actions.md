# MVC Action Architecture (KISS Contract)

This document defines the minimum architecture contract for state-changing flows.

## Contract

- UI/plugins/macros should trigger state changes through the shared action
  surface exposed by `chisurf.action_execute(...)` or facade/controller code.
- Structured history must be emitted through the core action/history path, not
  ad hoc from callsites.
- Lifecycle operations (project/dataset/fit/setup) should not rely on
  string-command routing (`chisurf.run("...")`).
- MCP automation must use the same action surface as GUI paths.

## Layering

- `chisurf.core.actions._infra.ActionDispatcher`: route action name -> handler.
- `chisurf.core.actions._infra.ActionRegistry`: action specs and metadata.
- `chisurf.core.actions.*_actions`: small domain action implementations.
- UI/plugins/macros: thin adapters only for migrated paths.

## Core Action Surface

- Project: `project.save`, `project.load`, `project.close`
- Dataset: `dataset.add`, `dataset.remove`, `dataset.group`, `dataset.ungroup`
- Fit: `fit.add`, `fit.close`, `fit.set_dataset`, `fit.run`
- Setup/experiment: `experiment.set`, `setup.select`, `setup.params.set`
- Tooling: `action.catalog.export`

## Review Gate

- New state mutation path must add/route through an action.
- New action should include schema/metadata in `build_default_dispatcher()`.
- History payload should be deterministic and replay-friendly.
