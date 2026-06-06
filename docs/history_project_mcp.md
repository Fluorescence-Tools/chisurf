# History, Project Persistence, and MCP Action Routing

This note documents the current action-routing path used for operation history,
project save/load metadata, and MCP-style external control.

## Current Routing Model

- State-change and observability events are emitted through `record_action(...)`
  in `chisurf.core.actions._infra`.
- `record_action(...)` first routes through `chisurf.action_dispatcher`
  (`ActionDispatcher.execute`).
- If dispatcher execution fails (for example, unknown legacy action name),
  `record_action(...)` falls back to direct history recording so events are not
  silently dropped.

## Action Names and MCP Aliases

- Canonical internal action IDs use underscore style, for example:
  - `project_save`
  - `project_load`
  - `fit_run_start`
- MCP-style dotted aliases are accepted by dispatcher execution:
  - `project.save` resolves to `project_save`
  - `fit.run.start` resolves to `fit_run_start`
- Emitted history events use canonical action IDs.

## Runtime Accessors

- `chisurf.action_catalog()`
  - Returns discoverable action metadata for adapters/tools.
  - Includes `name`, `mcp_name`, payload `schema`, replay/dedupe metadata.
- `chisurf.action_execute(name, payload, summary, source_uid, target_uid)`
  - Executes actions through the shared dispatcher path.
  - Accepts canonical or MCP-style action names.
- `chisurf.core.api.ChiSurfAPI`
  - Facade entrypoint for dataset, fit, parameter, project, session, model, and graph operations.
  - Supports `local`, `hybrid`, and `server` modes.

## Command Export (JSON/YAML)

- Action catalog export is available as a macro command:
  - `chisurf.macros.export_action_catalog(target_path="action_catalog.yaml", file_type="yaml")`
  - `chisurf.macros.export_action_catalog(target_path="action_catalog.json", file_type="json")`
- Action execution supports the same operation where the action is registered:
  - `chisurf.action_execute("action.catalog.export", {"target_path": "action_catalog.yaml", "file_type": "yaml"})`
- Format can be selected by `file_type` (`yaml`/`json`) or inferred from file
  suffix when possible.

## Macro Entry Alignment

- Dataset macros now delegate to controller actions by default and use internal
  `_from_controller` guards to avoid recursive dispatch.
- This keeps direct macro/script calls aligned with the same routed action path
  used by GUI entrypoints.
- Additional experiment/plugin callsites now use controller routing for dataset
  and fit creation intents, reducing residual string-command macro entry paths.

## MCP Experiment Loading Control

- Registered actions can set experiment and setup before loading data where those
  actions are present in the action registry.
- Use `chisurf.action_catalog()` to inspect the current action surface before
  calling an action from automation.
- Nested setup attributes are supported with dotted keys (for example
  `"noise_model.weight_type"`), which helps configure FCS noise settings,
  PDA time-window settings (`minimum_time_window_length`), and similar setup
  controls before dataset load.

## Project Save/Load Metadata

- Project and fit save flows embed an action-catalog snapshot into
  `proj.extra["action_catalog"]`.
- Project and fit load events include `history_loaded` payload flags to make
  persisted-history restore status explicit in the operation log.

## Scope Status

- Direct `history.record(...)` usage is centralized through
  `chisurf.core.actions._infra`.
- Remaining work is event-coverage completion: ensuring all state mutations
  emit routed actions (Phase 9 remains `[PARTIAL]`).
