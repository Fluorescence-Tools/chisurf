# PRD-23: Thin Widgets / View–API Separation (Architecture F)

## Goal

GUI widgets become pure **view**: no data processing, no DB/`tttrlib` calls, no
side effects on construction. All state and I/O live behind the api/RPC layer.
Mandatory construction smoke tests and a shared dockable-tool base remove the
class of bugs where logic hidden in widgets ships broken.

## Evidence (why)

- A missing `QComboBox` import crashed `DetectorWizardPage` at construction — never
  caught because the widget-build path had no test.
- The FCS dialog wrote to MFDB **on construction** (migrating/registering as a
  side effect), causing the "auth required" / write-on-open issues.
- Each tool re-implements Load/Save, drag-drop, docks differently (shifter vs
  others), so fixes don't propagate.

## Design

- **View-only widgets.** Widgets call the api/RPC client only; no `tttrlib`, no
  `MFDatabase`, no registration logic in `gui/`. Construction is read-only and
  cannot perform DB writes (the PRD-16 split, enforced).
- **Mandatory construction smoke test** per tool (mirror
  `test/gui/test_detector_wizard_page.py`): build the widget offscreen, assert it
  constructs — catches missing-import / side-effect-on-init regressions.
- **Shared dockable-tool base** (`ChisurfDockTool` or similar): standard Load/Save
  tool-menu, drag-drop target, dock management, MFDB-connection status, and the
  "From MFDB / Save to MFDB vs file" connectivity-aware behaviour — implemented
  once, reused by every transformer tool.
- Lean on the existing MVC controller work (AGENT/TODO reports MVC complete) so
  state lives in the model/controller, not the widget.

## Tasks

1. Define `ChisurfDockTool` base (tool-menu, docks, drag-drop, MFDB status,
   connectivity-aware load/save) and migrate the Microtime Shifter + FCS dialog +
   detector wizard onto it.
2. Move any residual `tttrlib`/DB logic out of `gui/` into `api/` + RPC.
3. Add construction smoke tests for every tool; make it a checklist item for new
   transformers (PRD-16 conformance).
4. Forbid DB writes during widget `__init__` (read-only construction); migration/
   registration only on explicit user action.

## Definition of Done

- [ ] Widgets are view-only (no DB/`tttrlib`); construction is read-only.
- [ ] A shared dockable-tool base is used by the transformer tools; Load/Save/docks
      are not re-implemented per plugin.
- [ ] Every tool has a construction smoke test; new transformers must add one.

## Definition of Clean

Layer purity (view ↔ api/RPC); no side effects on construction; GUI smoke tests
mandatory; reuse the base, don't fork.

## Relationship

Enforces the GUI half of PRD-16 (transformer contract). Generalizes the fixes made
to the shifter/FCS dialog this session. Incremental — apply per tool.
