# MFDB Admin Status

Last reviewed: 2026-07-06.

## Current State

- `manifest.json` declares `mfdb_admin` version `1.1.0`.
- GUI entrypoint: `mfdb.admin.gui.tool:MFDBWidget`.
- Service entrypoint: `mfdb.admin.backend.services:register_services`.
- The plugin has broad test coverage under `test/` for handlers, views, auth/session,
  navigation, and optical-component workflows.
- GUI confidence is tracked in `docs/GUI_TEST_COVERAGE.md`. Construction-only
  panels are explicitly marked untested for behavior.
- Backend services use temporary database patching in tests and configured
  `resolve_database_path()` in normal runtime.
- MFDB source is vendored at `modules/mfdb/src/mfdb`; MFDB Admin app code lives
  under `modules/mfdb/src/mfdb/admin`; `chisurf.core.mfdb` is a transitional
  facade that aliases loaded submodules to avoid duplicate class objects. MFDB
  Admin, Database Connector, and Project Browser now import the vendored `mfdb`
  package directly in their active backend/test paths.
- MFDB dictionary files (`*.dic`) live under `modules/mfdb/src/mfdb/data/`; the
  old `chisurf/core/mfdb/data` copy has been removed.
- MFDB runtime paths, object-store root, and default user are resolved through
  `mfdb.config`; ChiSurf publishes startup values via `MFDB_*` environment
  variables instead of being imported by the vendored package.
- The fluorophore reference `spectra.db` used for seed/reference imports is
  bundled in `modules/mfdb/src/mfdb/data/` and no longer resolved from the
  ChiSurf `_dev` plugin path by default.
- MFDB project archiving encodes curve arrays locally and no longer imports
  ChiSurf experiment serialization helpers.

## Implemented Surfaces

| Surface | Status |
| --- | --- |
| MFDB admin GUI | Implemented. |
| ZMQ/in-process client wrapper | Implemented through `MFDBClient`. |
| AutoForm entity detail forms | Implemented through `modules/mfdb/src/mfdb/admin/gui/autoform_entity_form.py`. |
| AutoForm JSON view specs | Implemented for connection/auth and optical-component detail forms; expand to remaining ordinary forms/tables. |
| Main `mfdb.*` handlers | Implemented across `modules/mfdb/src/mfdb/admin/backend/services.py` and helper modules. |
| Versioned `mfdb.v1.*` core API adapter | Implemented and auth-gated before delegation. |
| Vendored MFDB package import boundary | Implemented for `modules/mfdb/src/mfdb`, with `test/core/test_mfdb_vendor_package.py` covering canonical `mfdb` imports and the transitional `chisurf.core.mfdb` facade. |
| Legacy `sample_database.*` aliases | Removed from the supported MFDB Admin contract. |
| Database connector services | Implemented and covered by temporary-MFDB direct/in-process tests. |
| Project browser services and GUI smoke | Implemented and covered by sample-data-backed service tests plus headless Qt tree construction. |
| MFDB Admin core EntityDock GUI smoke | Implemented for Samples, Sample Conditions, Entities, Probes, Label Positions, FRET Pairs, Experiments, Experiment Types, Setups, Detector Channels, PIE Windows, FCS Pairs, Devices, Users, Raw Data, Processing Runs, Processed Products, Analyses, Objects, Projects, and Branches with sample-data-backed headless GUI interaction tests. Sample Conditions, Experiment Types, Devices, Branches, and Users additionally cover New-button creation, auto-save update, checked-row delete, and delete confirmation where applicable. Users also cover rename and built-in-user delete protection. Raw Data and Processed Products additionally cover Copy ID, Reveal/Open, provenance-seed, validation-status update, and confirmed soft-delete actions through the visible EntityDock. Analyses additionally cover Copy ID, Details drilldown with parameter/output-product tables, and provenance-seed actions. Objects additionally cover Copy UUID, Reveal, and confirmed delete through the visible Objects EntityDock. |
| Object-store handlers | Implemented through `mfdb.objects.*`. |
| Dataset browse/open handlers | Implemented, with default-user local GUI fallback. |
| Fluorophore curation handlers | Registered from `backend/fluorophore_services.py`. |
| CLI | Implemented under `mfdb.admin.cli`; ChiSurf keeps a compatibility wrapper. |

## Manifest Coverage

`manifest.json` declares every method registered by
`mfdb.admin.backend.services:register_services`. This is enforced by
`test/core/test_mfdb_vendor_package.py::test_mfdb_admin_manifest_declares_registered_methods`.

## Risks

- Destructive operations are present: delete handlers, ACL changes, backup/reset,
  import/export writes, object deletion, and branch/operation status mutation.
- Error shapes are not uniform across all handlers.
- Auth/ACL behavior differs by method family and by local GUI fallback path.
- The retired `sample_database` plugin directory has been deleted. Active callers
  should use the ChiSurf `mfdb_admin` shim and canonical `mfdb.admin` /
  `mfdb.*` / `mfdb.v1.*` services.
- The old MFDB Admin `backend/setup_services.py` compatibility module was removed;
  canonical setup handlers are registered as `mfdb.setups.*` from
  `modules/mfdb/src/mfdb/admin/backend/services.py`.
- Many MFDB Admin panels are still hand-written Qt. Ordinary form/table/detail
  surfaces should be migrated to AutoForm and JSON view specs.
- Many MFDB Admin GUI panels are construction-tested only. Manual testing has
  shown failures in GUI elements, so anything marked construction-only or untested
  in `docs/GUI_TEST_COVERAGE.md` must not be treated as done.
- The vendored `mfdb` package still contains ChiSurf-specific integration imports
  in `mfdb.chinet_adapter` for fit-state serialization and ChiNet parameter
  metadata. It is source-separated, but not yet independently clean or
  publishable.

## Recommended Next Work

1. Add request/response schemas for the highest-risk mutating methods in
   `docs/CONTRACT.md`.
2. Keep the deleted `sample_database` plugin out of active manifests, tests, and
   plugin-loader paths; do not reintroduce compatibility wrappers.
3. Review auth/ACL behavior method family by method family.
4. Migrate manual MFDB Admin UI panels to AutoForm. Use the shared
   `TableSection` primitive for read-only browse tables and add any missing
   primitives to `chisurf/gui/autoform` instead of local MFDB-only widgets.
5. Keep the `core/database_connector` and `core/project_browser` plugin-local
   tests in the MFDB smoke set; they now prove the non-legacy database path.
6. Continue converting the GUI coverage ledger into tests, next targeting
   analysis non-empty input-product/grouped-fit drilldowns, raw/processed
   file-content preview paths, user password/force-delete flows, non-branch
   entity mutation paths, and the remaining aggregate panels.

## Verification

Full plugin-local suite:

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/mfdb_admin/test
```

Last run on 2026-07-06 with the arm64 Qt environment: 111 passed, 2 warnings.

AutoForm-focused checks:

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest test/gui/test_autoform_table_section.py chisurf/plugins/core/mfdb_admin/test/test_autoform_entity_form.py chisurf/plugins/core/mfdb_admin/test/test_connection_dialog.py
```

Current focused smoke set:

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  test/core/test_mfdb_vendor_package.py \
  test/gui/test_autoform_table_section.py \
  chisurf/plugins/core/mfdb_admin/test/test_admin_navigation.py \
  chisurf/plugins/core/mfdb_admin/test/test_autoform_entity_form.py \
  chisurf/plugins/core/mfdb_admin/test/test_connection_dialog.py \
  chisurf/plugins/core/mfdb_admin/test/test_entity_dock_gui_interactions.py \
  chisurf/plugins/core/mfdb_admin/test/test_session_sso.py \
  chisurf/plugins/core/database_connector/test \
  chisurf/plugins/core/project_browser/test
```

Last run on 2026-07-06 with the arm64 Qt environment: 58 passed, 2 warnings.

MFDB Admin plus adjacent database surfaces:

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  test/core/test_mfdb_vendor_package.py \
  chisurf/plugins/core/mfdb_admin/test \
  chisurf/plugins/core/database_connector/test \
  chisurf/plugins/core/project_browser/test
```

Last run on 2026-07-06 with the arm64 Qt environment: 133 passed, 2 warnings.

GUI-oriented plugin-local subset excluding `test_admin_handlers.py`:

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  chisurf/plugins/core/mfdb_admin/test \
  --ignore=chisurf/plugins/core/mfdb_admin/test/test_admin_handlers.py
```

Last run on 2026-07-06 before restoring SQLite-backed structured sample creation:
88 passed, 2 warnings. Prefer the full plugin-local suite now that it is green.

Documentation drift check to run manually when changing services:

```bash
python3 - <<'PY'
import json, re
from pathlib import Path
root = Path("chisurf/plugins/core/mfdb_admin")
declared = {item["name"] for item in json.loads((root / "manifest.json").read_text())["rpc_methods"]}
registered = set()
for path in (root / "backend").glob("*.py"):
    registered.update(re.findall(r"dispatcher\\.register\\(\\s*['\\\"]([^'\\\"]+)['\\\"]", path.read_text()))
print("registered static not in manifest:")
for name in sorted(registered - declared):
    print(name)
PY
```
