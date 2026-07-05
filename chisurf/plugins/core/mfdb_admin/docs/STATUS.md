# MFDB Admin Status

Last reviewed: 2026-07-05.

## Current State

- `manifest.json` declares `mfdb_admin` version `1.1.0`.
- GUI entrypoint: `chisurf.plugins.core.mfdb_admin.gui.tool:MFDBWidget`.
- Service entrypoint: `chisurf.plugins.core.mfdb_admin.backend.services:register_services`.
- The plugin has broad test coverage under `test/` for handlers, views, auth/session,
  navigation, and optical-component workflows.
- GUI confidence is tracked in `docs/GUI_TEST_COVERAGE.md`. Construction-only
  panels are explicitly marked untested for behavior.
- Backend services use temporary database patching in tests and configured
  `resolve_database_path()` in normal runtime.

## Implemented Surfaces

| Surface | Status |
| --- | --- |
| MFDB admin GUI | Implemented. |
| ZMQ/in-process client wrapper | Implemented through `MFDBClient`. |
| AutoForm entity detail forms | Implemented through `gui/autoform_entity_form.py`. |
| AutoForm JSON view specs | Implemented for connection/auth and optical-component detail forms; expand to remaining ordinary forms/tables. |
| Main `mfdb.*` handlers | Implemented across `backend/services.py` and helper modules. |
| Versioned `mfdb.v1.*` core API adapter | Implemented and auth-gated before delegation. |
| Legacy `sample_database.*` aliases | Removed from the supported MFDB Admin contract. |
| Database connector services | Implemented and covered by temporary-MFDB direct/in-process tests. |
| Project browser services and GUI smoke | Implemented and covered by sample-data-backed service tests plus headless Qt tree construction. |
| Object-store handlers | Implemented through `mfdb.objects.*`. |
| Dataset browse/open handlers | Implemented, with default-user local GUI fallback. |
| Fluorophore curation handlers | Registered from `backend/fluorophore_services.py`. |
| CLI | No CLI command declared in manifest; `cli/` is only a package placeholder. |

## Known Drift

The current manifest is not a complete list of statically registered backend
methods.

Registered but not declared in `manifest.json`:

- `mfdb.auth.change_password`
- `fluorophores.ai_triage`
- `fluorophores.approve`
- `fluorophores.find_duplicates`
- `fluorophores.forster_radius.lookup`
- `fluorophores.get`
- `fluorophores.get_spectra_batch`
- `fluorophores.import_reference_set`
- `fluorophores.list`
- `fluorophores.merge`
- `fluorophores.probe_types.list`
- `fluorophores.reject`
- `fluorophores.set_quality`

Also review manifest coverage for dynamically registered method families in
`backend/services.py`, especially lifecycle, protocols, studies, reagents,
calibrations, pipelines, setup definitions, datasets, and object-store methods.

## Risks

- Destructive operations are present: delete handlers, ACL changes, backup/reset,
  import/export writes, object deletion, and branch/operation status mutation.
- Error shapes are not uniform across all handlers.
- Auth/ACL behavior differs by method family and by local GUI fallback path.
- The hidden `sample_database` directory still exists as a deprecated marker and can
  be deleted once remaining wrapper imports are audited.
- The old MFDB Admin `backend/setup_services.py` compatibility module was removed;
  canonical setup handlers are registered as `mfdb.setups.*` from
  `backend/services.py`.
- Many MFDB Admin panels are still hand-written Qt. Ordinary form/table/detail
  surfaces should be migrated to AutoForm and JSON view specs.
- Many MFDB Admin GUI panels are construction-tested only. Manual testing has
  shown failures in GUI elements, so anything marked construction-only or untested
  in `docs/GUI_TEST_COVERAGE.md` must not be treated as done.
- The full MFDB Admin plugin-local suite does not pass in the current arm64 Qt
  environment because structured sample tests import `chisurf.core.mfdb.orm`,
  which requires `sqlalchemy`. Current result: 80 passed, 3 failed, 6 errors.

## Recommended Next Work

1. Synchronize `manifest.json` with registered auth and fluorophore methods.
2. Add request/response schemas for the highest-risk mutating methods in
   `docs/CONTRACT.md`.
3. Delete or quarantine the retired `sample_database` wrapper directory after
   confirming no plugin loader or external launch path needs it.
4. Add a manifest-drift regression test that fails when statically registered method
   names are absent from `manifest.json`, or explicitly records intentional
   exceptions.
5. Review auth/ACL behavior method family by method family.
6. Migrate manual MFDB Admin UI panels to AutoForm. Use the shared
   `TableSection` primitive for read-only browse tables and add any missing
   primitives to `chisurf/gui/autoform` instead of local MFDB-only widgets.
7. Keep the `core/database_connector` and `core/project_browser` plugin-local
   tests in the MFDB smoke set; they now prove the non-legacy database path.
8. Convert the GUI coverage ledger into tests, starting with sample-data-backed
   `EntityDock` interactions for samples, experiments, data products, objects,
   projects, and users.

## Verification

Focused suite:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/mfdb_admin/test
```

AutoForm-focused checks:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest test/gui/test_autoform_table_section.py chisurf/plugins/core/mfdb_admin/test/test_autoform_entity_form.py chisurf/plugins/core/mfdb_admin/test/test_connection_dialog.py
```

Current focused smoke set:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  test/gui/test_autoform_table_section.py \
  chisurf/plugins/core/mfdb_admin/test/test_admin_navigation.py \
  chisurf/plugins/core/mfdb_admin/test/test_autoform_entity_form.py \
  chisurf/plugins/core/mfdb_admin/test/test_connection_dialog.py \
  chisurf/plugins/core/mfdb_admin/test/test_session_sso.py \
  chisurf/plugins/core/database_connector/test \
  chisurf/plugins/core/project_browser/test
```

Last run on 2026-07-05 with the arm64 Qt environment: 23 passed, 2 warnings.

Full plugin-local suite status in the same environment:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  chisurf/plugins/core/mfdb_admin/test
```

Last run on 2026-07-05: 80 passed, 3 failed, 6 errors. The failing/erroring tests
are structured-sample paths in `test_admin_handlers.py` that require
`sqlalchemy`, which is not installed in the arm64 Qt environment.

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
