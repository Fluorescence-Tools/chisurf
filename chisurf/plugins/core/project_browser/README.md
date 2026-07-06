# Project Browser

MFDB-backed project persistence plugin for browsing, saving, restoring, exporting,
importing, and versioning ChiSurf projects.

## Runtime Surface

The plugin exposes a Qt GUI entrypoint and backend services:

| Surface | Entry point |
| --- | --- |
| GUI | `chisurf.plugins.core.project_browser.gui.tool:ProjectBrowserTool` |
| Services | `chisurf.plugins.core.project_browser.backend.services:register_services` |

## RPC Contract

Read-only or browse-oriented methods:

- `project_browser.list`
- `project_browser.list_branches`
- `project_browser.version_graph`
- `project_browser.artifacts`
- `project_browser.parameters`
- `project_browser.import_preview`

Mutating or filesystem-writing methods:

- `project_browser.save`
- `project_browser.restore`
- `project_browser.export_csp`
- `project_browser.import_csp`
- `project_browser.delete_version`
- `project_browser.create_branch`

All project data is stored through MFDB operations, artifacts, object-store blobs,
branch heads, and provenance edges. Dataset payloads are decomposed into source-file
objects where available and processed-data JSON artifacts.

## GUI Behavior

`ProjectBrowserTool` builds an in-process `ProjectBrowserClient` by default. The tree
lists project groups with child version rows. Selecting a project restores the latest
version; selecting a child restores that exact version.

## Verification

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  chisurf/plugins/core/project_browser/test
```

The plugin-local tests cover:

- sample-data-backed save/list/restore/export using a temporary MFDB and object store;
- import preview rejection plus collision-remapped import from an exported `.csp`
  archive;
- branch creation, branch-scoped version counts, and version DAG roots/leaves;
- artifact and fit-parameter browsing for a version with sample data and a local fit;
- manage-permission enforcement for deleting project versions;
- in-process RPC registration through `ServiceDispatcher` and `InProcessClient`;
- headless Qt construction of the project tree with sample project/version data;
- headless Qt construction using the real in-process ChiSurf services against a
  temporary MFDB seeded with a CSV-backed sample project;
- headless Qt restore of a selected version into a patched ChiSurf context, verifying
  project payload load and active project metadata updates;
- headless Qt delete confirmation and collision-remapped `.csp` import actions.
