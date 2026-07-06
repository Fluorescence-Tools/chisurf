---
type: Plugin Profile
title: Project Browser plugin
description: OKF profile for MFDB-backed project browsing and archival.
resource: chisurf/plugins/core/project_browser/
tags: [plugins, projects, mfdb, provenance]
timestamp: '2026-07-06T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `project_browser` |
| Display name | `Tools:Open Project` |
| Categories | `Tools`, `Project` |
| Version | `1.0.0` |
| State namespace | `project_browser` |
| Local README | `chisurf/plugins/core/project_browser/README.md` |

The manifest describes browsing, saving, restoring, exporting, and importing ChiSurf
projects using MFDB-backed version control.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| GUI | `gui/tool.py`, `gui/client.py`. |
| Backend services | `backend/services.py` with project listing, save/restore, export/import, branch, graph, artifact, parameter, and delete handlers. |
| Tests | `test/test_project_browser_services.py` and `test/test_gui_headless.py` cover sample-data-backed services, in-process dispatch, headless Qt tree construction, GUI restore/delete/import actions, branch DAG behavior, artifact/parameter browsing, collision-remapped import, and delete permission enforcement. |

Manifest RPC methods include `project_browser.list`, `save`, `restore`,
`export_csp`, `import_preview`, `import_csp`, `delete_version`, `create_branch`,
`list_branches`, `version_graph`, `artifacts`, and `parameters`.

# Data And Provenance Impact

This plugin can create and restore project payloads, export/import `.csp` archives,
delete project versions, and manipulate branch/version graph state. The backend also
collects project dependencies and reconstructs payloads. That makes it a P0 docs
target for project persistence and provenance.

# Verification Surface

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  chisurf/plugins/core/project_browser/test
```

Current coverage proves save/list/restore/export with a tiny CSV-backed project
payload, in-process service registration, GUI construction, GUI restore into a
patched ChiSurf context, GUI delete and collision-remapped import actions, import
collision rejection/remapping, branch DAG roots/leaves, artifact browsing,
fit-parameter listing, and delete manage-permission enforcement.

Latest verification: Project Browser plugin-local suite reported 11 passed / 1
warning; the focused GUI/admin/database smoke reported 55 passed / 2 warnings; the
broad MFDB Admin + Database Connector + Project Browser smoke reported 130 passed /
2 warnings.

Remaining gap: branch/artifact/parameter endpoints are service-covered and
client-covered but do not yet have dedicated visible panels inside
`ProjectBrowserTool`.

# Documentation Work

- Expand `chisurf/plugins/core/project_browser/README.md` with concrete payload
  examples for version numbering and branch behavior.
