---
type: Plugin Profile
title: Database Connector plugin
description: OKF profile for core database connector services.
resource: chisurf/plugins/core/database_connector/
tags: [plugins, mfdb, database, rpc]
timestamp: '2026-07-06T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `database_connector` |
| Display name | `Core:Database Connector` |
| Categories | `Core`, `Database`, `Fluorescence` |
| Version | `0.1.0` |
| Local README | `chisurf/plugins/core/database_connector/README.md` |

The manifest describes source/user database resolution, migration, backup, reset,
repository access, and FLR CIF import/export.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| Services | `services.py` with `DatabaseConnector`, `register_services`, and handlers for status/open/close/backup/reset/repository/import/export. |
| GUI | None exposed by manifest. |
| CLI | None exposed by manifest. |
| Tests | `test/test_database_connector_services.py` covers temporary-MFDB handler counts, explicit open path status, in-process RPC registration, backup, reset-from-source, minimal CIF import, and FLR CIF text/file export. |

Manifest RPC methods are `database_connector.status`, `open`, `close`, `backup`,
`reset_from_source`, `repository`, `import_file`, and `export_sample`.

# Data And Provenance Impact

This is P0 because backup, reset, import, and export operations can affect real user
database state. The plugin is service-only, so the RPC contract is the user-facing
contract for other plugins.

# Verification Surface

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  chisurf/plugins/core/database_connector/test
```

Current coverage includes both non-destructive paths and side-effect paths against
temporary databases/files: backup, reset-from-source, import, and FLR CIF export.
Destructive semantics still need to remain explicit in docs and callers should keep
using temporary databases for automated reset/import/export tests.

# Documentation Work

- Extend the README into a compact request/response contract table when the RPC
  schema stabilizes.
- Mark destructive operations explicitly: reset, import, export overwrite behavior.
- Explain which database path is considered source versus user database.
