# Database Connector

Service-only core plugin for resolving, inspecting, backing up, resetting, importing,
and exporting the active MFDB database.

## Runtime Surface

The manifest exposes only backend services:

| RPC method | Side effect |
| --- | --- |
| `database_connector.status` | Read-only database path and count summary. |
| `database_connector.open` | Opens a database connection; accepts an explicit `database_path` for tests/advanced callers. |
| `database_connector.close` | Closes the active connection. |
| `database_connector.backup` | Writes a backup copy of the user database. |
| `database_connector.reset_from_source` | Replaces the user database from the curated source database after taking a backup when possible. |
| `database_connector.repository` | Read-only schema/count summary. |
| `database_connector.import_file` | Imports a PDBx/mmCIF, PDB-IHM, or FLR CIF file into MFDB. |
| `database_connector.export_sample` | Exports a sample as FLR CIF text or file. |

The plugin does not provide a GUI. Other plugins should call these services through
the ChiSurf RPC client or an in-process dispatcher in tests.

## Safety Model

`status` and `repository` are non-destructive. `backup`, `reset_from_source`,
`import_file`, and file-based `export_sample` can write user data or filesystem
outputs and should be tested against temporary databases.

`sample_database` is retired. New callers should use this connector plus canonical
MFDB services (`mfdb.*` / `mfdb.v1.*`) rather than `sample_database.*`.

## Verification

```bash
PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." python3 -m pytest \
  chisurf/plugins/core/database_connector/test
```

The plugin-local tests seed a temporary MFDB with sample, user, device, experiment
type, and experiment rows, then verify direct handlers, in-process RPC dispatch,
backup, reset-from-source, minimal CIF import, and FLR CIF text/file export.
