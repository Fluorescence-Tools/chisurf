# MFDB Admin RPC Contract

This file documents the MFDB Admin service surface at the family level. The
authoritative discoverability list is `manifest.json`; implementation lives in
`backend/services.py` and the service modules it registers.

## Conventions

| Topic | Contract |
| --- | --- |
| Transport | JSON-RPC through ChiSurf's service dispatcher. |
| GUI client | `gui/client.py:MFDBClient`. |
| Service registration | `backend/services.py:register_services`. |
| Database | Handlers open `MFDatabase(resolve_database_path())` unless delegated to `chisurf.core.mfdb.api`. |
| Auth | Most methods accept optional `auth: dict`. Versioned `mfdb.v1.*` methods require auth before delegation. |
| Errors | Some handlers raise exceptions, some return `{"error": ...}` or service-error dictionaries. Do not assume one uniform error shape yet. |
| Mutations | Methods ending in `save`, `delete`, `create`, `register`, `record`, `transition_status`, `chmod`, `chown`, `chgrp`, `grant`, `revoke`, `backup`, `reset_from_source`, `import_file`, and `put` are mutating or side-effectful. |

## Declared Method Families

| Family | Count | Purpose | Side effects |
| --- | ---: | --- | --- |
| `mfdb.status` | 1 | Database path/schema/count summary. | Read-only. |
| `mfdb.samples.*` | 9 | Sample browse, get, save, delete, search, structured sample helpers, export validation. | Mixed read/write/delete. |
| `mfdb.sample_conditions.*` | 2 | Sample-condition get/save. | Mixed read/write. |
| `mfdb.entities.*` | 3 | Sample entity list/save/delete. | Mixed read/write/delete. |
| `mfdb.probes.*` | 6 | Probe list/get/save, optical properties, and position listing. | Mixed read/write. |
| `mfdb.fret_pairs.*` | 3 | FRET pair list/save/delete. | Mixed read/write/delete. |
| `mfdb.pdbx.*` | 2 | PDBx key suggestion and value validation. | Read-only. |
| `mfdb.mock_data.populate` | 1 | Populate demonstration/mock MFDB data. | Writes database state. |
| `mfdb.users.*` | 3 | Legacy user list/save/delete. | Mixed read/write/delete. |
| `mfdb.devices.*` | 3 | Device list/save/delete. | Mixed read/write/delete. |
| `mfdb.experiment_types.*` | 3 | Experiment-type list/save/delete. | Mixed read/write/delete. |
| `mfdb.experiments.*` | 7 | Experiment list/get/save/delete, key-value save, data save/delete. | Mixed read/write/delete. |
| `raw_data.*` | 3 | Raw-data registration/list/get. | Mixed read/write. |
| `processed_data.*` | 3 | Processed-data registration/list/get. | Mixed read/write. |
| `processing.burst_selection.*` | 4 | Burst-selection record/run/get/list. | Mixed read/write/compute. |
| `provenance.*` | 2 | Provenance edge listing and processed-data trace. | Read-only. |
| `archive.*` | 1 | Export burst-processing archive manifest. | Writes export file/payload. |
| `mfdb.import_file` | 1 | Import structure/database file. | Writes database state. |
| `mfdb.export_sample` | 1 | Export a sample. | Writes output file/payload. |
| `mfdb.export_table` | 1 | Export a table. | Writes output file/payload. |
| `mfdb.backup` | 1 | Create database backup. | Writes backup file. |
| `mfdb.reset_from_source` | 1 | Reset user database from source database. | Destructive database mutation. |
| `mfdb.auth.*` | 5 declared | Login/logout/current user and session list/revoke. | Mixed read/write/session mutation. |
| `mfdb.groups.*` | 8 | Group CRUD and group membership management. | Mixed read/write/delete. |
| `mfdb.permissions.*` | 6 | ACL get/chmod/chown/chgrp/grant/revoke. | Mixed read/write. |
| `mfdb.objects.*` | 6 | Object-store put/get/delete/list/info. | Mixed object-store and database mutation. |
| `mfdb.v1.*` | 37 | Versioned core MFDB API: samples, experiments, artifacts, operations, graph, parameters, ChiNet sessions, setups, audit, branches, user branch state. | Mixed; auth enforced before delegation. |

Additional `mfdb.lifecycle.*`, `mfdb.protocols.*`, `mfdb.studies.*`,
`mfdb.reagents.*`, `mfdb.calibrations.*`, `mfdb.pipelines.*`,
`mfdb.setups.*`, and `mfdb.datasets.*` methods are registered from
`backend/services.py`; keep `manifest.json` synchronized with these registrations.

## Versioned MFDB API

`mfdb.v1.*` methods are delegated to `chisurf.core.mfdb.api` through
`VERSIONED_MFDB_METHODS` in `backend/services.py`.

| Prefix | Purpose |
| --- | --- |
| `mfdb.v1.samples.*` | Register/get/list canonical samples. |
| `mfdb.v1.experiments.*` | Register/get/list canonical experiments. |
| `mfdb.v1.artifacts.*` | Register/get/list artifacts. |
| `mfdb.v1.operations.*` | Record/list/get operations, link artifacts, link operations, transition operation status. |
| `mfdb.v1.graph.*` | Traverse/export upstream/downstream provenance graphs. |
| `mfdb.v1.parameters.*` | Record/get/list operation parameters. |
| `mfdb.v1.chinet.sessions.*` | Save/get/list/restore ChiNet sessions. |
| `mfdb.v1.setups.*` | Save/get/list setup definitions. |
| `mfdb.v1.audit.*` | List audit logs. |
| `mfdb.v1.branches.*` | Create/fork/get/list/update/delete provenance branches. |
| `mfdb.v1.users.*` | Active-branch and operation-jump state for users. |

Contract rule: callers should supply `auth` for all `mfdb.v1.*` calls. The adapter
opens the configured database, checks auth, then calls the core API handler.

## Retired Sample Database Namespace

`sample_database.*` is retired. ChiSurf is still prerelease, so compatibility aliases
are intentionally not part of the supported contract. New callers must use `mfdb.*`
or `mfdb.v1.*`.

Contract rule: do not add new `sample_database.*` methods. Remove remaining internal
call sites instead of preserving the old namespace.

## Auth, Groups, And Permissions

`backend/auth_services.py` registers:

- `mfdb.auth.login`
- `mfdb.auth.logout`
- `mfdb.auth.me`
- `mfdb.auth.change_password`
- `mfdb.auth.sessions.list`
- `mfdb.auth.sessions.revoke`
- `mfdb.groups.*`
- `mfdb.groups.members.*`
- `mfdb.permissions.*`

Status caveat: `mfdb.auth.change_password` is registered but not declared in the
current manifest. Fix the manifest before treating the manifest as a complete auth
contract.

## Fluorophore Curation

`backend/services.py` imports and registers fluorophore services from
`backend/fluorophore_services.py`.

Static registration currently includes:

- `fluorophores.list`
- `fluorophores.get`
- `fluorophores.get_spectra_batch`
- `fluorophores.probe_types.list`
- `fluorophores.find_duplicates`
- `fluorophores.merge`
- `fluorophores.set_quality`
- `fluorophores.approve`
- `fluorophores.reject`
- `fluorophores.import_reference_set`
- `fluorophores.forster_radius.lookup`
- `fluorophores.ai_triage`

Status caveat: these methods are registered but not declared in the current manifest.
Add them to `manifest.json` or move them behind a clearly documented plugin boundary.

## Object Store And Datasets

`mfdb.objects.*` exposes content-addressed object storage operations through the MFDB
repository layer. `mfdb.datasets.browse` and `mfdb.datasets.open` are registered in
`backend/services.py` and use the configured active/default user to scope local GUI
access.

Contract rule: object UUIDs are public identifiers at the service boundary; callers
should not depend on object-store MD5 paths.

## Compatibility And Drift Checks

Keep these checks green when editing service registration or the manifest:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/mfdb_admin/test/test_admin_handlers.py
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/mfdb_admin/test/test_session_sso.py
```

When changing method registration, also compare registered static method names with
`manifest.json` and update both this file and `docs/STATUS.md`.
