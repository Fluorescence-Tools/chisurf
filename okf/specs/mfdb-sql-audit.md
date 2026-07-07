# MFDB raw-SQL audit (PRD-26: no scattered SQL)

CRUD (select/insert/update/delete) should route through `db.dao` or a centralized
query method. `bespoke` (joins/aggregates/recursive/DISTINCT) and `ddl` may stay raw.

| file | select | insert | update | delete | bespoke | ddl |
|---|--:|--:|--:|--:|--:|--:|
| adapters/chinet.py | 1 | 0 | 0 | 0 | 0 | 0 |
| admin/backend/auth_services.py | 11 | 2 | 3 | 0 | 1 | 0 |
| admin/backend/fluorophore_services.py | 10 | 2 | 5 | 0 | 0 | 0 |
| admin/backend/password_services.py | 4 | 1 | 1 | 0 | 0 | 0 |
| admin/backend/services.py | 32 | 1 | 9 | 3 | 4 | 4 |
| admin/cli/__init__.py | 3 | 0 | 0 | 0 | 0 | 0 |
| admin/seed_example.py | 13 | 8 | 0 | 0 | 0 | 0 |
| api.py | 2 | 0 | 0 | 0 | 0 | 0 |
| lifecycle/event_log.py | 0 | 0 | 0 | 0 | 1 | 0 |
| lifecycle/lifecycle.py | 0 | 2 | 0 | 1 | 1 | 0 |
| lifecycle/staleness.py | 2 | 0 | 0 | 0 | 2 | 0 |
| project/project_archiver.py | 2 | 0 | 0 | 0 | 3 | 0 |
| provenance/graph.py | 5 | 0 | 0 | 0 | 0 | 0 |
| provenance/lineage.py | 2 | 0 | 0 | 0 | 1 | 0 |
| provenance/operation_parameters.py | 1 | 1 | 0 | 1 | 1 | 0 |
| queries/analysis.py | 10 | 3 | 6 | 0 | 1 | 0 |
| queries/artifacts.py | 10 | 8 | 11 | 1 | 1 | 0 |
| queries/branches.py | 3 | 1 | 3 | 0 | 2 | 0 |
| queries/experiments.py | 4 | 3 | 5 | 0 | 1 | 0 |
| queries/lifecycle.py | 3 | 1 | 0 | 0 | 1 | 0 |
| queries/objects.py | 4 | 1 | 2 | 1 | 0 | 0 |
| queries/parameters.py | 0 | 0 | 0 | 0 | 1 | 0 |
| queries/probes.py | 20 | 8 | 7 | 4 | 2 | 9 |
| queries/protocols.py | 4 | 1 | 0 | 0 | 1 | 0 |
| queries/samples.py | 7 | 8 | 4 | 2 | 3 | 0 |
| queries/setups.py | 10 | 6 | 4 | 0 | 2 | 0 |
| queries/studies.py | 6 | 3 | 1 | 0 | 4 | 0 |
| queries/users.py | 4 | 4 | 2 | 0 | 0 | 0 |
| repository.py | 29 | 13 | 0 | 1 | 1 | 4 |
| samples/importer.py | 0 | 3 | 0 | 0 | 0 | 0 |
| samples/reagents.py | 2 | 2 | 0 | 0 | 1 | 0 |
| samples/sample_manager.py | 24 | 0 | 0 | 0 | 3 | 0 |
| samples/seed_data.py | 10 | 1 | 1 | 0 | 1 | 0 |
| schema/dictionary_schema_map.py | 1 | 0 | 0 | 0 | 0 | 0 |
| schema/schema.py | 16 | 14 | 5 | 3 | 3 | 12 |
| schema/schema_from_dictionary.py | 1 | 0 | 0 | 0 | 0 | 0 |
| security/auth.py | 7 | 5 | 7 | 0 | 3 | 0 |
| store/database_resolver.py | 1 | 0 | 0 | 0 | 0 | 1 |
| store/transactions.py | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 264 | 102 | 76 | 17 | 45 | 30 |
