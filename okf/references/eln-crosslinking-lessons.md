---
type: Reference
title: ELN crosslinking & information-management lessons (eLabFTW)
description: What eLabFTW's auth, entity-linking, tagging and metadata model teaches MFDB — what was adopted and what is deferred.
tags: [mfdb, provenance, crosslinking, metadata, eln, lessons]
timestamp: '2026-07-09T00:00:00Z'
---

# Reference

Lessons distilled from studying a mature electronic-lab-notebook (eLabFTW) to
inform MFDB's information-management and crosslinking model. External software is
described by role, not name, per OKF convention (the studied system is a
widely-used open-source ELN with local + LDAP/SSO auth and a rich entity model).

## Auth (already applied, see [PRD-59](/prds/prd-59.md))
The ELN validates MFDB's design: search-then-bind LDAP, group/team reconciliation
on every login, JIT-vs-match-only provisioning, and — notably — **message parity**
for unknown-user vs wrong-password (a single generic "invalid credentials"),
which confirms MFDB was right to **skip timing-equalization**. Patterns worth a
later look: multiple LDAP login attributes (uid *or* mail), transparent password
rehash-on-login, and restricting local login to admins ("break-glass") when a
directory is the configured default.

## Crosslinking — MFDB is architecturally ahead
The ELN uses a **per-pair junction matrix** (~25 `X2YLinks` tables). MFDB's single
**polymorphic `mfdb_edge`** (`source/target node_type+id`, `relationship_type`,
`operation_id`, `metadata_json`) subsumes that and adds provenance semantics.
Backlinks are covered by the directional graph traversal
(`graph_upstream`/`downstream`). Keep MFDB's single-edge model; do **not** copy
the junction matrix.

## Information management — the philosophy difference
The ELN uses per-entity freeform typed `extra_fields` (type/value/options/groups).
MFDB uses **`.dic`-dictionary-driven typed vocabulary + a lifecycle state
machine**. Keep MFDB's dictionary approach (better for a standardized databank).

## Adopted (2026-07-09)
- **Metadata → edge materialization** (the ELN's "a metadata field that
  references another entity *is* a link"): a metadata value of the form
  `mfdb://<node_type>/<node_id>` on a sample/experiment key-value now materializes
  an idempotent `mfdb_edge` (`linked_to`, tagged with the originating key). See
  `queries/artifacts.py::_materialize_metadata_ref` / `parse_node_ref`.
- **Resolvable audit labels** on edge/link creation (`add_edge`,
  `record_operation_link`) — the target node's human title is recorded in the
  audit `details` (`_resolve_node_label`), mirroring the ELN's clickable-anchor
  changelog.

## Ideas for later (not yet done)
- **Tags** — a lightweight polymorphic tag layer (`mfdb_tag` / `mfdb_tag_member`)
  with user/team/everything **scope** and AND-search across tags, for cross-cutting
  discovery over the provenance graph. The ELN's `Tags2Entity` (`item_type`,
  `item_id`, `tag_id`; `HAVING COUNT(DISTINCT tag_id) = :n` for AND-search) is the
  reference shape. Optional — only if ELN-style browsing is wanted; MFDB is
  provenance-first, not a general ELN.
- **Copy-edges-on-clone** — a generic helper to copy a node's edges when it is
  duplicated (the ELN's `duplicate()` on links); MFDB currently handles bulk copy
  only via `project_archiver`.
- Extend metadata→edge materialization to `analysis_metadata` once its canonical
  node type is settled.
