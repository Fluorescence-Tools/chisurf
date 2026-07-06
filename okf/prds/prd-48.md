---
type: PRD
prd: "48"
title: "PRD-48: Provider-Agnostic ELN Integration"
description: A provider-agnostic ELN integration layer for MFDB with a single gateway abstraction and two concrete electronic-lab-notebook backends, supporting bidirectional deposit, import, and reconciliation.
status: draft
phase: "unassigned"
resource: chisurf/core/mfdb/eln/
tags: [prd, mfdb, eln]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
MFDB is the prototype implementation for a planned public fluorescence databank, so it must interoperate with the community's electronic lab notebook (ELN) platforms rather than live as an island. This PRD specifies a provider-agnostic `ElnGateway` abstraction under `chisurf/core/mfdb/eln/`, with a neutral entity model (Record, Resource, Chemical, Instrument, Link, Scope) and two concrete adapters plus a declared capability set so callers degrade gracefully. Integration is bidirectional — push (deposit records + attachments + links), pull (import chemicals/resources/instruments with CAS/InChIKey dedup), and conflict reconciliation — keeping a clean split between deposition and DOI dissemination. The guiding invariant is one identity per real-world object; users and groups are matched, never provisioned, and credentials live in the OS store per PRD-37.

# Status
Draft / unassigned (STATUS TABLE authoritative). Design complete; phased 1–4 from prime-backend push through import, secondary backend, and reconciliation/dissemination/GUI.

# Goal / Motivation
ChiSurf's **MFDB** (`chisurf/core/mfdb/`) is the prototype implementation for a planned **public fluorescence databank for chemical biology** under a national research-data initiative (grant in preparation). For that vision the databank must interoperate with the community's electronic lab notebooks, not live as an island.

This PRD specifies a **provider-agnostic ELN integration layer**: a single `ElnGateway` abstraction with two concrete backends.

- **The prime target — a chemistry-native national ELN.** It is the initiative's flagship ELN (institute-hosted, publicly funded), chemistry-native (molecules with InChIKey / canonical SMILES / chemical registry number), carries a rich *analysis → dataset → attachment* tree ideal for measured spectroscopy, models instruments with **DOIs** (device metadata), and feeds a public repository / DOI-minting dissemination path. Its data model and mission align with the planned databank.
- **The secondary target — an institutional ELN.** A general experiment + resource + booking model, useful where a lab already runs it.

Both are reached over their REST APIs. The integration is **bidirectional** (push deposition + pull import + reconciliation) and keeps a clean separation between **deposition** (writing the working record into an ELN) and **dissemination** (minting DOIs / publishing to a repository) — for the prime backend the latter is a first-class, later-phase capability; for the secondary backend it does not exist.

The guiding invariant is **one identity per real-world object** — a fluorophore, sample, instrument, or operator is the same thing in MFDB and in the ELN.

# Background — what already exists

## MFDB seams (`chisurf/core/mfdb/`)
`api.py` is transport-agnostic ("the API is just functions", per the access-layer strategy) and is the layer a sync engine drives. Entry points (all take `auth` → `principal_from_rpc_auth`): `register_artifact()` (~ln 60, has `url`+`metadata`), `record_operation()` (~ln 218), `record_operation_with_artifacts()` (~ln 303), `register_sample()` (~ln 440), `register_experiment()` (~ln 485), `record_operation_link()` (~ln 576), `export_graph()` (~ln 690).

- `models.py` — `MfdbOperation.metadata_json`, `MfdbArtifact.url` + `.metadata_json`, `SampleDefinition.extra`, reagent lots, `MfdbSetup`, `flr_instrument`.
- `object_store.py` — content-addressed (md5, dedup): the byte source for uploads.
- `payload_codec.py` / `payload_models.py` — typed payloads (`tcspc_decay`, `spectra`, `fcs_correlation`, `anisotropy_curve`, `pda_histogram`, `burst_table`) → natural ELN *datasets*.
- `events.py` — `EVENT_ARTIFACT_REGISTERED`, `EVENT_STATE_CHANGED` (async push).
- `auth/`, `session.py`, `credentials.py` — `Principal`/`SessionContext`, `flr_sample_users`, `mfdb_group`; OS credential store (PRD-37) for ELN tokens.
- `project_archiver.py` (`archive_project_to_mfdb`, ~ln 88) — the "decompose a workflow into operations+artifacts+edges" template; push mirrors it in reverse.

## Prime backend (chemistry-native national ELN)
- **API:** REST, base `/api/v1`; an OpenAPI/swagger doc at `/api/v1/swagger_doc`. No canonical Python SDK.
- **Auth:** a personal bearer API token (`Authorization: Bearer <token>`, minted via a token endpoint); JWT and session also accepted.
- **Entities:** samples (+ a chemistry-native **molecule** entity carrying InChIKey, InChI string, canonical SMILES, chemical registry number, sum formula), reactions, research plans, collections (sharing via per-collection shares with a permission level + per-element detail levels), and a polymorphic **container** tree (container root → analyses → analysis → dataset) with attachments on containers.
- **Identity:** single-table inheritance on users (Person / Group / Admin); membership via a users-groups join. Match key: email.
- **Instruments:** devices + device descriptions + **device metadata** (DOI, landing page, manufacturers, DOI-registration fields). **No booking model** (device-analyses only loosely schedule analyses).
- **External ids / metadata:** molecule InChIKey / registry number / canonical SMILES, a sample cross-reference JSON field, container extended-metadata, research-plan metadata (DOI / alternate identifier / related identifier), element-tag data — all viable back-reference stores.
- **Dissemination:** research-plan / device DOIs + a public repository / DOI-registration path.

## Secondary backend (institutional ELN)
- **API:** REST v2, base `/api/v2`; `Authorization: <userid>-<key>` (keys carry `can_write`, scoped to one `(user, team)`); an official Python SDK.
- **Entities:** experiments, items (resources), compounds (registry number / InChI / SMILES / external compound id), multipart uploads, entity extra-field metadata, entity links; **bookable items** + team events (booking with an experiment binding); users / teams / a users-teams join.

# Design

## 1. Provider-agnostic `ElnGateway`
A new subpackage **`chisurf/core/mfdb/eln/`** isolates all ELN knowledge; `api.py` stays transport-agnostic. One neutral interface, two adapters, a declared capability set so callers degrade gracefully where a backend lacks a feature:

```
chisurf/core/mfdb/eln/
  __init__.py        # ElnGateway Protocol, ExternalRef, capability constants
  model.py           # neutral dataclasses: ElnRecord/Resource/Chemical/Instrument
  prime.py           # prime-backend gateway  (PRIME)
  secondary.py       # secondary-backend gateway
  fake.py            # FakeElnGateway for offline tests
  identity.py        # user/team match (email/ORCID), no provisioning
  instruments.py     # instrument mapping (+ booking where supported)
  mapping.py         # MFDB node <-> neutral model + id-lookup helpers
  sync.py            # push()/pull() orchestration (backend-neutral)
  reconcile.py       # conflict policy
  test/
```

```python
CAP_RECORD, CAP_RESOURCE, CAP_CHEMICAL, CAP_ATTACH, CAP_LINK, CAP_SCOPE, \
CAP_USER_MATCH = ...            # every backend implements these
CAP_BOOKING, CAP_REACTION, CAP_ANALYSIS_TREE, CAP_DOI = ...   # optional

class ElnGateway(Protocol):
    backend: str                                   # "prime" | "secondary"
    def capabilities(self) -> frozenset[str]: ...
    def authenticate(self) -> None: ...
    def ensure_scope(self, name: str) -> ExternalRef: ...     # collection | team
    def upsert_record(self, rec: ElnRecord, scope: ExternalRef) -> ExternalRef: ...
    def upsert_resource(self, res: ElnResource, scope) -> ExternalRef: ...
    def upsert_chemical(self, chem: ElnChemical, scope) -> ExternalRef: ...
    def attach_file(self, record: ExternalRef, blob: bytes, meta: dict) -> ExternalRef: ...
    def set_metadata(self, ref: ExternalRef, extra: dict) -> None: ...
    def link(self, src: ExternalRef, dst: ExternalRef, rel: str) -> None: ...
    def match_user(self, email: str, orcid: str | None = None) -> ExternalRef | None: ...
    # optional (guard on capabilities())
    def upsert_instrument(self, inst: ElnInstrument, scope) -> ExternalRef: ...
    def book(self, instrument: ExternalRef, start, end, record: ExternalRef) -> ExternalRef: ...
    def mint_doi(self, ref: ExternalRef) -> str: ...          # prime backend only
```

All ELN traffic raises a single `ElnUnavailable` on network/auth failure, treated as **non-fatal** (mirrors `external_refs` returning `None`) — no partial writes. Backend selection is config-driven; more than one backend may be configured.

## 2. Neutral entity model & per-backend mapping

| ChiSurf / MFDB | neutral | prime backend | secondary backend |
|---|---|---|---|
| operation (measurement/analysis/fit) | **Record** | research plan + analysis container tree | experiment |
| result artifact / typed payload | **Attachment** | attachment on analysis→dataset container | upload on experiment |
| sample | **Resource** | sample (+ molecule) | item |
| reagent lot | **Resource** | sample / inventory | item |
| fluorophore | **Chemical** | molecule (InChIKey / canonical SMILES / registry number) | compound |
| provenance edge | **Link** | container nesting + collection grouping | items/experiments links |
| operator | **UserMatch** | user (Person) by email | user by email |
| group | **Scope member** | Group + users-groups | team |
| instrument + setup | **Instrument** | device + device metadata (**DOI**) | bookable item |
| usage / booking | **Booking** (opt) | — (no booking model) | team event |
| sync scope | **Scope** | collection (+ collection shares) | team |
| DOI / publish | **Publish** (opt) | research-plan/device DOI, repository/DOI-registration | — |

Capability matrix (what each backend can do): the prime backend adds `analysis_tree`, `reaction`, `doi`; the secondary backend adds `booking`. `sync.py` checks `capabilities()` before invoking an optional operation and records "skipped: unsupported" in the sync report rather than failing.

## 3. ID linkage + idempotency
External ids are written back into the MFDB record's existing JSON under a reserved `eln` namespace, **keyed by backend** so multiple ELNs coexist:

```json
"metadata_json": {
  "eln": {
    "prime":     {"record_id": 812, "kind": "research_plan",
                  "url": "https://.../research_plan/812",
                  "synced_at": "2026-07-01T09:00:00Z",
                  "remote_modified_at": "2026-07-01T08:58:00Z"},
    "secondary": {"record_id": 456, "kind": "experiment", "url": "..."}
  }
}
```

On the **ELN side**, the reverse pointer to the MFDB node is stored in that backend's free-form field — the prime backend's sample cross-reference / container extended-metadata / research-plan related-identifier; the secondary backend's entity extra-fields. `mapping.py` provides `set_external_ref(...)` / `find_node_by_external_id(backend, external_id)`; because `metadata_json` is not indexed, reverse lookup builds a cached `{external_id → node}` map per run (one scan). Adequate at current scale.

> **Future option (non-blocking).** Promote the mapping to a generic dictionary-driven `mfdb_external_ref(node_type, node_id, system, external_id, url, synced_at)` index if scans get hot. `mapping.py` is the seam; callers don't change. Deferred deliberately (no new schema now).

## 4. Push (deposit) flow
`sync.push(node_type, node_id, *, gateway, auth)` mirrors `project_archiver`:

1. Resolve the operation (+ sample/instrument/project context) from `api.py`.
2. `ensure_scope` (prime-backend collection / secondary-backend team) — the deposition boundary.
3. `upsert_record`: PATCH the existing record if `metadata_json.eln.<backend>` exists, else create and `set_external_ref`. The prime backend builds the research plan and its analysis container; the secondary backend an experiment.
4. Each output artifact → `attach_file` (bytes from `object_store`; typed payload → a prime-backend *dataset* under an *analysis*, or a secondary-backend upload).
5. Input sample/reagent/fluorophore → `upsert_resource`/`upsert_chemical` (deduped, §5), then `link` to the record.
6. Optional per capability: `upsert_instrument` + (`book` on the secondary backend) / (record the device on the prime backend); provenance edges → links / container nesting.

The gateway hides transport differences (the prime backend's chunked upload + complete + link-to-container vs the secondary backend's single multipart). An opt-in `EVENT_ARTIFACT_REGISTERED` subscriber enqueues async pushes (best-effort + retry; failure leaves MFDB untouched).

## 5. Pull (import) flow
`sync.pull(*, gateway, auth, kinds=("chemicals","resources","instruments"))`: list the backend's chemicals (prime-backend molecules / secondary-backend compounds), resources (samples/items), and instruments; for each, `find_node_by_external_id` → update, else create an MFDB sample/reagent/fluorophore/`flr_instrument`. **Dedup before create** via PRD-45 `normalize_cas` + PRD-06 fluorophore identity (and, for the prime backend, InChIKey) so the same substance is matched, not duplicated.

## 6. Conflict reconciliation (`reconcile.py`)
Compare local change time vs `metadata_json.eln.<backend>.remote_modified_at` and live remote `modified_at`:

| Field class | Authority | Rule |
|---|---|---|
| analysis results / fits / provenance | **ChiSurf** | local wins; ELN is a published copy |
| inventory: reagent lot, vendor, expiry | **ELN** | remote wins on pull |
| chemical identity (registry number/InChIKey/SMILES) | **ELN** | remote wins; local is cache |
| instrument / equipment record | **ELN** | remote wins (like inventory) |
| user identity (email/ORCID/name) | **neither** | match-only; never written |
| free-text title / notes | newer-wins | by `modified_at` |

Genuine both-sides-changed conflicts are **surfaced** in the sync report, never silently overwritten. Interactive resolution is a later phase.

## 7. Auth & security (PRD-37 alignment)
- Per-backend base URL + token in the OS credential store via `credentials.py`, keyed by `(backend, host, user)`; never on disk in plaintext, never logged. Prime-backend bearer token; secondary-backend `<userid>-<key>` (`can_write` gates push).
- **Fail-closed:** no credential / no network / TLS failure → `ElnUnavailable`, graceful degrade, no partial writes. TLS verification on by default.

## 8. Users & teams/groups — match, never provision
MFDB (`flr_sample_users`, `mfdb_group`) and each ELN are **independent identity authorities**. `identity.py` matches by **email** then **ORCID**; unmatched users are **surfaced, not created** (both ELNs restrict user creation to admins). A matched external `user_id` is cached in the record's `eln.<backend>` block. On push the record is authored by the matched operator (`mfdb_operation.operator_user_id`), falling back to the token's own user. No password/role/permission material ever crosses the boundary.

## 9. Instruments & bookings
MFDB `flr_instrument` + versioned `mfdb_setup` (opaque JSON today; PRD-08 will structure it; PRD-35 presets) maps to the prime backend's device + device metadata (DOI-bearing — a natural home for a citable instrument record) or a secondary-backend bookable item. Setup config → the prime backend's device description / extended metadata or the secondary backend's extra-fields. **Booking is an optional capability**: on the secondary backend an operation's time window becomes a team event bound to the experiment (usage record for free); the prime backend has no booking model, so that step is skipped (reported as unsupported). The ELN is authoritative for the instrument record.

## 10. Scope boundary
Prime-backend **collections** (with collection shares) and secondary-backend **teams** are the sync/sharing boundary. Config binds one MFDB `mfdb_group` to one collection / team per backend. `ensure_scope` creates/resolves it; nothing is deposited outside the configured scope.

## 11. Dissemination / DOI (prime backend, later phase)
Distinct from deposition ("deposition ≠ dissemination", "two clocks"). Once a record is deposited and curated, `mint_doi` (prime backend `CAP_DOI`) requests a DOI via the prime backend's public-repository path — the route by which databank content becomes citable and public. The secondary backend offers only trusted-timestamping, not DOIs. No dissemination happens automatically; it is an explicit, curated action.

## 12. Config & dependencies
- Add the secondary backend's official Python SDK to `chisurf-env.yaml`. The prime backend has no canonical SDK → a thin `requests`-based client generated/checked against its `/api/v1/swagger_doc`.
- Per-backend connection config (base URL, scope name, verify-TLS, default prime-backend collection / secondary-backend team) surfaced through ChiSurf settings.

# CLI / headless surface
Headless path required (repo rule — not GUI-only). `csc eln` group, backend selected by flag/config:

- `csc eln push <node-id> [--backend prime|secondary]` — deposit record + attachments + links (+ booking where supported).
- `csc eln pull [--kinds chemicals,resources,instruments]` — import.
- `csc eln users [--match]` — report user/team match status (no provisioning).
- `csc eln publish <node-id>` — mint DOI (prime backend `CAP_DOI` only).
- `csc eln status` — endpoint(s), scope, credential presence, capabilities, last sync.

GUI (a button in `mfdb_admin`) is a thin wrapper, deferred.

# Files

| Path | Change |
|---|---|
| `chisurf/core/mfdb/eln/__init__.py` | new — `ElnGateway` Protocol, `ExternalRef`, capability constants |
| `chisurf/core/mfdb/eln/model.py` | new — neutral `ElnRecord/Resource/Chemical/Instrument` |
| `chisurf/core/mfdb/eln/prime.py` | new — prime-backend gateway + thin REST client (prime) |
| `chisurf/core/mfdb/eln/secondary.py` | new — secondary-backend gateway (wraps its SDK) |
| `chisurf/core/mfdb/eln/fake.py` | new — `FakeElnGateway`, `ElnUnavailable` |
| `chisurf/core/mfdb/eln/identity.py` | new — user/team match, no provisioning (§8) |
| `chisurf/core/mfdb/eln/instruments.py` | new — instrument mapping + optional booking (§9) |
| `chisurf/core/mfdb/eln/mapping.py` | new — MFDB↔neutral translation + id lookup |
| `chisurf/core/mfdb/eln/sync.py` | new — capability-aware push/pull |
| `chisurf/core/mfdb/eln/reconcile.py` | new — conflict policy |
| `chisurf/core/mfdb/eln/test/` | new — offline unit + round-trip tests, both backends |
| `chisurf-env.yaml` | add the secondary backend's Python SDK |
| CLI registration (`csc`) | new `eln` group |
| `chisurf/core/mfdb/events.py` | (opt) async push subscriber, later phase |

# Verification
- **Offline unit tests** against `FakeElnGateway` and per-backend record/replay of the swagger / OpenAPI v2 shapes; no live network in CI (per `external_refs`).
- **Capability tests** — push against a backend lacking a capability skips it and reports "unsupported", never errors (e.g. booking on the prime backend, DOI on the secondary backend).
- **Round-trip** (both backends) — push an operation → pull it back → identity preserved, no duplicate node, ids reconciled under `metadata_json.eln.<backend>`.
- **Dedup** — importing a chemical whose registry number / InChIKey already exists matches the existing MFDB node (PRD-45/06) instead of creating a second.
- **Reconciliation** — both-sides-changed is reported, not overwritten.
- **Live smoke (opt-in)** — env-gated against throwaway ELN instances; never in default CI.

# Phasing
1. **Phase 1 (prime backend)** — `ElnGateway` + neutral model + capabilities; prime-backend gateway auth + user match + push (operation → research plan + analysis/dataset attachments) into a collection; id write-back; CLI `push`/`users`/`status`.
2. **Phase 2 (prime pull + chemicals/instruments)** — pull molecules/samples/devices → MFDB with registry-number/InChIKey/fluorophore dedup; instrument → device metadata.
3. **Phase 3 (secondary backend)** — secondary-backend gateway behind the same interface (experiments + uploads + bookable-item usage team event); reuse sync/reconcile.
4. **Phase 4 (reconciliation + dissemination + GUI)** — interactive conflicts, event-driven async push, prime-backend `mint_doi` / repository deposit, `mfdb_admin` button.

# Non-goals
- No changes to either ELN's schemas/servers.
- No real-time live-sync daemon (best-effort queue only).
- No migration of pre-existing ELN history into MFDB.
- No user/account provisioning in any system — match only (§8).
- No import of secondary-backend bookings / prime-backend device-analyses as MFDB entities.
- No password/role/permission synchronisation.
- Automatic DOI minting — dissemination is always an explicit, curated action.

# Open questions / decisions
- **Prime-backend record type.** Is a research plan the right home for a fluorescence measurement, or should some measurements become a sample + analysis tree only? Affects how operations map to records.
- **Prime-backend Python client.** Hand-roll a thin `requests` client vs. generate from `/api/v1/swagger_doc` — pin against a known release.
- **Dissemination scope.** Which databank content goes to the prime backend's public repository vs. stays in-ELN, and who curates the DOI step.
- **User match fallback.** When email/ORCID don't match, manual link via config map or `csc eln users --link <mfdb_user> <external_user_id>`?
- **Index table.** Confirm `metadata_json` scans stay acceptable, or schedule the optional `mfdb_external_ref` index (§3).

# Relationships
- Extends the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md); mirrors the project-archiver decomposition into operations + artifacts + edges.
- Keeps `api.py` transport-agnostic per the [RPC target](/specs/rpc.md).
- Related to fluorophore-identity ([PRD-06]) and chemical-identity ([PRD-45](prd-45.md)) work; aligns with the network-security (PRD-37), lineage/event-model (PRD-21), study/project entity (PRD-13), optical-configuration (PRD-08), and lightpath-storage (PRD-35) PRDs; CLI-first per the repo headless rule.
