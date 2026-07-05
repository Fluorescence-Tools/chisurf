# PRD-48 — ELN Integration (provider-agnostic; Chemotion + eLabFTW)

**Type:** Feature · **Status:** DRAFT
**Relates to:** PRD-41 (fdb4chembio access-layer strategy), PRD-37 (network
deployment security), PRD-45 (CAS chemical identity), PRD-06 (fluorophore
database), PRD-21 (lineage API / event model), PRD-13 (study/project entity),
PRD-08 (optical configuration), PRD-35 (lightpath storage).

## Goal / Motivation

ChiSurf's **MFDB** (`chisurf/core/mfdb/`) is the prototype implementation for
**fdb4chembio** — the planned **German fluorescence databank for chemical
biology** under **NFDI4Chem** (grant in preparation). For that vision the
databank must interoperate with the community's electronic lab notebooks, not
live as an island.

This PRD specifies a **provider-agnostic ELN integration layer**: a single
`ElnGateway` abstraction with two concrete backends.

- **Chemotion ELN — the prime target.** It is the NFDI4Chem ELN (KIT/ComPlat,
  DFG/NFDI4Chem-funded), chemistry-native (molecules with InChIKey / canonical
  SMILES / CAS), carries a rich *analysis → dataset → attachment* tree ideal for
  measured spectroscopy, models instruments with **DOIs** (`device_metadata`),
  and feeds a **Repository / DataCite** dissemination path. Its data model and
  mission align with fdb4chembio.
- **eLabFTW — secondary.** TU Dortmund's institutional ELN; a general
  experiment + resource + booking model, useful where a lab already runs it.

Both are reached over their REST APIs. The integration is **bidirectional**
(push deposition + pull import + reconciliation) and, following PRD-41, keeps a
clean separation between **deposition** (writing the working record into an ELN)
and **dissemination** (minting DOIs / publishing to a repository) — for
Chemotion the latter is a first-class, later-phase capability; for eLabFTW it
does not exist.

The guiding invariant is **one identity per real-world object** — a fluorophore,
sample, instrument, or operator is the same thing in MFDB and in the ELN.

## Background — what already exists

### MFDB seams (`chisurf/core/mfdb/`)

`api.py` is transport-agnostic ("the API is just functions", PRD-41) and is the
layer a sync engine drives. Entry points (all take `auth` →
`principal_from_rpc_auth`): `register_artifact()` (~ln 60, has `url`+`metadata`),
`record_operation()` (~ln 218), `record_operation_with_artifacts()` (~ln 303),
`register_sample()` (~ln 440), `register_experiment()` (~ln 485),
`record_operation_link()` (~ln 576), `export_graph()` (~ln 690).

- `models.py` — `MfdbOperation.metadata_json`, `MfdbArtifact.url` +
  `.metadata_json`, `SampleDefinition.extra`, reagent lots, `MfdbSetup`,
  `flr_instrument`.
- `object_store.py` — content-addressed (md5, dedup): the byte source for uploads.
- `payload_codec.py` / `payload_models.py` — typed payloads (`tcspc_decay`,
  `spectra`, `fcs_correlation`, `anisotropy_curve`, `pda_histogram`,
  `burst_table`) → natural Chemotion *datasets*.
- `events.py` — `EVENT_ARTIFACT_REGISTERED`, `EVENT_STATE_CHANGED` (async push).
- `auth/`, `session.py`, `credentials.py` — `Principal`/`SessionContext`,
  `flr_sample_users`, `mfdb_group`; OS credential store (PRD-37) for ELN tokens.
- `project_archiver.py` (`archive_project_to_mfdb`, ~ln 88) — the "decompose a
  workflow into operations+artifacts+edges" template; push mirrors it in reverse.

### Chemotion ELN (prime) — `thirdparty/chemotion_ELN/`

- **API:** Grape, base `/api/v1`, root `app/api/api.rb`; grape-swagger doc at
  `/api/v1/swagger_doc`. No canonical Python SDK.
- **Auth:** `Authorization: Bearer chemtoken_<hex>` (personal API token from
  `POST /api/v1/authentication/token`); JWT and session also accepted.
- **Entities** (`db/schema.rb`): `samples` (+ `molecules`: `inchikey`,
  `inchistring`, `cano_smiles`, `cas`, `sum_formular`), `reactions`,
  `research_plans`, `collections` (sharing via `collection_shares`,
  `permission_level` + per-element detail levels), the polymorphic **`containers`**
  tree (`containable_type`/`id` → `container_type` root→analyses→analysis→dataset)
  with `attachments` (`attachable_type='Container'`).
- **Identity:** STI on `users` (`type` = Person / Group / Admin); membership via
  `users_groups`. Match key: `email`.
- **Instruments:** `devices` + `device_descriptions` + **`device_metadata`**
  (`doi`, `landing_page`, `manufacturers`, DataCite fields). **No booking model**
  (`DevicesAnalysis` only loosely schedules analyses).
- **External ids / metadata:** `molecules.inchikey/cas/cano_smiles`,
  `samples.xref` (JSONB), `containers.extended_metadata` (hstore),
  `research_plan_metadata.doi/alternate_identifier/related_identifier`,
  `element_tags.taggable_data` — all viable back-reference stores.
- **Dissemination:** research_plan / device DOIs + Chemotion Repository/DataCite.

### eLabFTW (secondary) — `thirdparty/elabftw/`

- **API:** REST v2, base `/api/v2`; `Authorization: <userid>-<key>` (keys carry
  `can_write`, scoped to one `(user, team)`); official **`elabapi-python`** SDK.
- **Entities:** `experiments`, `items` (resources), `compounds`
  (CAS/InChI/SMILES/PubChem), multipart `uploads`, `metadata.extra_fields`,
  entity `links`; **bookable items** + `team_events` (booking with `experiment`
  binding); `users`/`teams`/`Users2Teams`.

## Design

### 1. Provider-agnostic `ElnGateway`

A new subpackage **`chisurf/core/mfdb/eln/`** isolates all ELN knowledge; `api.py`
stays transport-agnostic. One neutral interface, two adapters, a declared
capability set so callers degrade gracefully where a backend lacks a feature:

```
chisurf/core/mfdb/eln/
  __init__.py        # ElnGateway Protocol, ExternalRef, capability constants
  model.py           # neutral dataclasses: ElnRecord/Resource/Chemical/Instrument
  chemotion.py       # ChemotionGateway  (PRIME)
  elabftw.py         # ElabftwGateway
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
    backend: str                                   # "chemotion" | "elabftw"
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
    def mint_doi(self, ref: ExternalRef) -> str: ...          # Chemotion only
```

All ELN traffic raises a single `ElnUnavailable` on network/auth failure, treated
as **non-fatal** (mirrors `external_refs` returning `None`) — no partial writes.
Backend selection is config-driven; more than one backend may be configured.

### 2. Neutral entity model & per-backend mapping

| ChiSurf / MFDB | neutral | Chemotion (prime) | eLabFTW |
|---|---|---|---|
| operation (measurement/analysis/fit) | **Record** | `research_plan` + analysis container tree | `experiment` |
| result artifact / typed payload | **Attachment** | `attachment` on `analysis→dataset` container | `upload` on experiment |
| sample | **Resource** | `sample` (+ `molecule`) | `item` |
| reagent lot | **Resource** | `sample` / inventory | `item` |
| fluorophore | **Chemical** | `molecule` (inchikey/cano_smiles/cas) | `compound` |
| provenance edge | **Link** | container nesting + collection grouping | items/experiments_links |
| operator | **UserMatch** | `users(type=Person)` by email | `users` by email |
| group | **Scope member** | `Group` + `users_groups` | `team` |
| instrument + setup | **Instrument** | `device` + `device_metadata` (**DOI**) | bookable `item` |
| usage / booking | **Booking** (opt) | — (no booking model) | `team_event` |
| sync scope | **Scope** | `collection` (+ `collection_shares`) | `team` |
| DOI / publish | **Publish** (opt) | research_plan/device DOI, Repository/DataCite | — |

Capability matrix (what each backend can do): Chemotion adds `analysis_tree`,
`reaction`, `doi`; eLabFTW adds `booking`. `sync.py` checks `capabilities()`
before invoking an optional operation and records "skipped: unsupported" in the
sync report rather than failing.

### 3. ID linkage + idempotency

eLabFTW ids are written back into the MFDB record's existing JSON under a reserved
`eln` namespace, **keyed by backend** so multiple ELNs coexist:

```json
"metadata_json": {
  "eln": {
    "chemotion": {"record_id": 812, "kind": "research_plan",
                  "url": "https://.../research_plan/812",
                  "synced_at": "2026-07-01T09:00:00Z",
                  "remote_modified_at": "2026-07-01T08:58:00Z"},
    "elabftw":   {"record_id": 456, "kind": "experiment", "url": "..."}
  }
}
```

On the **ELN side**, the reverse pointer to the MFDB node is stored in that
backend's free-form field — Chemotion `samples.xref` / `containers.extended_metadata`
/ `research_plan_metadata.related_identifier`; eLabFTW `metadata.extra_fields`.
`mapping.py` provides `set_external_ref(...)` / `find_node_by_external_id(backend,
external_id)`; because `metadata_json` is not indexed, reverse lookup builds a
cached `{external_id → node}` map per run (one scan). Adequate at current scale.

> **Future option (non-blocking).** Promote the mapping to a generic
> dictionary-driven `mfdb_external_ref(node_type, node_id, system, external_id,
> url, synced_at)` index if scans get hot. `mapping.py` is the seam; callers
> don't change. Deferred deliberately (no new schema now).

### 4. Push (deposit) flow

`sync.push(node_type, node_id, *, gateway, auth)` mirrors `project_archiver`:

1. Resolve the operation (+ sample/instrument/project context) from `api.py`.
2. `ensure_scope` (Chemotion collection / eLabFTW team) — the deposition boundary.
3. `upsert_record`: PATCH the existing record if `metadata_json.eln.<backend>`
   exists, else create and `set_external_ref`. Chemotion builds the
   `research_plan` and its analysis container; eLabFTW an experiment.
4. Each output artifact → `attach_file` (bytes from `object_store`; typed payload
   → a Chemotion *dataset* under an *analysis*, or an eLabFTW upload).
5. Input sample/reagent/fluorophore → `upsert_resource`/`upsert_chemical`
   (deduped, §5), then `link` to the record.
6. Optional per capability: `upsert_instrument` + (`book` on eLabFTW) / (record
   the device on Chemotion); provenance edges → links / container nesting.

The gateway hides transport differences (Chemotion chunked `upload_chunk` +
`upload_chunk_complete` + link-to-container vs eLabFTW single multipart). An
opt-in `EVENT_ARTIFACT_REGISTERED` subscriber enqueues async pushes (best-effort
+ retry; failure leaves MFDB untouched).

### 5. Pull (import) flow

`sync.pull(*, gateway, auth, kinds=("chemicals","resources","instruments"))`:
list the backend's chemicals (Chemotion `molecules` / eLabFTW `compounds`),
resources (`samples`/`items`), and instruments; for each, `find_node_by_external_id`
→ update, else create an MFDB sample/reagent/fluorophore/`flr_instrument`.
**Dedup before create** via PRD-45 `normalize_cas` + PRD-06 fluorophore identity
(and, for Chemotion, InChIKey) so the same substance is matched, not duplicated.

### 6. Conflict reconciliation (`reconcile.py`)

Compare local change time vs `metadata_json.eln.<backend>.remote_modified_at` and
live remote `modified_at`:

| Field class | Authority | Rule |
|---|---|---|
| analysis results / fits / provenance | **ChiSurf** | local wins; ELN is a published copy |
| inventory: reagent lot, vendor, expiry | **ELN** | remote wins on pull |
| chemical identity (CAS/InChIKey/SMILES) | **ELN** | remote wins; local is cache |
| instrument / equipment record | **ELN** | remote wins (like inventory) |
| user identity (email/ORCID/name) | **neither** | match-only; never written |
| free-text title / notes | newer-wins | by `modified_at` |

Genuine both-sides-changed conflicts are **surfaced** in the sync report, never
silently overwritten. Interactive resolution is a later phase.

### 7. Auth & security (PRD-37 alignment)

- Per-backend base URL + token in the OS credential store via `credentials.py`,
  keyed by `(backend, host, user)`; never on disk in plaintext, never logged.
  Chemotion `chemtoken_…`; eLabFTW `<userid>-<key>` (`can_write` gates push).
- **Fail-closed:** no credential / no network / TLS failure → `ElnUnavailable`,
  graceful degrade, no partial writes. TLS verification on by default.

### 8. Users & teams/groups — match, never provision

MFDB (`flr_sample_users`, `mfdb_group`) and each ELN are **independent identity
authorities**. `identity.py` matches by **email** then **ORCID**; unmatched users
are **surfaced, not created** (both ELNs restrict user creation to admins). A
matched external `user_id` is cached in the record's `eln.<backend>` block. On
push the record is authored by the matched operator
(`mfdb_operation.operator_user_id`), falling back to the token's own user.
No password/role/permission material ever crosses the boundary.

### 9. Instruments & bookings

MFDB `flr_instrument` + versioned `mfdb_setup` (opaque JSON today; PRD-08 will
structure it; PRD-35 presets) maps to Chemotion `device` + `device_metadata`
(DOI-bearing — a natural home for a citable instrument record) or an eLabFTW
bookable item. Setup config → Chemotion `device_description`/`extended_metadata`
or eLabFTW `metadata.extra_fields`. **Booking is an optional capability**: on
eLabFTW an operation's time window becomes a `team_event` bound to the experiment
(usage record for free); Chemotion has no booking model, so that step is skipped
(reported as unsupported). ELN is authoritative for the instrument record.

### 10. Scope boundary

Chemotion **collections** (with `collection_shares`) and eLabFTW **teams** are
the sync/sharing boundary. Config binds one MFDB `mfdb_group` to one collection /
team per backend. `ensure_scope` creates/resolves it; nothing is deposited
outside the configured scope.

### 11. Dissemination / DOI (Chemotion, later phase)

Distinct from deposition (PRD-41 "deposition ≠ dissemination", "two clocks").
Once a record is deposited and curated, `mint_doi` (Chemotion `CAP_DOI`) requests
a DataCite DOI via the Chemotion Repository path — the route by which fdb4chembio
content becomes citable and public. eLabFTW offers only trusted-timestamping, not
DOIs. No dissemination happens automatically; it is an explicit, curated action.

### 12. Config & dependencies

- Add `elabapi-python` (eLabFTW) to `chisurf-env.yaml`. Chemotion has no canonical
  SDK → a thin `requests`-based `ChemotionClient` generated/checked against
  `/api/v1/swagger_doc`.
- Per-backend connection config (base URL, scope name, verify-TLS, default
  Chemotion collection / eLabFTW team) surfaced through ChiSurf settings.

## CLI / headless surface

Headless path required (repo rule — not GUI-only). `csc eln` group, backend
selected by flag/config:

- `csc eln push <node-id> [--backend chemotion|elabftw]` — deposit record +
  attachments + links (+ booking where supported).
- `csc eln pull [--kinds chemicals,resources,instruments]` — import.
- `csc eln users [--match]` — report user/team match status (no provisioning).
- `csc eln publish <node-id>` — mint DOI (Chemotion `CAP_DOI` only).
- `csc eln status` — endpoint(s), scope, credential presence, capabilities, last sync.

GUI (a button in `mfdb_admin`) is a thin wrapper, deferred.

## Files

| Path | Change |
|---|---|
| `chisurf/core/mfdb/eln/__init__.py` | new — `ElnGateway` Protocol, `ExternalRef`, capability constants |
| `chisurf/core/mfdb/eln/model.py` | new — neutral `ElnRecord/Resource/Chemical/Instrument` |
| `chisurf/core/mfdb/eln/chemotion.py` | new — `ChemotionGateway` + `ChemotionClient` (prime) |
| `chisurf/core/mfdb/eln/elabftw.py` | new — `ElabftwGateway` (wraps `elabapi-python`) |
| `chisurf/core/mfdb/eln/fake.py` | new — `FakeElnGateway`, `ElnUnavailable` |
| `chisurf/core/mfdb/eln/identity.py` | new — user/team match, no provisioning (§8) |
| `chisurf/core/mfdb/eln/instruments.py` | new — instrument mapping + optional booking (§9) |
| `chisurf/core/mfdb/eln/mapping.py` | new — MFDB↔neutral translation + id lookup |
| `chisurf/core/mfdb/eln/sync.py` | new — capability-aware push/pull |
| `chisurf/core/mfdb/eln/reconcile.py` | new — conflict policy |
| `chisurf/core/mfdb/eln/test/` | new — offline unit + round-trip tests, both backends |
| `chisurf-env.yaml` | add `elabapi-python` |
| CLI registration (`csc`) | new `eln` group |
| `chisurf/core/mfdb/events.py` | (opt) async push subscriber, later phase |

## Verification

- **Offline unit tests** against `FakeElnGateway` and per-backend record/replay
  of the Grape-swagger / OpenAPI v2 shapes; no live network in CI (per
  `external_refs`).
- **Capability tests** — push against a backend lacking a capability skips it and
  reports "unsupported", never errors (e.g. booking on Chemotion, DOI on eLabFTW).
- **Round-trip** (both backends) — push an operation → pull it back → identity
  preserved, no duplicate node, ids reconciled under `metadata_json.eln.<backend>`.
- **Dedup** — importing a chemical whose CAS/InChIKey already exists matches the
  existing MFDB node (PRD-45/06) instead of creating a second.
- **Reconciliation** — both-sides-changed is reported, not overwritten.
- **Live smoke (opt-in)** — env-gated against throwaway Chemotion / eLabFTW
  instances; never in default CI.

## Phasing

1. **Phase 1 (Chemotion, prime)** — `ElnGateway` + neutral model + capabilities;
   `ChemotionGateway` auth + user match + push (operation → research_plan +
   analysis/dataset attachments) into a collection; id write-back; CLI
   `push`/`users`/`status`.
2. **Phase 2 (Chemotion pull + chemicals/instruments)** — pull molecules/samples/
   devices → MFDB with CAS/InChIKey/fluorophore dedup; instrument → `device_metadata`.
3. **Phase 3 (eLabFTW backend)** — `ElabftwGateway` behind the same gateway
   (experiments + uploads + bookable-item usage `team_event`); reuse sync/reconcile.
4. **Phase 4 (reconciliation + dissemination + GUI)** — interactive conflicts,
   event-driven async push, Chemotion `mint_doi` / Repository deposit, `mfdb_admin`
   button.

## Non-goals

- No changes to Chemotion or eLabFTW schemas/servers.
- No real-time live-sync daemon (best-effort queue only).
- No migration of pre-existing ELN history into MFDB.
- No user/account provisioning in any system — match only (§8).
- No import of eLabFTW bookings / Chemotion device-analyses as MFDB entities.
- No password/role/permission synchronisation.
- Automatic DOI minting — dissemination is always an explicit, curated action.

## Open questions / decisions

- **Chemotion record type.** Is `research_plan` the right home for a fluorescence
  measurement, or should some measurements become `sample` + analysis tree only?
  Affects how operations map to records.
- **Chemotion Python client.** Hand-roll a thin `requests` client vs. generate
  from `/api/v1/swagger_doc` — pin against a known Chemotion release.
- **Dissemination scope.** Which fdb4chembio content goes to the Chemotion
  Repository vs. stays in-ELN, and who curates the DOI step (ties to PRD-41).
- **User match fallback.** When email/ORCID don't match, manual link via config
  map or `csc eln users --link <mfdb_user> <external_user_id>`?
- **Index table.** Confirm `metadata_json` scans stay acceptable, or schedule the
  optional `mfdb_external_ref` index (§3).
