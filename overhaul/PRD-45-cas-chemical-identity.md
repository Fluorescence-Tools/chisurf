# PRD-45: CAS Registry Number as a First-Class Chemical Identity in MFDB

> **Status: DRAFT (2026-06-29).** Design only — implementation deferred. A minimal
> property-bag foundation already exists and is forward-compatible with this
> design (see *Current foundation* below); the full integration described here is
> the work item.

> **Scope:** Make the CAS Registry Number a general, first-class chemical-identity
> feature of MFDB — dictionary-defined, validated, indexed, queryable, and
> surfaced in the GUI/CLI/RPC — rather than an ad-hoc optical property on probes.
> CAS uniquely identifies a chemical substance, so it is the natural join key for
> dyes/compounds across the reference set, samples, reagents, and external
> databases (PubChem, ChemSpider, FPbase, PhotochemCAD).

## Motivation

Scrapers already capture CAS numbers for chemical compounds (PhotochemCAD's
`cas`, ATTO product pages, FPbase dyes). Today CAS lands as a free-text optical
property, which means:

- **No identity guarantees** — the same compound scraped from two sources may
  carry differently-formatted CAS strings (`"71-43-2"` vs `" 71-43-2"`) and never
  be recognised as the same substance.
- **No validation** — CAS numbers have a check digit; malformed values are
  stored silently.
- **No cross-entity link** — a sample's probe, a reagent lot, and a reference
  dye that are the *same chemical* are not connected by identity.
- **Weak lookup** — finding "the probe for CAS 71-43-2" requires scanning the
  property bag; there is no index and no canonical column.

CAS is the right identity axis for chemistry (analogous to what UniProt/PDB
accessions are for proteins, already handled by PRD-39 sequence external
references). This PRD gives CAS the same first-class treatment.

## Current foundation (already in place, forward-compatible)

A minimal, non-invasive groundwork exists and should be preserved/extended, not
reworked:

- `repository.py::_PROP_ALIASES["cas"]` — canonicalises `CAS` / `CAS Number` /
  `CAS NBR` / `CASRN` / … → the single key `cas` on write.
- `repository.py::MFDatabase.find_probes_by_cas(cas)` — whitespace-tolerant
  lookup of probes by their `cas` property.
- `mfdb_adapter.register_component(..., cas=...)` — convenience parameter that
  records the CAS for a scraped component.

This PRD promotes that property-bag stopgap into a dictionary-defined,
validated, indexed identity.

## Goals & Constraints

1. **Dictionary-defined.** CAS is declared in the `.dic` schema authority (a
   `_mfdb_schema`-mapped item), not hand-added DDL — consistent with the rule
   that the `.dic` family is the schema source of truth.
2. **Validated.** Normalise to canonical `NNNNNNN-NN-N` form and verify the CAS
   check digit on write; reject/flag malformed values (fail-loud, per PRD-25).
3. **Indexed & queryable.** A real, indexed column (or a normalised identity
   table) so `lookup by CAS` is O(log n), not a property scan.
4. **Cross-entity.** CAS identity is shared by reference probes, sample probes,
   and reagent lots — the same substance resolves to one identity.
5. **Surfaced everywhere.** GUI detail forms (AutoForm view schemes), list
   columns/filters, CLI (`csc fluorophore cas <n>`), and the RPC API.
6. **Externally resolvable (optional).** A resolver that, given a CAS, can fetch
   canonical name/structure from PubChem (cache-first, offline-safe).
7. **Back-compatible.** Existing `cas` properties migrate into the new field; no
   data loss; the property alias keeps working during transition.

## Design

### 1. Dictionary + schema
- Add a CAS item to `mfdb_flr_ext.dic` (e.g. on the probe/chem-component
  category) with `_mfdb_schema.table_name`/`.column_name` mapping it to a real
  column, an enumeration-free `code`/`line` type, and a description. Regenerate
  DDL via `schema_from_dictionary.py` (do not hand-edit generated DDL).
- Where chemical identity is shared, model CAS on the `chem_descriptors` /
  chemical-component table rather than duplicating it per probe; probes/reagents
  reference the chemical identity. (Decision point — see *Open questions*.)
- Add a unique-ish index on the normalised CAS column. CAS is not guaranteed
  unique per row (mixtures, salts), so index but do not hard-`UNIQUE`.

### 2. Validation & normalisation (`chisurf/core/mfdb/`)
- `normalize_cas(s) -> str | None`: strip, collapse, validate the regex
  `\d{2,7}-\d{2}-\d` and the **check digit** (last digit = sum of digit·position
  mod 10). Return the canonical string or `None` if invalid.
- Used on every write (scraper ingest, GUI edit, import). Invalid values are
  recorded as a quality flag / kept in a `cas_raw` audit field rather than
  silently dropped.

### 3. API (`repository.py` / `api.py`)
- Promote `find_probes_by_cas` to query the indexed column; add
  `resolve_chemical_identity(cas)` returning the shared identity + all entities
  (probes, reagents, samples) that reference it.
- `set_probe_cas(probe_id, cas)` with validation.
- Expose through the transport-agnostic `api.py` and the mfdb-admin RPC
  (`chemistry.cas.lookup`, `chemistry.cas.set`).

### 4. GUI (AutoForm)
- Add a `cas` field to the optical-component view schemes
  (`fluorophore.view.json`, and the new chemical entities) with a tooltip and
  inline validity styling.
- Add a CAS column + filter to the component list, and a "look up by CAS" action
  that cross-links to the matching probe(s) (reuses the cross-link infra).

### 5. CLI
- `csc fluorophore cas <number>` — print probes for a CAS.
- `csc fluorophore list --cas <number>` filter.

### 6. External resolution (optional, phase 2)
- `chisurf/core/chem/cas_resolver.py`: CAS → PubChem CID → canonical name /
  InChIKey / structure, cache-first (store resolved metadata in MFDB), fully
  offline-safe (no network ⇒ no-op). Gated like the AI settings: never block
  ingest on network.

## Files (anticipated)

| File | Change |
|------|--------|
| `chisurf/core/mfdb/data/mfdb_flr_ext.dic` | Declare the CAS item + `_mfdb_schema` mapping |
| `chisurf/core/mfdb/schema*.py` | Regenerated DDL + index + migration of existing `cas` properties |
| `chisurf/core/mfdb/cas.py` (NEW) | `normalize_cas`, check-digit validation |
| `chisurf/core/mfdb/repository.py` | indexed `find_probes_by_cas`, `set_probe_cas`, `resolve_chemical_identity` |
| `chisurf/core/mfdb/api.py` | transport-agnostic CAS functions |
| `chisurf/plugins/core/mfdb_admin/backend/*` | `chemistry.cas.*` RPC |
| `chisurf/plugins/core/mfdb_admin/gui/optical_components/*.view.json` | CAS field + filter |
| `chisurf/plugins/core/mfdb_admin/cli/__init__.py` | `cas` lookup command |
| `chisurf/core/chem/cas_resolver.py` (NEW, phase 2) | PubChem resolution (cache-first) |

## Verification

- **Validation unit tests**: `normalize_cas` accepts known-good CAS (Benzene
  71-43-2, Fluorescein 2321-07-5), rejects bad check digits and malformed input.
- **Migration test**: existing `cas` optical properties move into the new column;
  no probe loses its CAS; the property alias still resolves during transition.
- **Lookup test**: `find_probes_by_cas` is index-backed and whitespace/format
  tolerant; `resolve_chemical_identity` returns all referencing entities.
- **GUI**: CAS shows in the detail form, the list filter works, look-up jumps to
  the probe; invalid CAS is flagged in the form.
- **Resolver** (phase 2): offline ⇒ no-op; online ⇒ caches PubChem name/InChIKey.

## Open questions / decisions for implementation

1. **Where does CAS live** — a column on `probes`, or on a shared
   `chem_descriptors`/chemical-component table that probes/reagents reference?
   (Recommendation: shared identity table, so the cross-entity goal is real.)
2. **Uniqueness** — index only (mixtures/salts/hydrates share names but differ),
   not a hard `UNIQUE` constraint.
3. **Scope of cross-entity linking in v1** — probes only, or probes + reagent
   lots + sample probes at once?

## Phasing

1. `cas.py` validation + indexed column + migration of existing properties +
   `find_probes_by_cas`/`set_probe_cas` (keeps the property alias as fallback).
2. GUI/CLI/RPC surfacing + cross-entity identity resolution.
3. External PubChem resolver (cache-first, optional).
