---
type: PRD
prd: "45"
title: "PRD-45: Chemical registry-number as a first-class chemical identity in MFDB"
description: Promotes the chemical registry number from an ad-hoc free-text property to a dictionary-defined, validated, indexed, cross-entity chemical identity surfaced across GUI, CLI, and RPC.
status: draft
phase: "unassigned"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-45 makes a standard chemical registry-number identity (CAS) a general, first-class feature of MFDB rather than an ad-hoc optical property on probes. A chemical registry number uniquely identifies a substance, so it is the natural join key for dyes and compounds across the reference set, samples, reagents, and public fluorophore spectral databases — the chemistry analogue of the sequence/structure accessions handled by [PRD-39](prd-39.md). The design declares the identifier in the `.dic` schema authority, validates and normalizes it (including its check digit) on every write, gives it a real indexed column or normalized identity table, shares one identity across reference probes / sample probes / reagent lots, and surfaces it through AutoForm view schemes, list filters, the CLI, and the RPC API, with an optional offline-safe external resolver. A minimal property-bag foundation already exists and is forward-compatible.

# Status
Draft (unassigned phase, STATUS TABLE authoritative). Design only — implementation deferred; a minimal property-bag groundwork (alias canonicalization, whitespace-tolerant lookup, a register convenience param) is in place and forward-compatible with this design.

# Motivation
Scrapers already capture chemical registry numbers for chemical compounds (from spectral-database and dye-vendor sources). Today the registry number lands as a free-text optical property, which means:

- **No identity guarantees** — the same compound scraped from two sources may carry differently-formatted registry strings (`"71-43-2"` vs `" 71-43-2"`) and never be recognised as the same substance.
- **No validation** — registry numbers have a check digit; malformed values are stored silently.
- **No cross-entity link** — a sample's probe, a reagent lot, and a reference dye that are the *same chemical* are not connected by identity.
- **Weak lookup** — finding "the probe for CAS 71-43-2" requires scanning the property bag; there is no index and no canonical column.

The registry number is the right identity axis for chemistry (analogous to what sequence/structure accession authorities are for proteins, already handled by PRD-39 sequence external references). This PRD gives it the same first-class treatment.

# Current foundation (already in place, forward-compatible)
A minimal, non-invasive groundwork exists and should be preserved/extended, not reworked:

- `repository.py::_PROP_ALIASES["cas"]` — canonicalises `CAS` / `CAS Number` / `CAS NBR` / `CASRN` / … → the single key `cas` on write.
- `repository.py::MFDatabase.find_probes_by_cas(cas)` — whitespace-tolerant lookup of probes by their `cas` property.
- `mfdb_adapter.register_component(..., cas=...)` — convenience parameter that records the registry number for a scraped component.

This PRD promotes that property-bag stopgap into a dictionary-defined, validated, indexed identity.

# Goals & Constraints
1. **Dictionary-defined.** The registry number is declared in the `.dic` schema authority (a `_mfdb_schema`-mapped item), not hand-added DDL — consistent with the rule that the `.dic` family is the schema source of truth.
2. **Validated.** Normalise to canonical `NNNNNNN-NN-N` form and verify the check digit on write; reject/flag malformed values (fail-loud, per PRD-25).
3. **Indexed & queryable.** A real, indexed column (or a normalised identity table) so `lookup by registry number` is O(log n), not a property scan.
4. **Cross-entity.** The identity is shared by reference probes, sample probes, and reagent lots — the same substance resolves to one identity.
5. **Surfaced everywhere.** GUI detail forms (AutoForm view schemes), list columns/filters, CLI (`csc fluorophore cas <n>`), and the RPC API.
6. **Externally resolvable (optional).** A resolver that, given a registry number, can fetch canonical name/structure from an external compound resolver (cache-first, offline-safe).
7. **Back-compatible.** Existing `cas` properties migrate into the new field; no data loss; the property alias keeps working during transition.

# Design

## 1. Dictionary + schema
- Add a registry-number item to `mfdb_flr_ext.dic` (e.g. on the probe/chem-component category) with `_mfdb_schema.table_name`/`.column_name` mapping it to a real column, an enumeration-free `code`/`line` type, and a description. Regenerate DDL via `schema_from_dictionary.py` (do not hand-edit generated DDL).
- Where chemical identity is shared, model the registry number on the `chem_descriptors` / chemical-component table rather than duplicating it per probe; probes/reagents reference the chemical identity. (Decision point — see *Open questions*.)
- Add a unique-ish index on the normalised registry-number column. The registry number is not guaranteed unique per row (mixtures, salts), so index but do not hard-`UNIQUE`.

## 2. Validation & normalisation (`chisurf/core/mfdb/`)
- `normalize_cas(s) -> str | None`: strip, collapse, validate the regex `\d{2,7}-\d{2}-\d` and the **check digit** (last digit = sum of digit·position mod 10). Return the canonical string or `None` if invalid.
- Used on every write (scraper ingest, GUI edit, import). Invalid values are recorded as a quality flag / kept in a `cas_raw` audit field rather than silently dropped.

## 3. API (`repository.py` / `api.py`)
- Promote `find_probes_by_cas` to query the indexed column; add `resolve_chemical_identity(cas)` returning the shared identity + all entities (probes, reagents, samples) that reference it.
- `set_probe_cas(probe_id, cas)` with validation.
- Expose through the transport-agnostic `api.py` and the mfdb-admin RPC (`chemistry.cas.lookup`, `chemistry.cas.set`).

## 4. GUI (AutoForm)
- Add a `cas` field to the optical-component view schemes (`fluorophore.view.json`, and the new chemical entities) with a tooltip and inline validity styling.
- Add a registry-number column + filter to the component list, and a "look up by registry number" action that cross-links to the matching probe(s) (reuses the cross-link infra).

## 5. CLI
- `csc fluorophore cas <number>` — print probes for a registry number.
- `csc fluorophore list --cas <number>` filter.

## 6. External resolution (optional, phase 2)
- `chisurf/core/chem/cas_resolver.py`: registry number → external compound identifier → canonical name / InChIKey / structure, cache-first (store resolved metadata in MFDB), fully offline-safe (no network ⇒ no-op). Gated like the AI settings: never block ingest on network.

# Files (anticipated)

| File | Change |
|------|--------|
| `chisurf/core/mfdb/data/mfdb_flr_ext.dic` | Declare the registry-number item + `_mfdb_schema` mapping |
| `chisurf/core/mfdb/schema*.py` | Regenerated DDL + index + migration of existing `cas` properties |
| `chisurf/core/mfdb/cas.py` (NEW) | `normalize_cas`, check-digit validation |
| `chisurf/core/mfdb/repository.py` | indexed `find_probes_by_cas`, `set_probe_cas`, `resolve_chemical_identity` |
| `chisurf/core/mfdb/api.py` | transport-agnostic registry-number functions |
| `chisurf/plugins/core/mfdb_admin/backend/*` | `chemistry.cas.*` RPC |
| `chisurf/plugins/core/mfdb_admin/gui/optical_components/*.view.json` | registry-number field + filter |
| `chisurf/plugins/core/mfdb_admin/cli/__init__.py` | `cas` lookup command |
| `chisurf/core/chem/cas_resolver.py` (NEW, phase 2) | external compound resolution (cache-first) |

# Verification
- **Validation unit tests**: `normalize_cas` accepts known-good registry numbers (Benzene 71-43-2, Fluorescein 2321-07-5), rejects bad check digits and malformed input.
- **Migration test**: existing `cas` optical properties move into the new column; no probe loses its registry number; the property alias still resolves during transition.
- **Lookup test**: `find_probes_by_cas` is index-backed and whitespace/format tolerant; `resolve_chemical_identity` returns all referencing entities.
- **GUI**: registry number shows in the detail form, the list filter works, look-up jumps to the probe; invalid values are flagged in the form.
- **Resolver** (phase 2): offline ⇒ no-op; online ⇒ caches canonical name/InChIKey.

# Open questions / decisions for implementation
1. **Where does the registry number live** — a column on `probes`, or on a shared `chem_descriptors`/chemical-component table that probes/reagents reference? (Recommendation: shared identity table, so the cross-entity goal is real.)
2. **Uniqueness** — index only (mixtures/salts/hydrates share names but differ), not a hard `UNIQUE` constraint.
3. **Scope of cross-entity linking in v1** — probes only, or probes + reagent lots + sample probes at once?

# Phasing
1. `cas.py` validation + indexed column + migration of existing properties + `find_probes_by_cas`/`set_probe_cas` (keeps the property alias as fallback).
2. GUI/CLI/RPC surfacing + cross-entity identity resolution.
3. External compound resolver (cache-first, optional).

# Relationships
- Chemistry analogue of [PRD-39](prd-39.md), which gives sequence/structure accessions the same first-class treatment for proteins.
- Depends on the vendor-neutral dictionary namespace from [PRD-44](prd-44.md) (the identifier is a `_mfdb_schema`-mapped item).
- Follows the fail-loud policy of PRD-25 for malformed values.
- Extends the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md); surfaced via [GUI & AutoForm](/subsystems/gui-autoform.md) and the [RPC target](/specs/rpc.md).
