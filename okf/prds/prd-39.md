---
type: PRD
prd: "39"
title: "PRD-39: Sequence Provenance & External References"
description: Records each entity's canonical sequence/structure cross-references and its engineered mutations as structured, exportable flrCIF/PDBx data using the standard struct_ref category family.
status: planned
phase: "4"
resource: overhaul/PRD-39-sequence-external-references.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-39 makes every protein or nucleic-acid entity traceable to its source record in public protein sequence/structure reference databases, and records every engineered mutation (notably the cysteine substitutions used as dye-attachment points) as structured, exportable data rather than free-text notes. It reuses the standard PDBx `struct_ref` / `struct_ref_seq` / `struct_ref_seq_dif` category family already present in the bundled dictionaries — no custom `.dic` invention — and adds matching schema tables, dataclass extensions (`EntityDefinition` fields plus a new `MutationDefinition`), offline-safe fetch services for the canonical sequence and structure-to-sequence chain mapping, a construct-vs-reference auto-diff, a consistency validator against probe positions, and flrCIF round-trip. Everything is usable headlessly first, with a thin GUI on top.

# Status
Planned (phase 4, STATUS TABLE authoritative). The source document records all tasks as implemented and tested (2026-06-27); the STATUS TABLE is authoritative for the concept's tracking status.

# Relationships
- Depends on PRD-02 (sample tracking), PRD-02a (mmCIF dictionary infrastructure), and PRD-02c (flrCIF alignment) — consumes their entity/sequence/probe model.
- Adjacent to PRD-06 (fluorophore database) and PRD-33 (acquisition to MFDB registration): samples those flows create carry these references.
- Named by [PRD-41](prd-41.md) as on the critical path (the dictionary and CIF round-trip), and a chemistry analogue of [PRD-45](prd-45.md) (CAS as the identity axis for chemistry).
- Extends the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-39-sequence-external-references.md`
