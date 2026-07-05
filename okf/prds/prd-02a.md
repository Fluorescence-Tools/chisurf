---
type: PRD
prd: "02a"
title: "PRD-02a: mmCIF Dictionary Infrastructure"
description: Parse bundled mmCIF dictionaries into a cached API for vocabulary validation and autocomplete
status: done
phase: "foundation"
resource: overhaul/PRD-02a-mmcif-dictionary-infrastructure.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Parses all bundled mmCIF `.dic` dictionary files to extract categories, items,
descriptions, data types, and enumerated allowed values, exposing them through a
fast cached Python API. That API backs vocabulary validation, GUI autocomplete,
and flrCIF export-compliance checking, treating the dictionaries as the schema
authority. The prior parser only read one dictionary, returned empty
descriptions, and parsed no enumerations, data types, category metadata, or
parent-child links; this PRD covers the full FLR/IHM/ModelCIF category set
including non-enumerated required fields needed for export validation.

# Status
Done. The dictionary API is consumed by the sample/probe persistence boundary so
validation lives on one canonical path rather than duplicated across raw SQL
call sites.

# Relationships
- Depends on [PRD-020](prd-020.md); prerequisite for [PRD-02](prd-02.md) vocabulary validation.
- Feeds export alignment in [PRD-02c](prd-02c.md).
- Encodes the dictionary-as-authority principle of the [MFDB (current)](/architecture/mfdb.md) store and its [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-02a-mmcif-dictionary-infrastructure.md`
