---
type: PRD
prd: "13"
title: "PRD-13: Study / Project Entity with Configurable Fields"
description: Promote the loose project-id string into a real study entity that groups samples and datasets with ownership, membership, and custom fields.
status: done
phase: "3"
resource: overhaul/PRD-13-study-project-entity.md
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
`project_id` was a loose string with no table, membership, or custom fields. This PRD makes a first-class **study** entity (`mfdb_study`) that groups samples and datasets, carries ownership and visibility (own + public), supports many-to-many membership (`mfdb_study_member`), and holds per-study configurable metadata via the existing key-value pattern (no forked EAV stack). The dataset browser gains a `study_id` facet, and studies are backfilled idempotently from distinct existing `project_id` values. Repository CRUD, membership, scoping, RPC handlers, and a standalone admin studies view are provided, all dictionary-declared and gate-covered.

# Status
Done. Schema, CRUD, many-to-many membership, configurable fields, browser facet, RPC, and admin view landed (23 tests). Wiring the studies view into the admin dock layout and adding a study selector to the browser widget are deferred to the dock rewrite.

# Relationships
- Reuses the own+public and multi-owner scoping model from [PRD-10](prd-10.md).
- Adds a `study_id` filter to the dataset browser of [PRD-10](prd-10.md).
- Shares the dictionary/key-value machinery with the other LIMS layers ([PRD-12](prd-12.md), [PRD-14](prd-14.md), [PRD-15](prd-15.md)).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-13-study-project-entity.md`
