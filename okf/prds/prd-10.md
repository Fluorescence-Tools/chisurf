---
type: PRD
prd: "10"
title: "PRD-10: MFDB Dataset Browser Widget"
description: A reusable Qt widget to pick a registered MFDB dataset, with scope/visibility, server-side filtering, co-ownership, and file groups.
status: in-progress
phase: "0"
resource: overhaul/PRD-10-dataset-browser-widget.md
tags: [prd, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
A reusable Qt widget lets users pick a registered MFDB dataset instead of hunting for a file on disk. It lists the active user's datasets (with Mine/Public/All scope), offers debounced server-side search and structured filters with pagination, and returns a typed `DatasetSelection` any plugin can resolve to a local path via an RPC. Object-store dedup is reconciled with a many-to-many ownership model (`mfdb_artifact_owner`) so co-owners all see a shared dataset under "Mine." Multi-file datasets are meant to register as one dataset via a dictionary-declared file-group membership table (`mfdb_artifact_member`), which is not yet implemented. The microtime shifter is the first consumer and end-to-end acceptance case. A companion bug note captures silent registration failures (unknown operation-type vocabulary; a missing active-user row breaking the owner foreign key) that caused processed datasets to never appear.

# Status
In-progress (re-verified against code 2026-07-05). Landed: the reusable browser + picker dialog, `datasets.browse`/`datasets.open` RPC, the co-ownership model (`mfdb_artifact_owner`) with Mine/Public/All scope and pagination, and the shifter integration. **Open:** the multi-file **file-group** membership schema (`mfdb_artifact_member`) is absent from the tree, so multi-file datasets are not yet registered/opened as one — this deliverable remains reopened.

# Relationships
- Consumed first by [PRD-09](prd-09.md) (microtime shifter).
- Ownership/visibility model reused by the study entity in [PRD-13](prd-13.md).
- Gains a study facet from [PRD-13](prd-13.md) (`browse_datasets` `study_id` filter).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-10-dataset-browser-widget.md`
- Supplementary: `overhaul/PRD-10-bug-processed-not-registered.md`
