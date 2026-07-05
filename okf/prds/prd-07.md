---
type: PRD
prd: "07"
title: "PRD-07: Plugin MFDB Integration"
description: Have high-priority plugins register their results in MFDB via a single registration call at each output point.
status: superseded
phase: "unassigned"
resource: overhaul/PRD-07-plugin-integration.md
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD proposed adding a single `register_result()` call at the output point of each high-priority plugin (lifetime fitting, maximum entropy, anisotropy, the companion photon-data exploration tool, microtime histogram, FCS correlation, etc.) so their results land in MFDB with sample-id propagation and provenance edges, wrapped in try/except so plugins still work without MFDB. It defined priority tiers and a per-plugin recipe (find output point, register, propagate sample id, test).

# Status
Superseded by [PRD-16](prd-16.md), the strict, uniform transformer contract that replaces these ad-hoc per-plugin registration calls.

# Relationships
- Superseded by [PRD-16](prd-16.md) (uniform transformer contract).
- Its registration model is formalized by [PRD-11](prd-11.md) (operation-node abstraction in MFDB).
- Touches the [plugin system](/architecture/plugin-system.md) and [Plugins target](/specs/plugins.md).

# Source
- Primary: `overhaul/PRD-07-plugin-integration.md`
