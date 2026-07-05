---
type: PRD
prd: "03"
title: "PRD-03: Result Registry"
description: A single register_result() API so any plugin can archive output to MFDB with full provenance
status: in-progress
phase: "0"
resource: overhaul/PRD-03-result-registry.md
tags: [prd, mfdb, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Provides a single `register_result()` function any plugin can call to archive its
output in MFDB with full provenance — object stored, artifact and operation rows
created, input/output and derived-from / measured-sample edges wired, and
parameters recorded — without the plugin needing to understand MFDB internals.
Most plugins produce data but do not write to MFDB; this gives them a
dead-simple API and one well-tested choke point so provenance is uniform. It is
the lightweight per-plugin path that complements the whole-project archiver,
both calling the same repository primitives, and it serializes exclusively
through the payload codecs rather than ad-hoc JSON.

# Status
In progress. Blocked on and built against the payload codec layer; the verified
repository API signatures are pinned in the PRD, and a code review is on record.

# Relationships
- Blocked by / uses codecs from [PRD-030](prd-030.md); links results to samples from [PRD-02](prd-02.md).
- Reference consumer: the burst pipeline [PRD-04](prd-04.md); downstream calibration [PRD-05](prd-05.md).
- Adds a uniform write path over the [MFDB (current)](/architecture/mfdb.md) store for [plugins](/architecture/plugin-system.md); relates to [Plugins target](/specs/plugins.md).

# Source
- Primary: `overhaul/PRD-03-result-registry.md`
- Supplementary: `overhaul/CODE_REVIEW_PRD03.md`
