---
type: PRD
prd: "16"
title: "PRD-16: General Transformer Contract"
description: Defines one uniform contract every data-transformer plugin must obey — typed ports, dictionary-declared parameters, a pure transform, and uniform provenance registration.
status: done
phase: "2"
resource: overhaul/PRD-16-transformer-contract.md
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-16 defines a single transformer abstraction that every data-transformer plugin (burst selection, microtime shifter, background correction, correlation, and others) must satisfy, replacing ad-hoc per-plugin structure. A conformant transformer declares typed input and output ports, reads its parameter schema from the `.dic` dictionary rather than hardcoding it, exposes a pure `transform` function free of Qt and database imports, and registers uniformly as an operation node. Invocations are serializable spec values, which makes them recordable, replayable, and composable. A registry enables discovery, and a parametrised conformance test gates any new transformer.

# Status
Done (phase 2, STATUS TABLE authoritative). The reference transformers (burst selection, microtime shifter) conform, and the conformance test gates new transformers.

# Relationships
- Plugin-side counterpart of [PRD-11](prd-11.md), which maps transformers to MFDB operation nodes and defines the parameter schema storage.
- Reference transformers derive from [PRD-04](prd-04.md) and [PRD-09](prd-09.md).
- A transformer execution may reference a protocol/version per [PRD-14](prd-14.md).
- Serializable invocations feed [PRD-21](prd-21.md) (replayable compute spec) and [PRD-22](prd-22.md) (pipeline composition).
- GUI half enforced by [PRD-23](prd-23.md).
- Aligns with the [Plugins target](/specs/plugins.md) and [Core target](/specs/core.md); registration routes through the [action layer](/architecture/action-layer.md).

# Source
- Primary: `overhaul/PRD-16-transformer-contract.md`
