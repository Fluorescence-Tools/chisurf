---
type: Reference
title: Roadmap PRDs
description: Numbered PRD design notes in the okf/prds group that drive current work.
resource: okf/prds/
tags: [reference, roadmap, prd, design]
timestamp: '2026-07-06T00:00:00Z'
---

# Reference

The [prds group](/prds/index.md) holds the numbered PRD design notes
(`prd-NN.md`, one self-contained concept per PRD with status/phase frontmatter)
that drive current work. Consult the relevant PRD before large changes in its
area — e.g. [PRD-40](/prds/prd-40.md) (model/UI split feeding
[AutoForm](/subsystems/gui-autoform.md)), [PRD-43](/prds/prd-43.md) /
[PRD-44](/prds/prd-44.md) (history as an [MFDB](/architecture/mfdb.md) projection),
and [PRD-48](/prds/prd-48.md) (provider-agnostic ELN gateway). The authoritative
implementation ordering is [PRD Implementation Order](/prds/master-order.md).

These PRDs were formerly the top-level `overhaul/` folder, now retired into the
bundle. Complementary clean-architecture targets live in this bundle's
[specs group](/specs/index.md), with the current-state gap captured in the
[assessment backlog](/specs/assessment.md).
