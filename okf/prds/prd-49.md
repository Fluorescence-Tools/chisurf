---
type: PRD
prd: "49"
title: "PRD-49: Multiparameter-Fluorescence Feature Parity"
description: Roadmap PRD to reach and surpass feature parity with an established multiparameter-fluorescence analysis suite across every analysis modality, delegating implementation to per-module sub-PRDs.
status: draft
phase: "unassigned"
resource: overhaul/PRD-49-pam-feature-parity.md
tags: [prd, plugins, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
An established multiparameter-fluorescence analysis suite is the incumbent integrated toolset for quantitative fluorescence microscopy and spectroscopy — FCS/FCCS/pCF, smFRET/MFD, phasor-FLIM, and image correlation. This roadmap PRD gives an honest method-by-method parity matrix, enumerates the real gaps, and lays out a phased plan that spawns per-module sub-PRDs ([PRD-50](prd-50.md) through [PRD-54](prd-54.md)). It decides what and in what order, delegating how to the sub-PRDs; it changes no code itself. A hard cross-cutting mandate: every new model and analysis UI ships as a declarative model plus `view.json` rendered by AutoForm — never hand-written Qt — so each module is headless-testable and provenance-tracked.

# Status
Draft / unassigned (STATUS TABLE authoritative). Roadmap only; authorizes the sub-PRDs and a rolling backlog (2-photon FCS, 3c-MFD/2CDE filters, native image-format import).

# Relationships
- Parent of children [PRD-50](prd-50.md), [PRD-51](prd-51.md), [PRD-52](prd-52.md), [PRD-53](prd-53.md), [PRD-54](prd-54.md).
- Mandates the model/view-spec split rendered through [GUI & AutoForm](/subsystems/gui-autoform.md).
- Leans on ChiSurf differentiators over the incumbent: MFDB provenance ([MFDB (current)](/architecture/mfdb.md)), structural-FRET, and headless/server modes ([Core target](/specs/core.md)).

# Source
- Primary: `overhaul/PRD-49-pam-feature-parity.md`
