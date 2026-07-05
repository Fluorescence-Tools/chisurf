---
type: PRD
prd: "56"
title: "PRD-56: Companion-Tool ↔ ChiSurf RPC + Phasor Overlays"
description: Make the companion photon-data exploration tool a live phasor front-end over a first-class RPC client, with ChiSurf serving phasor math and shared overlay-line geometry.
status: in-progress
phase: "unassigned"
resource: overhaul/PRD-56-ndxplorer-chisurf-rpc-phasor-overlays.md
tags: [prd, rpc, imaging]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The companion photon-data exploration tool derives columns from local formulas but cannot tap ChiSurf's domain math; the phasor toolkit of PRD-55 hands it only a static file drop. This PRD makes the companion tool a live phasor front-end by giving it a first-class, reusable RPC client (a `RpcClient` Protocol, a JSON-RPC 2.0 ZMQ client, an injected in-process client path, and a typed `PhasorService` facade) that stays free of any ChiSurf import. ChiSurf lands the phasor math in a Qt-free `analysis.py` and exposes it over a `phasor.*` service, plus overlay-geometry helpers (semicircle, iso-lifetime grid, FRET trajectory). Phasor reference lines and smFRET FRET lines share one LineSet contract so the calculator, the imaging plugin, and the companion tool all render identical geometry; a standalone AutoForm phasor calculator is added to the Calculators hub. Non-loopback exposure stays gated on network-security work.

# Status
In-progress / unassigned (STATUS TABLE authoritative). Backbone plus calculator done per memory; companion-tool GUI panel remains.

# Relationships
- Consumes and exposes the phasor math from [PRD-55](prd-55.md).
- The companion photon-data exploration tool becomes a first-class RPC client per the [RPC target](/specs/rpc.md); loopback-only until network-security work lands.
- Registers services via the [plugin system](/architecture/plugin-system.md); phasor calculator via [GUI & AutoForm](/subsystems/gui-autoform.md); related to imaging correlation [PRD-51](prd-51.md).

# Source
- Primary: `overhaul/PRD-56-ndxplorer-chisurf-rpc-phasor-overlays.md`
